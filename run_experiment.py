"""
Phase 5: The central experiment.

Compares three initialization strategies feeding into two classical IK
refiners (DLS, CCD), evaluated across three pose categories -- this is the
actual test of the project's thesis: does a physics-informed learned
warm-start, gated by a confidence check against the training distribution,
reduce iterations-to-converge compared to a cold start, while avoiding the
failure mode of blindly trusting the network on unfamiliar poses?

Initialization strategies:
  1. cold-start:  q_init = zeros(6)          -- the baseline every other
                                                  strategy is measured against
  2. pinn:        q_init = PINN(pose)         -- naive, unconditional learned warm-start
  3. gated:       q_init = PINN(pose) if the KD-tree gate is confident,
                  else the nearest training example's q               -- proposed method

Pose categories (drawn from the dataset's existing stratification):
  - general:        test_id poses, excluding near-singular and edge flags
  - near_singular:  test_id poses flagged is_near_singular
  - ood:            test_ood poses (the held-out spatial corner)

For each (strategy, solver, category) cell, N poses are solved and the
following are recorded per solve: iterations to converge, wall-clock time,
success (converged within max_iters), final position/orientation error.
Results are saved to a CSV for analyze_results.py to consume.
"""

import time
import numpy as np
import torch
import pandas as pd

from pinn_model import IKPinn
from confidence_gate import ConfidenceGate, COLD_START_Q
from dls_solver import dls_solve, fk_pose_np
from ccd_solver import ccd_solve
from fk_ur5 import N_JOINTS

DATASET_PATH = "ur5_ik_dataset.npz"
CHECKPOINT_PATH = "pinn_ur5_best.pt"
OUTPUT_CSV = "experiment_results.csv"

N_PER_CATEGORY = 300  # poses solved per (strategy, solver, category) cell
                       # -- 300*3*2*3 = 5400 total solves. Verified at N=60
                       # (1080 solves, ~4 minutes) that this runs correctly;
                       # N=300 should take roughly 20 minutes total.

RNG_SEED = 0


def load_everything():
    d = np.load(DATASET_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = IKPinn(hidden_dim=512, n_hidden_layers=6)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gate = ConfidenceGate(d["train_pos"], d["train_quat"], d["train_q"])
    gate.calibrate(d["val_pos"], d["val_quat"], d["test_ood_pos"], d["test_ood_quat"], method="youden")

    return d, model, gate


def build_category_pools(d, rng):
    """
    Slice out the three pose categories from the existing dataset splits,
    each capped/sampled down to N_PER_CATEGORY for a balanced experiment.
    Returns dict: category_name -> (pos array, quat array).
    """
    test_id_is_ns = d["test_id_is_near_singular"]
    test_id_is_edge = d["test_id_is_edge"]

    general_mask = (~test_id_is_ns) & (~test_id_is_edge)
    general_idx = np.where(general_mask)[0]
    ns_idx = np.where(test_id_is_ns)[0]

    def sample_idx(idx_pool, n):
        n = min(n, len(idx_pool))
        return rng.choice(idx_pool, size=n, replace=False)

    general_sel = sample_idx(general_idx, N_PER_CATEGORY)
    ns_sel = sample_idx(ns_idx, N_PER_CATEGORY)
    ood_sel = sample_idx(np.arange(len(d["test_ood_pos"])), N_PER_CATEGORY)

    return {
        "general": (d["test_id_pos"][general_sel], d["test_id_quat"][general_sel]),
        "near_singular": (d["test_id_pos"][ns_sel], d["test_id_quat"][ns_sel]),
        "ood": (d["test_ood_pos"][ood_sel], d["test_ood_quat"][ood_sel]),
    }


def get_warm_starts(pos, quat, model, gate):
    """Compute all three initialization strategies for a batch of poses at once."""
    with torch.no_grad():
        q_pinn = model(
            torch.tensor(pos, dtype=torch.float32), torch.tensor(quat, dtype=torch.float32)
        ).numpy()

    dist, nn_q = gate.nn_distance_and_q(pos, quat)
    confident = dist <= gate.threshold
    q_gated = np.where(confident[:, None], q_pinn, nn_q)

    q_cold = np.tile(COLD_START_Q, (pos.shape[0], 1))

    return {"cold": q_cold, "pinn": q_pinn, "gated": q_gated}


def run_solver_batch(solver_fn, q_inits, target_pos, target_quat):
    """Run a solver over a batch of (q_init, target) pairs, return list of result dicts."""
    results = []
    for i in range(q_inits.shape[0]):
        r = solver_fn(q_inits[i], target_pos[i], target_quat[i])
        results.append(r)
    return results


if __name__ == "__main__":
    print("Loading model, dataset, and calibrating gate...")
    d, model, gate = load_everything()

    rng = np.random.default_rng(RNG_SEED)
    categories = build_category_pools(d, rng)

    solvers = {"DLS": dls_solve, "CCD": ccd_solve}

    rows = []
    total_start = time.time()

    for cat_name, (pos, quat) in categories.items():
        print(f"\n=== Category: {cat_name} (n={pos.shape[0]}) ===")
        warm_starts = get_warm_starts(pos, quat, model, gate)

        for strategy_name, q_inits in warm_starts.items():
            for solver_name, solver_fn in solvers.items():
                t0 = time.time()
                results = run_solver_batch(solver_fn, q_inits, pos, quat)
                elapsed = time.time() - t0

                n_converged = sum(r["converged"] for r in results)
                print(f"  {strategy_name:6s} + {solver_name}: {n_converged}/{len(results)} converged, "
                      f"{elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/solve)")

                for r in results:
                    rows.append({
                        "category": cat_name, "strategy": strategy_name, "solver": solver_name,
                        "n_iters": r["n_iters"], "converged": r["converged"],
                        "wall_time": r["wall_time"],
                        "final_pos_err_mm": r["final_pos_err"] * 1000,
                        "final_orient_err_deg": np.degrees(r["final_orient_err"]),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed/60:.1f} minutes")
    print(f"Saved {len(df)} solve records to {OUTPUT_CSV}")
