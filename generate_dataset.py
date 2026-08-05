"""
Phase 1: Dataset generation for the PINN-warm-start IK project.

Produces (target_pose -> ground_truth_q) pairs by sampling joint configs
and running them through MuJoCo forward kinematics (the same ur5_from_dh.xml
verified against the PyTorch FK in verify_fk.py).

Three strata, drawn in this order:
  1. GENERAL       - uniform random q across full joint range
  2. NEAR-SINGULAR - biased toward configs where the Jacobian's smallest
                      singular value is small (arm near-extended, wrist-flip)
  3. EDGE          - biased toward configs near joint limits

Plus a held-out OOD set defined directly in POSE SPACE (the network's actual
input), not joint space: end-effector positions with x > 0.15m AND
y > 0.15m (a spatial corner covering ~9% of the reachable workspace,
verified empirically against the generated data) are excluded from all ID
generation and used exclusively for the OOD test set.

IMPORTANT DESIGN NOTE, worth including in the report: an earlier version of
this holdout restricted a JOINT (or pair of joints) instead of a pose
region, on the assumption that one would imply the other. That assumption
turned out to be false for UR5 specifically -- correlation between base
joint angle and end-effector azimuth was measured at ~0.10, essentially
uncorrelated, because UR5's nonzero d4 offset (the "elbow offset" that
makes it a non-spherical-wrist manipulator) means many different joint
combinations can reach the same pose region. A joint-space hole does not
reliably create a pose-space hole on this robot. Defining the holdout
directly on the network's actual input (pose), rather than an upstream
generative variable (joints), is the change that fixed this.

Output: a single .npz file with arrays for each split, plus per-sample
metadata (is_near_singular, is_edge) so later evaluation can stratify by
category without recomputing anything.
"""

import numpy as np
import torch
import mujoco

from fk_ur5 import N_JOINTS, jacobian

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_GENERAL = 14000       # scaled down 10x from the 200k spec for a fast local
N_NEAR_SINGULAR = 4000  # first run -- see note at the bottom on scaling back up
N_EDGE = 2000
N_OOD = 2000

JOINT_LOW = -np.pi * np.ones(N_JOINTS)
JOINT_HIGH = np.pi * np.ones(N_JOINTS)

# OOD holdout: a spatial corner of the reachable workspace, defined on the
# end-effector POSITION directly (not on any joint). ~9% of samples fall
# here naturally, verified against generated data -- big enough for a
# meaningful test set, small enough to leave most of the workspace for
# training.
POSE_CORNER_X = 0.15
POSE_CORNER_Y = 0.15

SINGULARITY_THRESHOLD = 0.05   # smallest singular value below this -> "near-singular"
EDGE_MARGIN = 0.15             # radians from a joint limit -> "edge"

RNG_SEED = 0

model = mujoco.MjModel.from_xml_path("ur5_from_dh.xml")
data = mujoco.MjData(model)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")


def mujoco_fk_pose(q_np):
    """Return (pos[3], quat_wxyz[4]) for a single joint config via MuJoCo."""
    data.qpos[:N_JOINTS] = q_np
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[ee_site_id].copy()
    R = data.site_xmat[ee_site_id].copy().reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return pos, quat


def min_singular_value(q_np):
    """Smallest singular value of the 6x6 Jacobian at this config."""
    q_t = torch.tensor(q_np, dtype=torch.float32)
    J = jacobian(q_t)
    sv = torch.linalg.svdvals(J)
    return sv.min().item()


def near_joint_limit(q_np, margin=EDGE_MARGIN):
    return np.any(q_np < (JOINT_LOW + margin)) or np.any(q_np > (JOINT_HIGH - margin))


def in_ood_corner(pos):
    """True if this end-effector position sits in the held-out spatial corner."""
    return (pos[0] > POSE_CORNER_X) and (pos[1] > POSE_CORNER_Y)


def sample_general(n, rng, id_only=True, max_tries=50):
    """
    Uniform random q, rejecting (redrawing) any candidate whose resulting
    pose falls in the OOD corner when id_only=True.
    """
    out = []
    while len(out) < n:
        cand = rng.uniform(JOINT_LOW, JOINT_HIGH)
        if id_only:
            pos, _ = mujoco_fk_pose(cand)
            if in_ood_corner(pos):
                continue
        out.append(cand)
    return np.array(out)


def sample_ood(n, rng):
    """Rejection-sample q whose resulting pose DOES fall in the OOD corner."""
    out = []
    while len(out) < n:
        cand = rng.uniform(JOINT_LOW, JOINT_HIGH)
        pos, _ = mujoco_fk_pose(cand)
        if in_ood_corner(pos):
            out.append(cand)
    return np.array(out)


def sample_near_singular(n, rng, max_tries_per_sample=200):
    """
    Rejection-sample toward small min-singular-value configs, excluding the
    OOD pose corner from the ID (near-singular) training pool.
    """
    out = []
    while len(out) < n:
        cand = rng.uniform(JOINT_LOW, JOINT_HIGH)
        # bias the elbow (joint index 2) toward 0 -> arm-straight singularity
        cand[2] = np.clip(rng.normal(0, 0.08), JOINT_LOW[2], JOINT_HIGH[2])

        pos, _ = mujoco_fk_pose(cand)
        if in_ood_corner(pos):
            continue  # keep the OOD corner exclusively for the OOD test set

        sv = min_singular_value(cand)
        if sv < SINGULARITY_THRESHOLD:
            out.append(cand)
    return np.array(out)


def sample_edge(n, rng):
    out = []
    while len(out) < n:
        cand = rng.uniform(JOINT_LOW, JOINT_HIGH)
        j = rng.integers(0, N_JOINTS)
        sign = rng.choice([-1, 1])
        cand[j] = sign * rng.uniform(np.pi - EDGE_MARGIN, np.pi)

        pos, _ = mujoco_fk_pose(cand)
        if in_ood_corner(pos):
            continue

        if near_joint_limit(cand):
            out.append(cand)
    return np.array(out)


def build_split(q_array, near_singular_flag=False, edge_flag=False):
    n = q_array.shape[0]
    positions = np.zeros((n, 3))
    quats = np.zeros((n, 4))
    min_svs = np.zeros(n)

    for i in range(n):
        pos, quat = mujoco_fk_pose(q_array[i])
        positions[i] = pos
        quats[i] = quat
        min_svs[i] = min_singular_value(q_array[i])

    return {
        "q": q_array.astype(np.float32),
        "pos": positions.astype(np.float32),
        "quat": quats.astype(np.float32),
        "min_singular_value": min_svs.astype(np.float32),
        "is_near_singular": (min_svs < SINGULARITY_THRESHOLD),
        "is_edge": np.array([near_joint_limit(q) for q in q_array]),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)

    print(f"Generating {N_GENERAL} general samples...")
    q_general = sample_general(N_GENERAL, rng, id_only=True)

    print(f"Generating {N_NEAR_SINGULAR} near-singular samples "
          f"(rejection sampling, may take a moment)...")
    q_singular = sample_near_singular(N_NEAR_SINGULAR, rng)

    print(f"Generating {N_EDGE} workspace-edge samples...")
    q_edge = sample_edge(N_EDGE, rng)

    print(f"Generating {N_OOD} OOD samples "
          f"(pose corner: x>{POSE_CORNER_X} AND y>{POSE_CORNER_Y}, rejection sampling)...")
    q_ood = sample_ood(N_OOD, rng)

    print("Running FK + Jacobian over all samples to build final dataset...")
    id_q = np.concatenate([q_general, q_singular, q_edge], axis=0)
    rng.shuffle(id_q)

    id_data = build_split(id_q)
    ood_data = build_split(q_ood)

    # Train/val/test split on the ID data (OOD is entirely its own test set)
    n_id = id_q.shape[0]
    n_train = int(0.85 * n_id)
    n_val = int(0.05 * n_id)

    def slice_dict(d, s, e):
        return {k: v[s:e] for k, v in d.items()}

    train = slice_dict(id_data, 0, n_train)
    val = slice_dict(id_data, n_train, n_train + n_val)
    test_id = slice_dict(id_data, n_train + n_val, n_id)

    np.savez(
        "ur5_ik_dataset.npz",
        train_q=train["q"], train_pos=train["pos"], train_quat=train["quat"],
        train_min_sv=train["min_singular_value"],
        train_is_near_singular=train["is_near_singular"], train_is_edge=train["is_edge"],

        val_q=val["q"], val_pos=val["pos"], val_quat=val["quat"],
        val_min_sv=val["min_singular_value"],
        val_is_near_singular=val["is_near_singular"], val_is_edge=val["is_edge"],

        test_id_q=test_id["q"], test_id_pos=test_id["pos"], test_id_quat=test_id["quat"],
        test_id_min_sv=test_id["min_singular_value"],
        test_id_is_near_singular=test_id["is_near_singular"], test_id_is_edge=test_id["is_edge"],

        test_ood_q=ood_data["q"], test_ood_pos=ood_data["pos"], test_ood_quat=ood_data["quat"],
        test_ood_min_sv=ood_data["min_singular_value"],
        test_ood_is_near_singular=ood_data["is_near_singular"], test_ood_is_edge=ood_data["is_edge"],

        pose_corner_x=POSE_CORNER_X, pose_corner_y=POSE_CORNER_Y,
    )

    print("\n--- Dataset summary ---")
    print(f"Train: {n_train}  Val: {n_val}  Test (ID): {n_id - n_train - n_val}  Test (OOD): {N_OOD}")
    print(f"Near-singular fraction in train: {train['is_near_singular'].mean():.3f}")
    print(f"Edge fraction in train: {train['is_edge'].mean():.3f}")
    print("Saved to ur5_ik_dataset.npz")
