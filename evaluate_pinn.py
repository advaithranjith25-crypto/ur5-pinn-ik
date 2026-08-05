"""
Quick evaluation: load the best checkpoint, run it on the ID test set and
the OOD test set separately, and report human-interpretable numbers --
position error in millimeters, orientation error in degrees -- instead of
the abstract loss units used during training. This is the number worth
putting in your report, not the raw loss value.
"""

import numpy as np
import torch

from pinn_model import IKPinn
from fk_ur5 import forward_kinematics, rotation_matrix_to_quaternion

CHECKPOINT_PATH = "pinn_ur5_best.pt"
DATASET_PATH = "ur5_ik_dataset.npz"


def quat_angle_error_deg(q1, q2):
    """Geodesic angle between two batches of quaternions, in degrees."""
    dot = (q1 * q2).sum(dim=-1).abs().clamp(-1.0, 1.0)  # abs handles q/-q sign ambiguity
    angle_rad = 2 * torch.acos(dot)
    return torch.rad2deg(angle_rad)


def evaluate(model, pos, quat, split_name):
    with torch.no_grad():
        q_pred = model(pos, quat)
        pos_pred, R_pred, _ = forward_kinematics(q_pred)
        quat_pred = rotation_matrix_to_quaternion(R_pred)

    pos_err_mm = (pos_pred - pos).norm(dim=-1) * 1000.0
    orient_err_deg = quat_angle_error_deg(quat_pred, quat)

    print(f"\n--- {split_name} (n={pos.shape[0]}) ---")
    print(f"Position error (mm):   mean={pos_err_mm.mean():.2f}  median={pos_err_mm.median():.2f}  "
          f"p90={pos_err_mm.quantile(0.9):.2f}  max={pos_err_mm.max():.2f}")
    print(f"Orientation error (deg): mean={orient_err_deg.mean():.2f}  median={orient_err_deg.median():.2f}  "
          f"p90={orient_err_deg.quantile(0.9):.2f}  max={orient_err_deg.max():.2f}")


if __name__ == "__main__":
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = IKPinn(hidden_dim=512, n_hidden_layers=6)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_fk={ckpt['val_fk']:.6f})")

    d = np.load(DATASET_PATH)

    for split in ["test_id", "test_ood"]:
        pos = torch.tensor(d[f"{split}_pos"], dtype=torch.float32)
        quat = torch.tensor(d[f"{split}_quat"], dtype=torch.float32)
        label = "ID Test Set" if split == "test_id" else "OOD Test Set (held-out region)"
        evaluate(model, pos, quat, label)

    print("\nNote: OOD error should be visibly worse than ID error -- that's expected and "
          "is exactly the gap the KD-tree confidence gating (Phase 3) is designed to catch.")
