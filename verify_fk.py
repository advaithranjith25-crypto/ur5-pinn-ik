"""
Cross-check: does our PyTorch differentiable FK agree with MuJoCo?

This is the single most important sanity check in the whole project. If
this passes, the PyTorch FK can be trusted as the physics-consistency loss
target and Jacobian source; MuJoCo can be trusted as the dataset oracle.
If it fails, nothing downstream (dataset, PINN loss, DLS Jacobian) can be
trusted until it's fixed.
"""

import numpy as np
import torch
import mujoco

from fk_ur5 import forward_kinematics, rotation_matrix_to_quaternion, N_JOINTS

N_TEST = 200
POS_TOL = 1e-4   # meters
QUAT_TOL = 1e-3  # allow sign-flip ambiguity (q and -q are the same rotation)

model = mujoco.MjModel.from_xml_path("ur5_from_dh.xml")
data = mujoco.MjData(model)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")


def mujoco_fk(q_np):
    """Run MuJoCo FK for a single joint config, return (pos, quat_wxyz)."""
    data.qpos[:N_JOINTS] = q_np
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[ee_site_id].copy()

    R = data.site_xmat[ee_site_id].copy().reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return pos, quat


def quat_geodesic_close(q1, q2, tol):
    """Quaternions q and -q represent the same rotation, so check both."""
    d1 = np.linalg.norm(q1 - q2)
    d2 = np.linalg.norm(q1 + q2)
    return min(d1, d2) < tol


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    q_test = (torch.rand(N_TEST, N_JOINTS) * 2 - 1) * torch.pi

    pos_torch, R_torch, _ = forward_kinematics(q_test)
    quat_torch = rotation_matrix_to_quaternion(R_torch)

    pos_errors = []
    quat_failures = 0
    pos_failures = 0

    for i in range(N_TEST):
        q_np = q_test[i].numpy()
        pos_mj, quat_mj = mujoco_fk(q_np)

        pos_pt = pos_torch[i].detach().numpy()
        quat_pt = quat_torch[i].detach().numpy()

        pos_err = np.linalg.norm(pos_mj - pos_pt)
        pos_errors.append(pos_err)

        if pos_err > POS_TOL:
            pos_failures += 1
            print(f"[POS MISMATCH] sample {i}: err={pos_err:.6f}  "
                  f"mujoco={pos_mj}  torch={pos_pt}")

        if not quat_geodesic_close(quat_mj, quat_pt, QUAT_TOL):
            quat_failures += 1
            print(f"[QUAT MISMATCH] sample {i}: mujoco={quat_mj}  torch={quat_pt}")

    pos_errors = np.array(pos_errors)
    print("\n--- Summary ---")
    print(f"Tested {N_TEST} random joint configurations")
    print(f"Position error:  mean={pos_errors.mean():.8f}  max={pos_errors.max():.8f}  (tol={POS_TOL})")
    print(f"Position failures: {pos_failures}/{N_TEST}")
    print(f"Quaternion failures: {quat_failures}/{N_TEST}")

    if pos_failures == 0 and quat_failures == 0:
        print("\nPASS: PyTorch FK matches MuJoCo FK within tolerance.")
    else:
        print("\nFAIL: investigate DH parameter signs/order before proceeding.")
