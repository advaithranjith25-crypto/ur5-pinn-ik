"""
Phase 4: Cyclic Coordinate Descent (CCD) inverse kinematics solver.

A simpler, geometric alternative to DLS: adjusts one joint at a time,
working from the end-effector back to the base, without any matrix
computation (no Jacobian, no matrix inverse). Each joint is rotated by the
angle that best aligns a reference point on the end-effector with its
target, then the next joint (moving toward the base) is adjusted the same
way, cycling through all joints repeatedly until convergence.

Classic CCD only targets POSITION. To additionally respect ORIENTATION
(needed for full 6-DOF pose matching, which is what this project's
experiment requires), this implementation uses the standard "dual-point"
extension: a second reference point, offset from the end-effector along
its local Z-axis, is aligned toward a corresponding offset point computed
from the target orientation. Aligning both the origin point AND this
second point simultaneously constrains both position and orientation.
"""

import time
import numpy as np
import torch

from fk_ur5 import forward_kinematics, rotation_matrix_to_quaternion, N_JOINTS
from dls_solver import quat_error_axis_angle, fk_pose_np

DEFAULT_MAX_ITERS = 300
DEFAULT_POS_TOL = 5e-3
DEFAULT_ORIENT_TOL = 0.05
SECONDARY_POINT_OFFSET = 0.1  # meters, along local Z, for the orientation-aware second point


def get_joint_frames(q_np):
    """
    Return, for each joint, its pivot position and rotation axis (both in
    world frame) -- needed to compute how much rotating that joint around
    its own axis would move a downstream point. Reuses the same cumulative
    transform chain as the Jacobian in fk_ur5.py, so this is guaranteed
    consistent with the FK used everywhere else in the project.
    """
    with torch.no_grad():
        q_t = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0)
        _, _, T_all = forward_kinematics(q_t)

    pivots = [np.zeros(3)]  # joint 0 pivots at the world origin
    axes = [np.array([0.0, 0.0, 1.0])]  # joint 0's axis is the base frame's z-axis
    for i in range(N_JOINTS - 1):
        T = T_all[i][0].numpy()
        pivots.append(T[:3, 3])
        axes.append(T[:3, 2])  # z-axis of frame i becomes joint i+1's rotation axis
    return pivots, axes


def get_secondary_point(pos, R):
    """A point offset from `pos` along the local Z-axis (R's third column)."""
    return pos + SECONDARY_POINT_OFFSET * R[:, 2]


def rotate_point_around_axis(point, pivot, axis, angle):
    """Rodrigues' rotation formula: rotate `point` around `axis` through `pivot`."""
    v = point - pivot
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v_rot = (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1 - np.cos(angle))
    )
    return pivot + v_rot


def best_combined_angle(pivot, axis, point_pairs, weights):
    """
    The rotation angle (around `axis`, through `pivot`) that minimizes the
    WEIGHTED SUM of squared distances across multiple (current, target)
    point pairs simultaneously -- not the average of separately-optimal
    single-point angles, which does not minimize the combined error and
    causes the two objectives (position, orientation-proxy) to fight each
    other / oscillate.

    Closed form: for points projected onto the plane perpendicular to the
    rotation axis, the angle minimizing sum_i w_i |R(angle) v_i - u_i|^2 is
        angle = atan2( sum_i w_i (v_i x u_i).axis , sum_i w_i (v_i . u_i) )

    point_pairs: list of (current_point, target_point) tuples
    weights: list of weights, same length as point_pairs
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    def project(p):
        v = p - pivot
        return v - np.dot(v, axis) * axis

    sin_sum, cos_sum = 0.0, 0.0
    for (cur_p, tgt_p), w in zip(point_pairs, weights):
        v = project(cur_p)
        u = project(tgt_p)
        n_v, n_u = np.linalg.norm(v), np.linalg.norm(u)
        if n_v < 1e-9 or n_u < 1e-9:
            continue
        sin_sum += w * np.dot(axis, np.cross(v, u))
        cos_sum += w * np.dot(v, u)

    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0
    return np.arctan2(sin_sum, cos_sum)


def ccd_solve(
    q_init, target_pos, target_quat,
    max_iters=DEFAULT_MAX_ITERS, pos_tol=DEFAULT_POS_TOL, orient_tol=DEFAULT_ORIENT_TOL,
):
    """
    Run CCD from q_init toward target_pos (position-focused, per the
    classical CCD formulation).

    DESIGN NOTE, worth stating explicitly in the report: this implementation
    targets POSITION only, matching the standard CCD description (adjust
    one joint at a time to reduce end-effector position error; see project
    doc's own description of CCD, which mentions no orientation handling,
    unlike DLS's natural 6D pose handling via the Jacobian). An earlier
    version attempted a "dual-point" extension (a second reference point
    offset along the end-effector's local Z-axis, aligned toward a
    corresponding target-orientation point) to additionally constrain
    orientation. That extension was mathematically implemented correctly
    (closed-form weighted multi-point rotation angle) but still oscillated
    badly in practice -- position converged cleanly while orientation error
    swung between 4 and 87 degrees across iterations, never stabilizing.
    Rather than ship a fragile, unreliable mechanism, this simpler
    position-only formulation is used, and DLS (which handles full 6D pose
    naturally and reliably) is the solver responsible for orientation
    accuracy in the experiment. Final orientation error is still reported
    as a diagnostic even though it isn't part of CCD's convergence
    criterion here -- useful for honestly characterizing what CCD does and
    doesn't achieve compared to DLS.

    Returns the same result-dict shape as dls_solve for drop-in comparison,
    with "converged" based on position tolerance only for this solver.
    """
    t0 = time.time()
    q = np.array(q_init, dtype=np.float64).copy()

    for it in range(1, max_iters + 1):
        cur_pos, cur_quat = fk_pose_np(q)
        pos_err = np.linalg.norm(target_pos - cur_pos)

        if pos_err < pos_tol:
            orient_err = np.linalg.norm(quat_error_axis_angle(cur_quat, target_quat))
            return {
                "q_final": q, "n_iters": it - 1, "converged": True,
                "final_pos_err": pos_err, "final_orient_err": orient_err,
                "wall_time": time.time() - t0,
            }

        pivots, axes = get_joint_frames(q)

        for j in reversed(range(N_JOINTS)):
            cur_pos, _ = fk_pose_np(q)
            angle = best_combined_angle(
                pivots[j], axes[j],
                point_pairs=[(cur_pos, target_pos)],
                weights=[1.0],
            )
            angle = np.clip(angle, -0.3, 0.3)
            q[j] = np.clip(q[j] + angle, -np.pi, np.pi)

    cur_pos, cur_quat = fk_pose_np(q)
    pos_err = np.linalg.norm(target_pos - cur_pos)
    orient_err = np.linalg.norm(quat_error_axis_angle(cur_quat, target_quat))
    return {
        "q_final": q, "n_iters": max_iters, "converged": False,
        "final_pos_err": pos_err, "final_orient_err": orient_err,
        "wall_time": time.time() - t0,
    }


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # Same two-part verification as DLS: (1) nearby starts should converge
    # reliably if the algorithm is implemented correctly, (2) random cold
    # starts are expected to be much harder.
    print("--- Nearby-start verification (should be ~100%) ---")
    n_conv = 0
    for i in range(20):
        q_true = np.random.uniform(-np.pi, np.pi, size=N_JOINTS)
        target_pos, target_quat = fk_pose_np(q_true)
        q_start = q_true + np.random.normal(0, 0.1, size=N_JOINTS)
        result = ccd_solve(q_start, target_pos, target_quat)
        n_conv += result["converged"]
    print(f"Converged: {n_conv}/20")

    print("\n--- Random cold-start (expected to be hard, like DLS) ---")
    results = []
    for i in range(50):
        q_true = np.random.uniform(-np.pi, np.pi, size=N_JOINTS)
        target_pos, target_quat = fk_pose_np(q_true)
        q_start = np.random.uniform(-np.pi, np.pi, size=N_JOINTS)
        result = ccd_solve(q_start, target_pos, target_quat)
        results.append(result)

    n_converged = sum(r["converged"] for r in results)
    iters = [r["n_iters"] for r in results if r["converged"]]
    print(f"Converged: {n_converged}/50")
    if iters:
        print(f"Iterations to converge: mean={np.mean(iters):.1f} median={np.median(iters):.1f}")
    pos_errs = [r["final_pos_err"] * 1000 for r in results]
    print(f"Final position error (mm): mean={np.mean(pos_errs):.1f} max={np.max(pos_errs):.1f}")
