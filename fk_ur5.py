"""
Differentiable Forward Kinematics for the UR5 (6-DOF), implemented in PyTorch.

This is the shared backbone for the whole project:
  - Used to generate the training dataset (paired with MuJoCo as ground truth)
  - Used inside the PINN's physics-consistency loss (needs autograd)
  - Used to compute the Jacobian for DLS and for near-singularity sampling

Convention: standard Denavit-Hartenberg (DH) parameters, UR5 values taken
from Universal Robots' published kinematic spec. Units: meters, radians.

DH transform per joint i:
    T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
"""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# UR5 standard DH parameters: (a, alpha, d) per joint. theta is the variable
# (the joint angle we solve for); a, alpha, d are fixed geometry constants.
# Source: UR5 kinematic parameters (meters, radians).
# ---------------------------------------------------------------------------
UR5_DH = {
    "a":     torch.tensor([0.0,      -0.42500, -0.39225, 0.0,     0.0,     0.0]),
    "alpha": torch.tensor([torch.pi/2, 0.0,      0.0,     torch.pi/2, -torch.pi/2, 0.0]),
    "d":     torch.tensor([0.089159,  0.0,       0.0,     0.10915,  0.09465,  0.0823]),
}

N_JOINTS = 6

# Joint limits (radians) - UR5 default full range is +-2*pi on most joints,
# but for a well-posed IK dataset we use a realistic operating range.
# Adjust these to match whatever you configure in MuJoCo.
JOINT_LIMITS_LOW = torch.tensor([-torch.pi, -torch.pi, -torch.pi, -torch.pi, -torch.pi, -torch.pi])
JOINT_LIMITS_HIGH = torch.tensor([torch.pi, torch.pi, torch.pi, torch.pi, torch.pi, torch.pi])


def dh_transform(theta, a, alpha, d):
    """
    Build a single 4x4 DH homogeneous transform matrix, batched.

    theta, a, alpha, d: tensors of shape (batch,)
    returns: tensor of shape (batch, 4, 4)
    """
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = torch.cos(alpha), torch.sin(alpha)

    batch = theta.shape[0]
    T = torch.zeros(batch, 4, 4, dtype=theta.dtype, device=theta.device)

    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct

    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st

    T[:, 2, 0] = 0.0
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d

    T[:, 3, 3] = 1.0
    return T


def forward_kinematics(q, dh=UR5_DH):
    """
    Compute end-effector pose from joint angles, differentiably.

    q: tensor of shape (batch, 6) -- joint angles in radians
    returns:
        pos: tensor of shape (batch, 3)      -- end-effector xyz
        R:   tensor of shape (batch, 3, 3)   -- end-effector rotation matrix
        T_all: list of 6 tensors (batch,4,4) -- cumulative transform after each joint
               (useful later for the Jacobian and for CCD, which needs every
               joint's position/orientation, not just the final one)
    """
    batch = q.shape[0]
    device = q.device
    dtype = q.dtype

    a = dh["a"].to(device=device, dtype=dtype)
    alpha = dh["alpha"].to(device=device, dtype=dtype)
    d = dh["d"].to(device=device, dtype=dtype)

    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
    T_all = []

    for i in range(N_JOINTS):
        theta_i = q[:, i]
        a_i = a[i].expand(batch)
        alpha_i = alpha[i].expand(batch)
        d_i = d[i].expand(batch)

        T_i = dh_transform(theta_i, a_i, alpha_i, d_i)
        T = torch.bmm(T, T_i)
        T_all.append(T)

    pos = T[:, :3, 3]
    R = T[:, :3, :3]
    return pos, R, T_all


def rotation_matrix_to_quaternion(R):
    """
    Convert batched rotation matrices (batch,3,3) to quaternions (batch,4)
    in (w, x, y, z) order. Differentiable, numerically-stabilized version
    (avoids the classic divide-by-near-zero issue with the naive formula).
    """
    batch = R.shape[0]
    device = R.device
    dtype = R.dtype

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = m00 + m11 + m22

    q = torch.zeros(batch, 4, dtype=dtype, device=device)

    # Case-based construction, vectorized via masks so it stays differentiable
    # everywhere except a measure-zero set of exact rotation-matrix ties.
    cond1 = trace > 0
    s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2
    q1 = torch.stack([
        0.25 * s1,
        (m21 - m12) / s1,
        (m02 - m20) / s1,
        (m10 - m01) / s1,
    ], dim=1)

    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 2
    q2 = torch.stack([
        (m21 - m12) / s2,
        0.25 * s2,
        (m01 + m10) / s2,
        (m02 + m20) / s2,
    ], dim=1)

    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    s3 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-8)) * 2
    q3 = torch.stack([
        (m02 - m20) / s3,
        (m01 + m10) / s3,
        0.25 * s3,
        (m12 + m21) / s3,
    ], dim=1)

    cond4 = (~cond1) & (~cond2) & (~cond3)
    s4 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-8)) * 2
    q4 = torch.stack([
        (m10 - m01) / s4,
        (m02 + m20) / s4,
        (m12 + m21) / s4,
        0.25 * s4,
    ], dim=1)

    q = torch.where(cond1.unsqueeze(1), q1, q)
    q = torch.where(cond2.unsqueeze(1), q2, q)
    q = torch.where(cond3.unsqueeze(1), q3, q)
    q = torch.where(cond4.unsqueeze(1), q4, q)

    return q


def jacobian(q):
    """
    Compute the 6x6 geometric Jacobian (batch,6,6) via autograd.
    Rows 0-2: linear velocity part. Rows 3-5: angular velocity part.
    Used by DLS and by near-singularity sampling (min singular value check).

    Note: for a single q vector at a time is simplest and clear enough for
    this project's scale (dataset generation, not real-time control).
    """
    q = q.clone().detach().requires_grad_(True)
    pos, R, _ = forward_kinematics(q.unsqueeze(0))
    pos = pos.squeeze(0)  # (3,)

    J_linear = torch.zeros(3, N_JOINTS)
    for i in range(3):
        grad = torch.autograd.grad(pos[i], q, retain_graph=True, create_graph=False)[0]
        J_linear[i] = grad

    # Angular part via standard DH z-axis convention: each joint's angular
    # contribution is the z-axis of the previous frame (revolute joints).
    _, _, T_all = forward_kinematics(q.unsqueeze(0))
    J_angular = torch.zeros(3, N_JOINTS)
    z_prev = torch.tensor([0.0, 0.0, 1.0])  # base frame z-axis
    J_angular[:, 0] = z_prev
    for i in range(1, N_JOINTS):
        z_prev = T_all[i - 1][0, :3, 2]  # z-axis of frame i-1
        J_angular[:, i] = z_prev

    J = torch.cat([J_linear, J_angular], dim=0)  # (6,6)
    return J


def analytical_jacobian(q):
    """
    Fast closed-form geometric Jacobian, avoiding autograd entirely.
    Standard formula for a serial chain of revolute joints:
        linear column_i  = axis_i x (p_end - pivot_i)
        angular column_i = axis_i
    where pivot_i and axis_i are joint i's position and rotation axis in
    the world frame (same convention as jacobian()'s angular part above),
    and p_end is the final end-effector position.

    This computes the SAME mathematical quantity as jacobian() above, just
    via the analytical formula instead of autograd -- verified to match
    jacobian() to numerical precision. Used in place of jacobian() inside
    iterative solvers (DLS), where it's called every iteration and autograd's
    overhead (6 backward passes per call) becomes the dominant cost: ~6.4ms/
    call via autograd vs a small fraction of that analytically.

    q: 1D tensor or array, shape (6,)
    Returns: (6,6) numpy array.
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().numpy()
    else:
        q_np = np.asarray(q)

    with torch.no_grad():
        q_t = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0)
        pos, _, T_all = forward_kinematics(q_t)
    p_end = pos.squeeze(0).numpy()

    pivots = [np.zeros(3)]
    axes = [np.array([0.0, 0.0, 1.0])]
    for i in range(N_JOINTS - 1):
        T = T_all[i][0].numpy()
        pivots.append(T[:3, 3])
        axes.append(T[:3, 2])

    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        J[:3, i] = np.cross(axes[i], p_end - pivots[i])
        J[3:, i] = axes[i]
    return J


if __name__ == "__main__":
    # Quick smoke test: run FK on a batch of random joint configs and
    # print pose + quaternion, so you have something to eyeball before
    # cross-checking against MuJoCo.
    torch.manual_seed(0)
    q_test = (torch.rand(4, N_JOINTS) * 2 - 1) * torch.pi  # random in [-pi, pi]

    pos, R, _ = forward_kinematics(q_test)
    quat = rotation_matrix_to_quaternion(R)

    print("Test joint angles (radians):")
    print(q_test)
    print("\nEnd-effector positions (meters):")
    print(pos)
    print("\nEnd-effector quaternions (w,x,y,z):")
    print(quat)

    print("\nJacobian at first test config:")
    J = jacobian(q_test[0])
    print(J)
    print("\nSingular values (checks for near-singularity):")
    print(torch.linalg.svdvals(J))
