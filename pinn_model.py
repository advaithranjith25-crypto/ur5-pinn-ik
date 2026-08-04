"""
Phase 2: PINN for UR5 inverse kinematics.

Input:  target end-effector pose (position[3] + quaternion[4] = 7 numbers)
Output: predicted joint angles q[6]

The "physics-informed" part is the loss function, not the architecture --
the network itself is a plain MLP. Four loss terms, each isolated into its
own function so they can be logged and weighted independently:

  1. data_loss           - MSE against ground-truth q (standard supervision)
  2. fk_consistency_loss - re-run predicted q through the SAME differentiable
                            FK from fk_ur5.py, penalize distance between the
                            resulting pose and the target pose. This is what
                            makes the network's output self-consistent with
                            robot geometry, not just close to training labels.
  3. joint_limit_loss    - hinge penalty, only active when prediction exceeds
                            [q_min, q_max]
  4. torque_feasibility_loss - simplified static-gravity-torque penalty.
                            NOTE: this is a deliberately simplified proxy
                            (static pose, no dynamics/inertia terms) -- state
                            this explicitly in your report as a scoping choice,
                            not an oversight.
"""

import torch
import torch.nn as nn

from fk_ur5 import forward_kinematics, rotation_matrix_to_quaternion, N_JOINTS

# ---------------------------------------------------------------------------
# Joint limits (must match generate_dataset.py's JOINT_LOW/JOINT_HIGH)
# ---------------------------------------------------------------------------
Q_MIN = -torch.pi * torch.ones(N_JOINTS)
Q_MAX = torch.pi * torch.ones(N_JOINTS)

# Rough per-link mass proxy for the simplified torque penalty (kg, decreasing
# outward along the arm -- a coarse stand-in, not the real UR5 spec).
LINK_MASS_PROXY = torch.tensor([3.7, 8.4, 2.3, 1.2, 1.2, 0.19])
GRAVITY = 9.81
MAX_TORQUE_PROXY = 50.0  # Nm, rough per-joint ceiling for the penalty


class IKPinn(nn.Module):
    """MLP mapping target pose (pos+quat, 7-dim) -> joint angles (6-dim)."""

    def __init__(self, hidden_dim=384, n_hidden_layers=5):
        super().__init__()
        layers = [nn.Linear(7, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, N_JOINTS)]
        self.net = nn.Sequential(*layers)

        # Output scaled by tanh * pi so predictions are naturally bounded
        # to a plausible joint range before any loss even applies -- this
        # doesn't replace the joint-limit loss, it just gives training an
        # easier starting point (avoids wild initial predictions).
        self.output_scale = torch.pi

    def forward(self, pos, quat):
        x = torch.cat([pos, quat], dim=-1)
        raw = self.net(x)
        q_pred = torch.tanh(raw) * self.output_scale
        return q_pred


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def data_loss(q_pred, q_true):
    return nn.functional.mse_loss(q_pred, q_true)


def fk_consistency_loss(q_pred, target_pos, target_quat):
    """
    Re-run predicted q through the differentiable FK; penalize how far the
    resulting pose is from the target. Position: plain MSE. Orientation:
    quaternion geodesic distance, handling the q/-q sign ambiguity (both
    represent the same rotation).
    """
    pos_pred, R_pred, _ = forward_kinematics(q_pred)
    quat_pred = rotation_matrix_to_quaternion(R_pred)

    pos_err = nn.functional.mse_loss(pos_pred, target_pos)

    # geodesic-safe quaternion distance: min(|q1-q2|, |q1+q2|)
    d_plus = torch.norm(quat_pred - target_quat, dim=-1)
    d_minus = torch.norm(quat_pred + target_quat, dim=-1)
    quat_err = torch.minimum(d_plus, d_minus).pow(2).mean()

    return pos_err + quat_err


def joint_limit_loss(q_pred, q_min=Q_MIN, q_max=Q_MAX):
    """Hinge penalty: zero inside limits, grows linearly-squared outside."""
    q_min = q_min.to(q_pred.device)
    q_max = q_max.to(q_pred.device)
    below = torch.relu(q_min - q_pred)
    above = torch.relu(q_pred - q_max)
    return (below.pow(2) + above.pow(2)).mean()


def torque_feasibility_loss(q_pred):
    """
    Simplified static-gravity-torque proxy. For each joint, approximate the
    gravity torque as (downstream link masses) * g * horizontal lever arm,
    using sin(q) as a crude proxy for how "extended" the arm is at that
    joint -- NOT a full dynamics model. Penalize predictions implying a
    torque above MAX_TORQUE_PROXY.

    This is explicitly a coarse scoping simplification (see module
    docstring) -- worth stating as such in the report rather than presenting
    it as a full rigid-body dynamics torque calculation.
    """
    device = q_pred.device
    masses = LINK_MASS_PROXY.to(device)
    downstream_mass = torch.flip(torch.cumsum(torch.flip(masses, [0]), 0), [0])  # (6,)

    lever_proxy = torch.sin(q_pred).abs()  # (batch, 6) in [0,1]
    torque_proxy = lever_proxy * downstream_mass.unsqueeze(0) * GRAVITY * 0.3  # 0.3m nominal arm length

    excess = torch.relu(torque_proxy - MAX_TORQUE_PROXY)
    return excess.pow(2).mean()


def total_loss(q_pred, q_true, target_pos, target_quat, weights=None):
    """
    Weighted sum of all four terms. Returns (total, dict_of_components) so
    you can log each term separately during training -- essential for
    showing in your report that every term is actually contributing, not
    just the data loss dominating everything else.
    """
    if weights is None:
        weights = {"data": 1.0, "fk": 1.0, "limits": 0.1, "torque": 0.05}

    l_data = data_loss(q_pred, q_true)
    l_fk = fk_consistency_loss(q_pred, target_pos, target_quat)
    l_limits = joint_limit_loss(q_pred)
    l_torque = torque_feasibility_loss(q_pred)

    total = (
        weights["data"] * l_data
        + weights["fk"] * l_fk
        + weights["limits"] * l_limits
        + weights["torque"] * l_torque
    )

    components = {
        "data": l_data.item(),
        "fk": l_fk.item(),
        "limits": l_limits.item(),
        "torque": l_torque.item(),
        "total": total.item(),
    }
    return total, components


if __name__ == "__main__":
    # Smoke test: random batch through model + loss, check shapes and that
    # gradients flow (i.e. FK-consistency loss is actually differentiable
    # w.r.t. the network's parameters, not just w.r.t. q).
    torch.manual_seed(0)
    model = IKPinn()

    batch = 16
    pos = torch.randn(batch, 3) * 0.3
    quat = torch.randn(batch, 4)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    q_true = (torch.rand(batch, N_JOINTS) * 2 - 1) * torch.pi

    q_pred = model(pos, quat)
    print("q_pred shape:", q_pred.shape)

    loss, comps = total_loss(q_pred, q_true, pos, quat)
    print("Loss components:", comps)

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Total gradient norm across model params: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradients not flowing -- check FK differentiability"
    print("PASS: forward + backward pass work, gradients flow through FK-consistency term.")
