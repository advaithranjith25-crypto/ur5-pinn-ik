"""
Build a MuJoCo MJCF model directly from the UR5 DH parameters in fk_ur5.py.

Why generate it instead of downloading a UR5e model from MuJoCo Menagerie:
the Menagerie UR5e has different link lengths than the classic UR5 DH
parameters. Comparing against it would introduce a real risk of link-length
mismatch masquerading as an FK bug. Generating the MJCF from the *same*
DH numbers used in the PyTorch FK guarantees that any mismatch we find
during verification is a genuine implementation bug, not a spec mismatch.

DH -> MJCF mapping used here (standard technique):
  T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)

Since Trans_z(d_i) and Trans_x(a_i) are pure translations with no rotation
between them, they combine into a single translation vector (a_i, 0, d_i)
in the frame *after* Rot_z(theta_i). So each DH row becomes two nested
MuJoCo bodies:
  - a "joint body" with a hinge about z (this carries theta_i, the DOF)
  - a nested "link body" with fixed pos=[a_i, 0, d_i] and quat=Rot_x(alpha_i)
    (this carries the fixed geometry, and becomes the parent for the next joint)
"""

import numpy as np
from fk_ur5 import UR5_DH, N_JOINTS


def rotx_quat(alpha):
    """Quaternion (w,x,y,z) for a rotation of `alpha` about the x-axis."""
    return (np.cos(alpha / 2), np.sin(alpha / 2), 0.0, 0.0)


def build_mjcf():
    a = UR5_DH["a"].numpy()
    alpha = UR5_DH["alpha"].numpy()
    d = UR5_DH["d"].numpy()

    body_xml = ""
    close_tags = ""
    joint_names = []

    for i in range(N_JOINTS):
        jname = f"joint{i+1}"
        joint_names.append(jname)
        qw, qx, qy, qz = rotx_quat(alpha[i])

        # Joint body: sits at parent's current frame origin, carries the hinge.
        # Nominal mass/inertia added purely because MuJoCo requires moving
        # bodies to have nonzero inertia -- these values are NOT physically
        # meaningful here since gravity is disabled and this model exists
        # only to query kinematics (mj_forward), not to simulate dynamics.
        body_xml += f'<body name="jbody{i+1}" pos="0 0 0">\n'
        body_xml += f'  <joint name="{jname}" type="hinge" axis="0 0 1" range="-6.28 6.28"/>\n'
        body_xml += f'  <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>\n'
        # Link body: fixed offset (a_i, 0, d_i), rotated by alpha_i about x.
        body_xml += (
            f'  <body name="lbody{i+1}" pos="{a[i]:.8f} 0 {d[i]:.8f}" '
            f'quat="{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}">\n'
        )
        close_tags = "  </body>\n</body>\n" + close_tags

    # End-effector site, placed at the final link body's origin.
    ee_site = '<site name="end_effector" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>\n'

    mjcf = f"""<mujoco model="ur5_from_dh">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    {body_xml}
    {ee_site}
    {close_tags}
  </worldbody>
</mujoco>"""

    return mjcf, joint_names


if __name__ == "__main__":
    mjcf_string, joint_names = build_mjcf()
    with open("ur5_from_dh.xml", "w") as f:
        f.write(mjcf_string)
    print("Wrote ur5_from_dh.xml")
    print("Joint order:", joint_names)
