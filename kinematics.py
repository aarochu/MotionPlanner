"""Forward kinematics: the end pose a set of joint angles puts an arm at, without moving it.

Uses the arm's own kinematics model (Arm.get_kinematics, Viam SVA JSON), so the result matches what
get_end_position reports once the arm is there: x, y, z in mm and a Viam orientation vector (o_x, o_y, o_z,
theta in degrees), in the arm's base frame. Used to send joint keyframes through the motion planner.
"""
import json

import numpy as np


def rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def ov_to_matrix(o_x, o_y, o_z, theta):
    """Viam orientation vector -> rotation matrix (Z-Y-Z: longitude, latitude, theta)."""
    o = np.array([o_x, o_y, o_z], float)
    o = o / np.linalg.norm(o)
    lat = np.degrees(np.arccos(np.clip(o[2], -1, 1)))
    lon = np.degrees(np.arctan2(o[1], o[0])) if 1 - abs(o[2]) > 1e-4 else 0.0
    return rot((0, 0, 1), lon) @ rot((0, 1, 0), lat) @ rot((0, 0, 1), theta)


def matrix_to_ov(m):
    o_x, o_y, o_z = m[:, 2]
    lat = np.degrees(np.arccos(np.clip(o_z, -1, 1)))
    lon = np.degrees(np.arctan2(o_y, o_x)) if 1 - abs(o_z) > 1e-4 else 0.0
    r = rot((0, 1, 0), -lat) @ rot((0, 0, 1), -lon) @ m  # what is left is the spin about the tool axis
    return float(o_x), float(o_y), float(o_z), float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


class Kinematics:
    """One arm's model. `model` is the parsed SVA JSON from get_kinematics."""

    def __init__(self, model):
        self.child = {}  # each frame's single child: the chain world -> base -> joint -> link -> ... -> end
        for link in model["links"]:
            self.child[link["parent"]] = ("link", link)
        for joint in model["joints"]:
            self.child[joint["parent"]] = ("joint", joint)

    @classmethod
    def from_arm(cls, run, arm):
        _, data = run(arm.get_kinematics())
        return cls(json.loads(data))

    def end_pose(self, joints_deg):
        """{"x", "y", "z", "o_x", "o_y", "o_z", "theta"} of the arm's end in its base frame."""
        m, p, name, n = np.eye(3), np.zeros(3), "world", 0
        while name in self.child:
            kind, e = self.child[name]
            if kind == "link":
                t = e["translation"]
                p = p + m @ np.array([t["X"], t["Y"], t["Z"]])
                if e.get("orientation"):
                    v = e["orientation"]["value"]
                    m = m @ ov_to_matrix(v.get("x", 0), v.get("y", 0), v.get("z", 0), v.get("th", 0))
            else:
                a = e["axis"]
                m = m @ rot((a["X"], a["Y"], a["Z"]), joints_deg[n])
                n += 1
            name = e["id"]
        o_x, o_y, o_z, theta = matrix_to_ov(m)
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                "o_x": o_x, "o_y": o_y, "o_z": o_z, "theta": theta}
