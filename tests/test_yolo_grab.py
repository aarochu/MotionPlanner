"""Tests for yolo_grab: plan math, straight steps, levelling, and the reset / pick & place sequences (fake arm)."""
import math
import unittest
from types import SimpleNamespace as N
from unittest import mock

import numpy as np

import yolo_grab as YG

NEWMID = [-105.97, 4.33, -37.43, 77.91, 95.88, 207.08]
PARKING = [-34.85, -75.79, -2.94, 164.91, 9.43, 194.11]
NEWMID_END = {"x": 3.284, "y": -359.117, "z": 235.121, "o_x": 0.9985, "o_y": -0.0461, "o_z": -0.028, "theta": -5.2}


def measurement(end=NEWMID_END):
    R = YG.ov_to_R(end)
    g0 = np.array([end["x"], end["y"], end["z"]]) + 105 * R[:, 2]
    return {"axis": np.array([560., -292., 250.]), "top": np.array([560., -292., 475.]),
            "bottom": np.array([560., -292., 151.]), "width_mm": 91.0,
            "profile": {0.25: 81., 0.35: 80., 0.45: 80., 0.55: 75.}, "frames": 5, "of": 5, "spread_mm": 1.0,
            "clipped": False, "neighbor": None, "end": dict(end),
            "gripper": {"origin": g0, "x": R[:, 0], "y": R[:, 1], "z": R[:, 2]}}


def pads(pose, reach=165.0):
    return np.array([pose[k] for k in "xyz"]) + reach * YG.ov_to_R(pose)[:, 2]


class PlanTests(unittest.TestCase):
    def test_grab_puts_the_pads_on_the_trimmed_axis_with_a_level_gripper(self):
        cfg = YG.DEFAULTS
        p = YG.plan(measurement(), cfg)
        self.assertFalse([c for c in p["checks"] if c[0] == "error"])
        np.testing.assert_allclose(pads(p["grab"], p["reach"]), p["A"], atol=1e-6)
        pitch, roll, _ = YG.attitude(p["grab"])
        self.assertAlmostEqual(pitch, 0, places=6)
        self.assertAlmostEqual(roll, 0, places=6)
        axis = np.array([560., -292., p["zg"]])
        self.assertAlmostEqual(float((p["A"] - axis).dot(p["lat"])), -cfg["lat_trim_mm"], places=6)   # right = -lat
        self.assertAlmostEqual(float((p["A"] - axis).dot(p["a"])), cfg["depth_trim_mm"], places=6)
        np.testing.assert_allclose(pads(p["pre"], p["reach"]), p["A"] - cfg["pre_mm"] * p["a"], atol=1e-6)

    def test_a_bottle_wider_than_the_gripper_is_refused(self):
        m = measurement()
        m["profile"] = {f: 90.0 for f in m["profile"]}
        self.assertIn("error", [c[0] for c in YG.plan(m, YG.DEFAULTS)["checks"]])

    def test_unstable_detection_is_refused(self):
        m = measurement()
        m["spread_mm"] = 25.0
        self.assertIn("error", [c[0] for c in YG.plan(m, YG.DEFAULTS)["checks"]])


class GeometryTests(unittest.TestCase):
    def test_orientation_vector_round_trip(self):
        for e in (NEWMID_END, {"o_x": 0.0, "o_y": 0.6, "o_z": 0.8, "theta": 40.0}):
            back = YG.R_to_ov(YG.ov_to_R(e))
            np.testing.assert_allclose(YG.ov_to_R(back), YG.ov_to_R(e), atol=1e-9)

    def test_steps_are_straight_short_and_end_on_the_goal(self):
        a = dict(NEWMID_END)
        b = dict(NEWMID_END, x=300.0, y=-280.0, z=370.0)
        w = YG.steps(a, b, 40.0)
        pts = [np.array([a[k] for k in "xyz"])] + [np.array([s[k] for k in "xyz"]) for s in w]
        self.assertTrue(all(np.linalg.norm(q - p) <= 40.0 + 1e-9 for p, q in zip(pts, pts[1:])))
        np.testing.assert_allclose(pts[-1], [300.0, -280.0, 370.0])

    def test_level_target_zeroes_pitch_and_roll_and_keeps_the_pads_still(self):
        target, turn = YG.level_target(NEWMID_END, 165.0)
        self.assertLess(turn, 10)
        pitch, roll, heading = YG.attitude(target)
        self.assertAlmostEqual(pitch, 0, places=6)
        self.assertAlmostEqual(roll, 0, places=6)
        self.assertAlmostEqual(heading, YG.attitude(NEWMID_END)[2], places=6)
        np.testing.assert_allclose(pads(target), pads(NEWMID_END), atol=1e-6)


class FakeArm:
    """Records moves; joint moves to known keyframes set a matching end pose."""
    def __init__(self, joints, end):
        self.joints, self.end, self.log = list(joints), dict(end), []

    def get_joint_positions(self):
        return N(values=list(self.joints))

    def get_end_position(self):
        return N(**self.end)

    def move_to_position(self, pose):
        self.end = {k: getattr(pose, k) for k in YG.POSE_KEYS}
        self.joints = [None] * 6
        self.log.append(("pose", dict(self.end)))

    def move_to_joint_positions(self, jp):
        self.joints = list(jp.values)
        if self.joints == list(YG.PLACE_ABOVE_JOINTS):
            o = math.radians(YG.PLACE_HEADING_DEG)
            self.end = {"x": YG.PLACE_XY[0], "y": YG.PLACE_XY[1], "z": 200.0,
                        "o_x": math.cos(o), "o_y": math.sin(o), "o_z": 0.0, "theta": 0.0}
        elif self.joints == NEWMID:
            self.end = dict(NEWMID_END)
        self.log.append(("joints", list(self.joints)))


def call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.gripper = N(open=mock.Mock(), grab=mock.Mock(return_value=True))

    def test_reset_backs_out_rises_then_parks(self):
        o = math.radians(YG.PLACE_HEADING_DEG)
        arm = FakeArm([0] * 6, {"x": -91.1, "y": -379.5, "z": 104.0, "o_x": math.cos(o), "o_y": math.sin(o),
                                "o_z": 0.0, "theta": 0.0})
        YG.reset_arm(call, arm, self.gripper, PARKING, log=lambda s: None)
        self.gripper.open.assert_called_once()
        poses = [e for kind, e in arm.log if kind == "pose"]
        self.assertTrue(all(e["z"] >= 104.0 - 1e-9 for e in poses))
        self.assertAlmostEqual(poses[-1]["z"], YG.RESET_CLEAR_Z)
        self.assertEqual(arm.log[-1], ("joints", PARKING))

    def test_reset_at_parking_only_opens(self):
        arm = FakeArm(PARKING, NEWMID_END)
        YG.reset_arm(call, arm, self.gripper, PARKING, log=lambda s: None)
        self.assertEqual(arm.log, [])

    def test_pick_and_place_ends_holding_at_the_computed_set_down_height(self):
        arm = FakeArm(PARKING, NEWMID_END)
        m = measurement()
        with mock.patch.object(YG, "measure", return_value=m):
            m_out, p = YG.pick_and_place(call, call, None, arm, self.gripper, None, "cam", "gripper", NEWMID,
                                         PARKING, YG.DEFAULTS, log=lambda s: None)
        self.assertEqual(arm.log[0], ("joints", NEWMID))
        self.assertIn(("joints", list(YG.PLACE_ABOVE_JOINTS)), arm.log)
        self.gripper.grab.assert_called_once()
        expected = YG.TABLE_Z + (p["zg"] - m["bottom"][2])
        self.assertAlmostEqual(arm.end["z"], expected, places=6)
        self.assertAlmostEqual(arm.end["x"], YG.PLACE_XY[0])

    def test_pick_and_place_refuses_to_start_away_from_parking_or_newmid(self):
        arm = FakeArm([0, 0, 0, 0, 0, 0], NEWMID_END)
        with self.assertRaises(ValueError):
            YG.pick_and_place(call, call, None, arm, self.gripper, None, "cam", "gripper", NEWMID, PARKING,
                              YG.DEFAULTS, log=lambda s: None)
        self.assertEqual(arm.log, [])

    def test_a_missed_grab_opens_and_backs_out(self):
        self.gripper.grab.return_value = False
        arm = FakeArm(NEWMID, NEWMID_END)
        with mock.patch.object(YG, "measure", return_value=measurement()):
            with self.assertRaises(ValueError):
                YG.pick_and_place(call, call, None, arm, self.gripper, None, "cam", "gripper", NEWMID, PARKING,
                                  YG.DEFAULTS, log=lambda s: None)
        self.assertEqual(self.gripper.open.call_count, 2)
        self.assertNotIn(("joints", list(YG.PLACE_ABOVE_JOINTS)), arm.log)


if __name__ == "__main__":
    unittest.main()
