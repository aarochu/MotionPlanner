"""Tests for the wrist wave (fixed-direction joint moves) and Simultaneous Start Up, against fake arms.

Uses the real Mover / BottleTask code and the real poses.json; only the robot is faked, so nothing moves."""
import asyncio
import types
import unittest
from unittest import mock

import bottle_task as BT
import poses_io

WRIST = BT.WRIST


def run(coro):
    return asyncio.run(coro)


class FakeArm:
    def __init__(self, name, joints, log):
        self.name, self.joints, self.log = name, list(joints), log

    async def get_joint_positions(self):
        return types.SimpleNamespace(values=list(self.joints))

    async def move_to_joint_positions(self, jp):
        self.log.append(("start", self.name, list(jp.values)))
        await asyncio.sleep(0)          # let the other arm start too if run together
        self.joints = list(jp.values)
        self.log.append(("joints", self.name, list(jp.values)))

    async def stop(self):
        self.log.append(("stop", self.name, None))

    async def do_command(self, cmd):
        if "set_acceleration" in cmd:
            self.log.append(("accel", self.name, cmd["set_acceleration"]))
        return {}


class FakeGripper:
    def __init__(self, name, log):
        self.name, self.log = name, log

    async def grab(self):
        self.log.append(("grab", self.name, None))
        return False

    async def open(self):
        self.log.append(("open", self.name, None))


class FakeMover(BT.Mover):
    """Real Mover.go / joints_to; planned moves are recorded and the arm 'arrives' at the keyframe."""
    def __init__(self, arms, log, poses_by_name):
        self.run, self.arms, self._kin, self.log = run, arms, {}, log
        self._lock = mock.MagicMock()
        self.poses_by_name = poses_by_name

    def go(self, kf):
        if not kf.get("joint_move"):
            self.log.append(("planned", kf["arm"], kf["name"]))
            arrive = kf.get("values") or self.poses_by_name.get((kf["arm"], kf["name"]), {}).get("_joints")
            if arrive:
                self.arms[kf["arm"]].joints = list(arrive)
            return
        super().go(kf)


def arm1_keyframes():
    return [p for p in poses_io.load("poses.json") if p["arm"] == "arm"]


class WaveKeyframeTests(unittest.TestCase):
    def setUp(self):
        self.kf = {p["name"]: p for p in arm1_keyframes()}
        self.park = self.kf["parking"]["values"]
        self.log = []
        self.arms = {"arm": FakeArm("arm", self.park, self.log)}
        self.mover = FakeMover(self.arms, self.log, {})

    def test_wave_keyframes_are_flagged_and_others_are_planned(self):
        flagged = [n for n, p in self.kf.items() if p.get("joint_move")]
        self.assertEqual(sorted(flagged), sorted(["right1", "left1", "right2", "left2", "parking_close", "parking_open"]))
        for n in ("parking", "NewMid", "grabframe1", "end"):
            self.assertFalse(self.kf[n].get("joint_move"), f"{n} must stay planned")

    def test_wave_keyframe_is_a_direct_joint_move(self):
        self.mover.go(self.kf["right1"])
        moves = [e for e in self.log if e[0] == "joints"]
        self.assertEqual(len(moves), 1)
        self.assertAlmostEqual(moves[0][2][WRIST] - self.park[WRIST], 59.93, places=1)
        self.assertNotIn("planned", [e[0] for e in self.log])

    def test_wave_keyframe_turns_with_high_acceleration_then_restores_it(self):
        self.mover.go(self.kf["right1"])
        seq = [(e[0], e[2]) if e[0] == "accel" else (e[0], None) for e in self.log if e[0] in ("accel", "joints")]
        self.assertEqual(seq, [("accel", BT.WAVE_ACCEL), ("joints", None), ("accel", BT.DEFAULT_ACCEL)])

    def test_wave_keyframe_is_refused_away_from_its_pose(self):
        self.arms["arm"].joints = list(self.kf["NewMid"]["values"])
        with self.assertRaises(ValueError):
            self.mover.go(self.kf["right1"])
        self.assertEqual([e for e in self.log if e[0] in ("start", "joints")], [], "nothing may move")


class Arm1IntroValidation(unittest.TestCase):
    """Aaron's arm 1 intro through the real Full run keyframe step: parking -> turns -> close -> open, and the rest
    of the sequence is untouched (still planned)."""

    def test_intro_then_rest_of_sequence(self):
        kfs = arm1_keyframes()
        park = next(p for p in kfs if p["name"] == "parking")["values"]
        log = []
        arms = {"arm": FakeArm("arm", [0, -40, -20, 160, 30, 150], log)}   # start away from parking
        grippers = {"arm": FakeGripper("arm", log)}
        mover = FakeMover(arms, log, {})
        task = BT.BottleTask(run, arms, grippers, unscrewer=None, mover=mover)
        for n, kf in enumerate(kfs, 1):
            task._keyframe(f"{n}", kf)

        kinds = [(e[0], e[2] if e[0] == "planned" else None) for e in log if e[0] in ("planned", "joints", "grab", "open")]
        names = [p["name"] for p in kfs]
        self.assertEqual(names[:7], ["parking", "right1", "left1", "right2", "left2", "parking_close", "parking_open"])

        # 1. parking is reached through the planner (from anywhere)
        self.assertEqual(kinds[0], ("planned", "parking"))
        # 2. the four turns, then back to parking: direct joint moves, wrist only, in these fixed directions
        joint_moves = [e[2] for e in log if e[0] == "joints"]
        wrist = [j[WRIST] - park[WRIST] for j in joint_moves]
        self.assertEqual(len(wrist), 6)
        for got, want in zip(wrist, [59.93, -59.50, 59.93, -59.50, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=1)
        steps = [b - a for a, b in zip([0.0] + wrist[:-1], wrist)]
        self.assertEqual([("+" if d > 0.01 else "-" if d < -0.01 else "0") for d in steps], ["+", "-", "+", "-", "+", "0"],
                         "right, left, right, left, back to parking (turning right), then no turn")
        for j in joint_moves:
            self.assertTrue(all(abs(a - b) < 0.01 for a, b in zip(j[:WRIST], park[:WRIST])), "only the wrist turns")
        # 3. claw closes at parking_close, opens at parking_open
        grip = [e[0] for e in log if e[0] in ("grab", "open")]
        self.assertEqual(grip[-2:], ["grab", "open"])
        # 4. the rest of arm 1's sequence is still planned, exactly as before
        rest = [e[2] for e in log if e[0] == "planned"][1:]
        self.assertEqual(rest, ["NewMid", "grabframe1", "end"])


class StartupTests(unittest.TestCase):
    PARK = [-34.85, -75.79, -2.94, 164.91, 9.43, 194.11]
    HAND_PLACED = [34.855, -75.763, -2.996, -164.868, 9.437, -14.027]   # arm2 as Aaron placed it (camera upside down)

    def setUp(self):
        self.log = []
        self.arms = {"arm": FakeArm("arm", [0, 0, 0, 0, 0, 0], self.log),
                     "arm2": FakeArm("arm2", [-8.6, -4.4, -91.7, 0.2, 96.1, -96.9], self.log)}
        self.grippers = {n: FakeGripper(n, self.log) for n in self.arms}
        self.mirror = BT.mirror_joints(self.PARK)
        self.starts = {"arm2": {"arm": "arm2", "name": "mirror of arm's parking", "values": self.mirror,
                                "approach_values": BT.half_turn(self.mirror)},
                       "arm": {"arm": "arm", "name": "parking", "values": list(self.PARK)}}
        self.finish = {"arm2": {"arm": "arm2", "name": "safe_bottle_start_pose", "pose": {}}}
        self.mover = FakeMover(self.arms, self.log, {})
        self.offsets = BT.wave_offsets(poses_io.load("poses.json"))
        self.waves = BT.startup_waves(self.offsets)

    def run_startup(self):
        BT.simultaneous_startup(run, self.arms, self.grippers, self.mover, self.starts, self.waves, self.finish,
                                log=lambda *_: None)

    def wave_moves(self, arm):
        """arm's joint moves during the wave (after any single-arm move to its start)."""
        moves = [e for e in self.log if e[0] == "joints" and e[1] == arm]
        return moves[-(len(self.offsets) + 1):]

    def test_offsets_match_arm1_wave(self):
        for got, want in zip(self.offsets, [59.93, -59.50, 59.93, -59.50]):
            self.assertAlmostEqual(got, want, places=1)

    def test_arm2_start_is_a_true_mirror_camera_upright(self):
        for got, want in zip(self.mirror, [34.85, -75.79, -2.94, -164.91, 9.43, -194.11]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(BT.half_turn(self.mirror)[WRIST], -14.11, places=6)
        self.assertLess(max(abs(a - b) for a, b in zip(BT.half_turn(self.mirror), self.HAND_PLACED)),
                        BT.JOINT_MOVE_TOL_DEG, "the half-turn pose is where Aaron placed arm2 by hand")

    def test_arm2_turns_are_reversed_left_first(self):
        self.assertEqual([n for n, _ in self.waves["arm"]], ["right1", "left1", "right2", "left2"])
        self.assertEqual([n for n, _ in self.waves["arm2"]], ["left1", "right1", "left2", "right2"])
        self.assertLess(self.waves["arm2"][0][1], 0, "arm2's first turn is the left (negative) one")
        self.assertGreater(self.waves["arm"][0][1], 0, "arm's first turn is the right (positive) one")

    def test_from_the_hand_placed_pose_only_the_wrist_turns_into_place(self):
        self.arms["arm"].joints = list(self.PARK)
        self.arms["arm2"].joints = list(self.HAND_PLACED)
        self.run_startup()
        planned = [(e[1], e[2]) for e in self.log if e[0] == "planned"]
        self.assertEqual(planned, [("arm2", "safe_bottle_start_pose")], "no planned move to the starts")
        first = next(e for e in self.log if e[0] == "joints")
        self.assertEqual(first[1], "arm2")
        self.assertAlmostEqual(first[2][WRIST], -194.11, places=6)
        self.assertTrue(all(abs(a - b) < 0.1 for a, b in zip(first[2][:WRIST], self.HAND_PLACED[:WRIST])),
                        "turning into place moves only the wrist")

    def test_full_startup_from_elsewhere(self):
        self.run_startup()
        planned = [(e[1], e[2]) for e in self.log if e[0] == "planned"]
        self.assertEqual(planned, [("arm2", "mirror of arm's parking (wrist half a turn round)"), ("arm", "parking"),
                                   ("arm2", "safe_bottle_start_pose")])
        base = {"arm": self.PARK, "arm2": self.mirror}
        for n in ("arm", "arm2"):
            wave = self.wave_moves(n)
            wrist = [round(e[2][WRIST] - base[n][WRIST], 2) for e in wave]
            self.assertEqual(wrist, [round(o, 2) for _, o in self.waves[n]] + [0.0], f"{n} wrist turns")
            for e in wave:
                self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(e[2][:WRIST], base[n][:WRIST])),
                                "only the wrist turns during the wave")
        # the wave turns start on both arms before either finishes: they move at the same time
        idx = max(i for i, e in enumerate(self.log) if e[0] == "planned" and e[2] == "parking")
        moves = [(e[0], e[1]) for e in self.log[idx + 1:] if e[0] in ("start", "joints")]
        self.assertEqual(len(moves), 4 * (len(self.offsets) + 1))
        for i in range(0, len(moves), 4):
            self.assertEqual([k for k, _ in moves[i:i + 2]], ["start", "start"], f"turn {i // 4 + 1} runs on both together")
        tail = [e[0] if e[0] != "planned" else ("planned", e[1], e[2]) for e in self.log if e[0] in ("grab", "open", "planned")]
        self.assertEqual(tail[-5:], ["grab", "grab", "open", "open", ("planned", "arm2", "safe_bottle_start_pose")])
        self.assertEqual(self.arms["arm"].joints, self.PARK, "arm ends at parking")

    def test_wave_uses_high_acceleration_and_planned_moves_do_not(self):
        self.run_startup()
        events = [(e[0], e[1], e[2]) for e in self.log if e[0] in ("accel", "joints", "planned")]
        parked = next(i for i, e in enumerate(events) if e[0] == "planned" and e[2] == "parking")
        first_accel = next(i for i, e in enumerate(events) if e[0] == "accel" and i > parked)   # the shared wave
        restore = [i for i, e in enumerate(events) if e[0] == "accel" and e[2] == BT.DEFAULT_ACCEL]
        self.assertEqual({e[1] for e in events[first_accel:first_accel + 2]}, {"arm", "arm2"})
        self.assertTrue(all(e[2] == BT.WAVE_ACCEL for e in events[first_accel:first_accel + 2]))
        wave_idx = [i for i, e in enumerate(events) if e[0] == "joints"][-10:]          # the 5 shared turns x 2 arms
        self.assertTrue(all(first_accel < i < restore[-1] for i in wave_idx), "every wave turn at high acceleration")
        final = next(i for i, e in enumerate(events) if e[0] == "planned" and e[2] == "safe_bottle_start_pose")
        self.assertLess(restore[-1], final, "normal acceleration restored before arm2's planned move back")
        planned_starts = [i for i, e in enumerate(events) if e[0] == "planned" and e[2] != "safe_bottle_start_pose"]
        self.assertTrue(all(i < first_accel for i in planned_starts), "planned moves to the starts at normal acceleration")

    def test_failure_restores_normal_acceleration(self):
        self.arms["arm"].joints = list(self.PARK)
        self.arms["arm2"].joints = list(self.mirror)

        def boom(text):
            if text.startswith("Wave 2"):
                raise RuntimeError("stopped")
        with self.assertRaises(RuntimeError):
            BT.simultaneous_startup(run, self.arms, self.grippers, self.mover, self.starts, self.waves, self.finish,
                                    log=boom)
        last = {e[1]: e[2] for e in self.log if e[0] == "accel"}
        self.assertEqual(last, {"arm": BT.DEFAULT_ACCEL, "arm2": BT.DEFAULT_ACCEL})

    def test_failure_stops_both_arms(self):
        def boom(kf):
            raise RuntimeError("no collision-free path")
        self.mover.go = boom
        with self.assertRaises(RuntimeError):
            self.run_startup()
        self.assertEqual(sorted(e[1] for e in self.log if e[0] == "stop"), ["arm", "arm2"])
        self.assertEqual([e for e in self.log if e[0] in ("start", "joints")], [])


if __name__ == "__main__":
    unittest.main()
