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
            self.log.append(("accel", self.name, (cmd.get("set_speed"), cmd["set_acceleration"])))
        return {}


class FakeGripper:
    def __init__(self, name, log):
        self.name, self.log = name, log

    async def do_command(self, cmd):
        if "set_gripper_speed" in cmd:
            self.log.append(("gspeed", self.name, cmd["set_gripper_speed"]))
        return {}

    async def grab(self):
        self.log.append(("grab", self.name, None))
        return False

    async def open(self):
        self.log.append(("open", self.name, None))


class FakeKin:
    """end_pose(joints) -> a stand-in pose that remembers the joints, so a 'planned' move can arrive there."""
    def end_pose(self, joints):
        return {"_joints": [float(v) for v in joints]}


class FakeMover(BT.Mover):
    """Real Mover.go / joints_to / wave_turn; planned moves are recorded and the arm 'arrives'."""
    refuse = ()          # arm names whose planned wave turns the fake planner refuses at planning time
    fail_midway = ()     # arm names whose planned wave turns fail after starting

    def __init__(self, arms, log, poses_by_name):
        self.run, self.arms, self._kin, self.log = run, arms, {FakeMover.__name__: None}, log
        self._lock = mock.MagicMock()
        self.poses_by_name = poses_by_name
        self._kin = {n: FakeKin() for n in arms}

    def target(self, kf):
        return FakeKin().end_pose(kf["values"])

    def to_pose(self, arm_name, pose, linear=None):
        if arm_name in self.refuse:
            raise RuntimeError("all IK solutions failed constraints. Failures: { self-collision constraint ... }")
        self.log.append(("planned_pose", arm_name, pose["_joints"]))
        self.arms[arm_name].joints = list(pose["_joints"])

    def plan_together(self, moves):
        out = []
        for n, pose in moves:
            if n in self.refuse:
                out.append(RuntimeError("all IK solutions failed constraints. Failures: { self-collision constraint }"))
            elif n in self.fail_midway:
                out.append(RuntimeError("context canceled"))
            else:
                self.log.append(("planned_turn", n, pose["_joints"]))
                self.arms[n].joints = list(pose["_joints"])
                out.append(None)
        return out

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
        patcher = mock.patch.object(BT, "WAVE_PLANNED", False)   # these tests cover the direct-joint-move mode
        patcher.start()
        self.addCleanup(patcher.stop)
        self.kf = {p["name"]: p for p in arm1_keyframes()}
        self.park = self.kf["parking"]["values"]
        self.log = []
        self.arms = {"arm": FakeArm("arm", self.park, self.log)}
        self.mover = FakeMover(self.arms, self.log, {})

    def test_wave_keyframes_are_flagged_and_others_are_planned(self):
        flagged = [n for n, p in self.kf.items() if p.get("joint_move")]
        self.assertEqual(sorted(flagged), sorted(["right1", "left1", "parking_close", "parking_open",
                                                  "parking_close2", "parking_open2"]))
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
        self.assertEqual(seq, [("accel", BT.WAVE_MOTION), ("joints", None), ("accel", BT.NORMAL_MOTION)])

    def test_wave_keyframe_is_refused_away_from_its_pose(self):
        self.arms["arm"].joints = list(self.kf["NewMid"]["values"])
        with self.assertRaises(ValueError):
            self.mover.go(self.kf["right1"])
        self.assertEqual([e for e in self.log if e[0] in ("start", "joints")], [], "nothing may move")


class Arm1IntroValidation(unittest.TestCase):
    """Aaron's arm 1 intro through the real Full run keyframe step (direct-joint-move mode): parking -> turns ->
    close -> open, and the rest of the sequence is untouched (still planned)."""

    def setUp(self):
        patcher = mock.patch.object(BT, "WAVE_PLANNED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        self.assertEqual(names[:7], ["parking", "right1", "left1", "parking_close", "parking_open",
                                     "parking_close2", "parking_open2"])

        # 1. parking is reached through the planner (from anywhere)
        self.assertEqual(kinds[0], ("planned", "parking"))
        # 2. one turn each way, then back to parking: direct joint moves, wrist only, in these fixed directions
        joint_moves = [e[2] for e in log if e[0] == "joints"]
        wrist = [j[WRIST] - park[WRIST] for j in joint_moves]
        self.assertEqual(len(wrist), 6)
        for got, want in zip(wrist, [59.93, -59.50, 0.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=1)
        steps = [b - a for a, b in zip([0.0] + wrist[:-1], wrist)]
        self.assertEqual([("+" if d > 0.01 else "-" if d < -0.01 else "0") for d in steps], ["+", "-", "+", "0", "0", "0"],
                         "right, left, back to parking (turning right), then the claw keyframes don't turn")
        for j in joint_moves:
            self.assertTrue(all(abs(a - b) < 0.01 for a, b in zip(j[:WRIST], park[:WRIST])), "only the wrist turns")
        # 3. claw closes and opens twice
        grip = [e[0] for e in log if e[0] in ("grab", "open")]
        self.assertEqual(grip[-4:], ["grab", "open", "grab", "open"])
        # 4. the rest of arm 1's sequence is still planned, exactly as before
        rest = [e[2] for e in log if e[0] == "planned"][1:]
        self.assertEqual(rest, ["NewMid", "grabframe1", "end"])


class StartupTests(unittest.TestCase):
    PARK = [-34.85, -75.79, -2.94, 164.91, 9.43, 194.11]
    HAND_PLACED = [34.855, -75.763, -2.996, -164.868, 9.437, -14.027]   # arm2 as Aaron placed it (camera upside down)
    PLANNED = False      # PlannedStartupTests runs the same checks that apply to both modes with the planner

    def setUp(self):
        patcher = mock.patch.object(BT, "STARTUP_PLANNED", self.PLANNED)
        patcher.start()
        self.addCleanup(patcher.stop)
        FakeMover.refuse = FakeMover.fail_midway = ()
        self.addCleanup(setattr, FakeMover, "refuse", ())
        self.addCleanup(setattr, FakeMover, "fail_midway", ())
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
        self.assertEqual(len(self.offsets), 2, "one turn each way")
        for got, want in zip(self.offsets, [59.93, -59.50]):
            self.assertAlmostEqual(got, want, places=1)

    def test_arm2_start_is_a_true_mirror_camera_upright(self):
        for got, want in zip(self.mirror, [34.85, -75.79, -2.94, -164.91, 9.43, -194.11]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(BT.half_turn(self.mirror)[WRIST], -14.11, places=6)
        self.assertLess(max(abs(a - b) for a, b in zip(BT.half_turn(self.mirror), self.HAND_PLACED)),
                        BT.JOINT_MOVE_TOL_DEG, "the half-turn pose is where Aaron placed arm2 by hand")

    def test_arm2_turns_are_reversed_left_first(self):
        self.assertEqual([n for n, _ in self.waves["arm"]], ["right1", "left1"])
        self.assertEqual([n for n, _ in self.waves["arm2"]], ["left1", "right1"])
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
        self.assertEqual(tail[-9:], ["grab", "grab", "open", "open"] * BT.GRIP_CYCLES
                         + [("planned", "arm2", "safe_bottle_start_pose")], "both grippers close and open twice")
        self.assertEqual(self.arms["arm"].joints, self.PARK, "arm ends at parking")

    def test_wave_uses_high_acceleration_and_planned_moves_do_not(self):
        self.run_startup()
        events = [(e[0], e[1], e[2]) for e in self.log if e[0] in ("accel", "joints", "planned")]
        parked = next(i for i, e in enumerate(events) if e[0] == "planned" and e[2] == "parking")
        first_accel = next(i for i, e in enumerate(events) if e[0] == "accel" and i > parked)   # the shared wave
        restore = [i for i, e in enumerate(events) if e[0] == "accel" and e[2] == BT.NORMAL_MOTION]
        self.assertEqual({e[1] for e in events[first_accel:first_accel + 2]}, {"arm", "arm2"})
        self.assertTrue(all(e[2] == BT.WAVE_MOTION for e in events[first_accel:first_accel + 2]))
        wave_idx = [i for i, e in enumerate(events) if e[0] == "joints"][-2 * (len(self.offsets) + 1):]   # shared turns
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
        self.assertEqual(last, {"arm": BT.NORMAL_MOTION, "arm2": BT.NORMAL_MOTION})

    def test_failure_stops_both_arms(self):
        def boom(kf):
            raise RuntimeError("no collision-free path")
        self.mover.go = boom
        with self.assertRaises(RuntimeError):
            self.run_startup()
        self.assertEqual(sorted(e[1] for e in self.log if e[0] == "stop"), ["arm", "arm2"])
        self.assertEqual([e for e in self.log if e[0] in ("start", "joints")], [])


class DefaultsTests(unittest.TestCase):
    def test_single_arm_wave_is_planned_but_start_up_turns_are_simultaneous_joint_moves(self):
        self.assertTrue(BT.WAVE_PLANNED)
        self.assertFalse(BT.STARTUP_PLANNED, "the motion service cancels a second concurrent move ('context canceled')")


class PlannedWaveTests(unittest.TestCase):
    """WAVE_PLANNED (default): the wave goes through the motion planner at wave speed, restored afterwards."""

    def setUp(self):
        self.kf = {p["name"]: p for p in arm1_keyframes()}
        self.log = []
        self.arms = {"arm": FakeArm("arm", self.kf["parking"]["values"], self.log)}
        self.grippers = {"arm": FakeGripper("arm", self.log)}
        self.mover = FakeMover(self.arms, self.log, {})
        FakeMover.refuse = FakeMover.fail_midway = ()
        self.addCleanup(setattr, FakeMover, "refuse", ())

    def test_default_is_planned(self):
        self.assertTrue(BT.WAVE_PLANNED)

    def test_wave_keyframe_is_planned_at_wave_speed_then_restored(self):
        self.mover.go(self.kf["right1"])
        ev = [(e[0], e[2]) for e in self.log if e[0] in ("accel", "planned_pose")]
        self.assertEqual([k for k, _ in ev], ["accel", "planned_pose", "accel"])
        self.assertEqual(ev[0][1], BT.WAVE_MOTION)
        self.assertEqual(ev[2][1], BT.NORMAL_MOTION)
        self.assertNotIn("joints", [e[0] for e in self.log], "no direct joint move")

    def test_gripper_keyframe_at_parking_moves_nothing(self):
        self.mover.go(self.kf["parking_close"])
        self.assertEqual([e for e in self.log if e[0] in ("planned_pose", "joints", "accel")], [])

    def test_planner_refusal_falls_back_to_a_joint_turn(self):
        FakeMover.refuse = ("arm",)
        self.mover.go(self.kf["right1"])
        self.assertEqual([e[0] for e in self.log if e[0] in ("planned_pose", "joints")], ["joints"])
        self.assertAlmostEqual(self.arms["arm"].joints[WRIST] - self.kf["parking"]["values"][WRIST], 59.93, places=1)
        self.assertEqual([e[2] for e in self.log if e[0] == "accel"][-1], BT.NORMAL_MOTION)

    def test_gripper_keyframes_use_fast_gripper_speed_and_restore_it(self):
        task = BT.BottleTask(run, self.arms, self.grippers, unscrewer=None, mover=self.mover)
        task._keyframe("x", self.kf["parking_close"])
        ev = [(e[0], e[2]) for e in self.log if e[0] in ("gspeed", "grab")]
        self.assertEqual(ev, [("gspeed", BT.WAVE_GRIPPER_SPEED), ("grab", None), ("gspeed", BT.DEFAULT_GRIPPER_SPEED)])

    def test_normal_gripper_actions_do_not_change_speed(self):
        BT.apply_gripper(run, self.grippers["arm"], "grab")
        self.assertEqual([e[0] for e in self.log if e[0] == "gspeed"], [])

    def test_gripper_speed_restored_even_if_the_grab_fails(self):
        class Boom(FakeGripper):
            async def grab(self):
                raise RuntimeError("jammed")
        g = Boom("arm", self.log)
        with self.assertRaises(RuntimeError):
            BT.apply_gripper(run, g, "grab", fast=True)
        self.assertEqual([e[2] for e in self.log if e[0] == "gspeed"][-1], BT.DEFAULT_GRIPPER_SPEED)


class PlannedStartupTests(StartupTests):
    """Simultaneous Start Up with the planner: same sequence and results, turns planned together per step."""
    PLANNED = True

    def test_full_startup_from_elsewhere(self):
        self.run_startup()
        planned = [(e[1], e[2]) for e in self.log if e[0] == "planned"]
        self.assertEqual(planned, [("arm2", "mirror of arm's parking (wrist half a turn round)"), ("arm", "parking"),
                                   ("arm2", "safe_bottle_start_pose")])
        turns = [e for e in self.log if e[0] == "planned_turn"]
        self.assertEqual(len(turns), 2 * (len(self.offsets) + 1), "every turn planned for both arms together")
        for i in range(0, len(turns), 2):
            self.assertEqual({t[1] for t in turns[i:i + 2]}, {"arm", "arm2"})
        tail = [e[0] if e[0] != "planned" else ("planned", e[1], e[2]) for e in self.log if e[0] in ("grab", "open", "planned")]
        self.assertEqual(tail[-9:], ["grab", "grab", "open", "open"] * BT.GRIP_CYCLES
                         + [("planned", "arm2", "safe_bottle_start_pose")], "both grippers close and open twice")
        self.assertEqual(self.arms["arm"].joints, self.PARK, "arm ends at parking")

    def test_wave_uses_high_acceleration_and_planned_moves_do_not(self):
        self.run_startup()
        ev = [(e[0], e[2]) for e in self.log if e[0] in ("accel", "planned_turn", "planned")]
        kinds = [k for k, _ in ev]
        first_turn, last_turn = kinds.index("planned_turn"), len(kinds) - 1 - kinds[::-1].index("planned_turn")
        accels = [i for i, k in enumerate(kinds) if k == "accel"]
        self.assertTrue(any(i < first_turn and ev[i][1] == BT.WAVE_MOTION for i in accels), "fast before the turns")
        restore = [i for i in accels if i > last_turn and ev[i][1] == BT.NORMAL_MOTION]
        self.assertTrue(restore, "normal motion restored after the turns")
        final = next(i for i, e in enumerate(ev) if e[0] == "planned" and e[1] == "safe_bottle_start_pose")
        self.assertLess(restore[0], final, "normal motion before arm2's planned move back")

    def test_turns_are_planned_together_and_only_the_wrist_changes(self):
        self.arms["arm"].joints = list(self.PARK)
        self.arms["arm2"].joints = list(self.HAND_PLACED)
        self.run_startup()
        turns = [e for e in self.log if e[0] == "planned_turn"]
        self.assertEqual(len(turns), 2 * (len(self.offsets) + 1))
        base = {"arm": self.PARK, "arm2": self.mirror}
        for n in ("arm", "arm2"):
            mine = [e[2] for e in turns if e[1] == n]
            self.assertEqual([round(j[WRIST] - base[n][WRIST], 2) for j in mine],
                             [round(o, 2) for _, o in self.waves[n]] + [0.0])
            for j in mine:
                self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(j[:WRIST], base[n][:WRIST])))
        self.assertEqual([e[0] for e in self.log if e[0] == "joints"][1:], [], "no joint moves for the turns")

    def test_refused_arm_turns_by_joint_move_while_the_other_stays_planned(self):
        self.arms["arm"].joints = list(self.PARK)
        self.arms["arm2"].joints = list(self.HAND_PLACED)
        FakeMover.refuse = ("arm2",)
        self.run_startup()
        self.assertEqual({e[1] for e in self.log if e[0] == "planned_turn"}, {"arm"})
        self.assertEqual(len([e for e in self.log if e[0] == "joints" and e[1] == "arm2"]), len(self.offsets) + 1 + 1)
        self.assertEqual(self.arms["arm2"].joints, self.mirror, "arm2 still ends at its wave start")

    def test_mid_move_failure_stops_everything(self):
        self.arms["arm"].joints = list(self.PARK)
        self.arms["arm2"].joints = list(self.HAND_PLACED)
        FakeMover.fail_midway = ("arm2",)
        with self.assertRaises(RuntimeError):
            self.run_startup()
        self.assertEqual(sorted(e[1] for e in self.log if e[0] == "stop"), ["arm", "arm2"])
        self.assertEqual([e[2] for e in self.log if e[0] == "gspeed"][-2:], [BT.DEFAULT_GRIPPER_SPEED] * 2)

    def test_gripper_speed_is_fast_for_the_grips_and_restored(self):
        self.run_startup()
        ev = [e[0] if e[0] != "gspeed" else ("gspeed", e[2]) for e in self.log if e[0] in ("gspeed", "grab", "open")]
        self.assertEqual(ev[0:2], [("gspeed", BT.WAVE_GRIPPER_SPEED)] * 2)
        self.assertEqual(ev[-2:], [("gspeed", BT.DEFAULT_GRIPPER_SPEED)] * 2)
        self.assertEqual(sum(1 for e in ev if e == "grab"), 2 * BT.GRIP_CYCLES)


class Arm2StartupKeyframeTests(unittest.TestCase):
    """arm2's start-up poses are real keyframes (visible in the table), read by Simultaneous Start Up."""

    def setUp(self):
        self.poses = poses_io.load("poses.json")

    def test_keyframes_are_in_the_file_for_arm2_and_flagged(self):
        kf = {p["name"]: p for p in self.poses if p["arm"] == "arm2"}
        for n in (BT.ARM2_START, *BT.ARM2_TURNS):
            self.assertIn(n, kf)
            self.assertTrue(kf[n].get("startup"), f"{n} is flagged start-up")
        self.assertEqual([p["name"] for p in self.poses if p.get("startup")], [BT.ARM2_START, *BT.ARM2_TURNS])
        names = [p["name"] for p in self.poses if p["arm"] == "arm2"]
        self.assertLess(names.index(BT.ARM2_TURNS[1]), names.index("safe_bottle_start_pose"), "listed before the real arm2 job")

    def test_start_is_the_mirror_of_arm1_parking_and_turns_match_the_wave(self):
        park = next(p for p in self.poses if p["arm"] == "arm" and p["name"] == "parking")["values"]
        kf = {p["name"]: p for p in self.poses if p["arm"] == "arm2"}
        start = kf[BT.ARM2_START]["values"]
        for got, want in zip(start, BT.mirror_joints(park)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(kf[BT.ARM2_TURNS[0]]["values"][WRIST] - start[WRIST], -59.50, places=1)   # left first
        self.assertAlmostEqual(kf[BT.ARM2_TURNS[1]]["values"][WRIST] - start[WRIST], 59.93, places=1)

    def test_generated_keyframes_equal_the_saved_ones(self):
        saved = [p for p in self.poses if p.get("startup")]
        self.assertEqual(BT.arm2_startup_keyframes(self.poses), saved)

    def test_run_sequence_and_full_run_never_include_them(self):
        for choice in poses_io.RUN_OPTIONS:
            played = [p["name"] for p in poses_io.sequence(self.poses, choice)]
            self.assertFalse({BT.ARM2_START, *BT.ARM2_TURNS} & set(played), choice)
        self.assertIn("safe_bottle_start_pose", [p["name"] for p in poses_io.sequence(self.poses, "arm2")])

    def test_startup_plan_reads_the_keyframes(self):
        plan, why = BT.startup_plan(self.poses, "safe_bottle_start_pose")
        self.assertIsNone(why)
        self.assertEqual([n for n, _ in plan["waves"]["arm"]], ["right1", "left1"])
        self.assertEqual([n for n, _ in plan["waves"]["arm2"]], list(BT.ARM2_TURNS))
        self.assertEqual([round(o, 1) for _, o in plan["waves"]["arm2"]], [-59.5, 59.9])
        self.assertEqual(plan["finish"]["arm2"]["name"], "safe_bottle_start_pose")

    def test_editing_the_arm2_keyframes_changes_the_startup(self):
        edited = [dict(p, values=list(p["values"])) if p.get("startup") else p for p in self.poses]
        left = next(p for p in edited if p["name"] == BT.ARM2_TURNS[0])
        left["values"][WRIST] = next(p for p in edited if p["name"] == BT.ARM2_START)["values"][WRIST] - 40.0
        plan, _ = BT.startup_plan(edited, "safe_bottle_start_pose")
        self.assertAlmostEqual(plan["waves"]["arm2"][0][1], -40.0, places=6)

    def test_missing_keyframes_fall_back_to_the_mirror(self):
        without = [p for p in self.poses if not p.get("startup")]
        plan, why = BT.startup_plan(without, "safe_bottle_start_pose")
        self.assertIsNone(why)
        self.assertEqual([round(o, 1) for _, o in plan["waves"]["arm2"]], [-59.5, 59.9])

    def test_a_turn_that_moves_other_joints_is_refused(self):
        bad = [dict(p, values=[p["values"][0] + 10] + list(p["values"][1:])) if p["name"] == BT.ARM2_TURNS[1] else p
               for p in self.poses]
        plan, why = BT.startup_plan(bad, "safe_bottle_start_pose")
        self.assertIsNone(plan)
        self.assertIn("only at the wrist", why)


if __name__ == "__main__":
    unittest.main()
