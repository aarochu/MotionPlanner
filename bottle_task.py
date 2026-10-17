"""Full bottle-opening run: keyframe sequence -> unscrew -> retreat up -> via poses -> drop the cap at the trash pose.

Runs in a background thread so the dashboard stays responsive and Stop / STOP ALL can interrupt it. Each phase
waits for the previous one to finish; if any step fails or is stopped, the arms are stopped and the run ends there.
"""
import asyncio
import threading

from viam.components.arm import JointPositions
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from kinematics import Kinematics

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
WRIST = 5                      # Joint 5: wrist twist
JOINT_MOVE_TOL_DEG = 2.0       # a "joint_move" keyframe may only differ from where the arm is at the wrist
WAVE_KEYFRAMES = ("right1", "left1", "right2", "left2")   # arm's wrist-wave intro, turned in this order
# Wrist-only wave turns run faster than normal moves; normal values are restored after every turn and on errors.
WAVE_SPEED = 120.0             # joint speed (deg/s) for wave turns (module range 3-180)
WAVE_ACCEL = 1145.0            # joint acceleration (deg/s^2) for wave turns (module maximum)
DEFAULT_SPEED = 60.0           # restored afterwards: both arms' configured speed_degs_per_sec
DEFAULT_ACCEL = 381.67         # restored afterwards: the xArm module's default (neither arm's config sets it)
WAVE_MOTION = (WAVE_SPEED, WAVE_ACCEL)
NORMAL_MOTION = (DEFAULT_SPEED, DEFAULT_ACCEL)


def set_motion(run, arms, motion):
    """Set joint speed (deg/s) and acceleration (deg/s^2) of each xArm in `arms` through its DoCommand."""
    speed, accel = motion
    for arm in arms:
        run(arm.do_command({"set_speed": float(speed), "set_acceleration": float(accel)}))


class Mover:
    """Every keyframe move goes through the motion service, which plans around the obstacles in the machine
    config (table, walls, ceiling, box-shelf) and the other arm where it stands, instead of the arm's own
    move_to_position (self-collision only) or a joint move (no checks at all).

    XYZ keyframes ("pose": x, y, z in mm and the orientation vector o_x, o_y, o_z, theta, in the arm's base frame
    as get_end_position reports) are the target as is. Joint keyframes ("values") are converted to the end pose
    those joints give (forward kinematics) and planned the same way. If no collision-free path is found, the
    move raises and the arm does not move."""

    def __init__(self, run, robot, arms, motion="builtin"):
        self.run, self.arms = run, arms
        self.motion = MotionClient.from_robot(robot, motion)
        self._kin = {}
        self._lock = threading.Lock()  # one planned move at a time: a second Move would cancel the first

    def kinematics(self, arm_name):
        if arm_name not in self._kin:
            self._kin[arm_name] = Kinematics.from_arm(self.run, self.arms[arm_name])
        return self._kin[arm_name]

    def target(self, kf):
        """The keyframe's end pose (dict of POSE_KEYS) in its arm's base frame."""
        if "pose" in kf:
            return {k: float(kf["pose"][k]) for k in POSE_KEYS}
        return self.kinematics(kf["arm"]).end_pose(kf["values"])

    def to_pose(self, arm_name, pose, linear=None):
        """Plan and move arm_name's end to pose (Pose or dict, in its base frame), collision-checked.
        linear: (line_tolerance_mm, orientation_tolerance_degs) for one smooth straight-line move; it fails rather
        than curve if no straight path exists."""
        if isinstance(pose, dict):
            pose = Pose(**{k: float(pose[k]) for k in POSE_KEYS})
        dest = PoseInFrame(reference_frame=f"{arm_name}_origin", pose=pose)
        constraints = None
        if linear is not None:
            constraints = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=linear[0],
                                                                          orientation_tolerance_degs=linear[1])])
        with self._lock:
            self.run(self.motion.move(component_name=arm_name, destination=dest, constraints=constraints))

    def go(self, kf):
        if kf.get("joint_move"):
            self.joints_to(kf)
        else:
            self.to_pose(kf["arm"], self.target(kf))

    def joints_to(self, kf):
        """A wrist-only keyframe (e.g. the wave), as a direct joint move instead of a planned one. A joint move
        turns each joint straight from its current angle to the target, so the wrist always turns the way the
        keyframe says (+ or -), where the planner could pick either way round. Refused unless joints 0-4 already
        match: the arm never swings through space unchecked, only the wrist turns in place."""
        arm = self.arms[kf["arm"]]
        target = [float(v) for v in kf["values"]]
        now = list(self.run(arm.get_joint_positions()).values)
        off = max(abs(a - b) for a, b in zip(now[:WRIST], target[:WRIST]))
        if off > JOINT_MOVE_TOL_DEG:
            raise ValueError(f"'{kf['name']}' only turns the wrist, so {kf['arm']} must already be at the matching "
                             f"pose (joints 0-4 are {off:.1f}° away, limit {JOINT_MOVE_TOL_DEG:g}°). "
                             "Play the keyframe before it first.")
        with self._lock:
            set_motion(self.run, [arm], WAVE_MOTION)      # faster wrist turns; restored even if the move fails
            try:
                self.run(arm.move_to_joint_positions(JointPositions(values=target)))
            finally:
                set_motion(self.run, [arm], NORMAL_MOTION)


def wave_offsets(poses, arm="arm", base="parking", wave=WAVE_KEYFRAMES):
    """The wave's wrist turns (degrees, + / -) relative to `base`, read from arm's keyframes, so every wave uses the
    same angles as Aaron's taught right1/left1/right2/left2."""
    kf = {p["name"]: p for p in poses if p.get("arm", "arm") == arm and "values" in p}
    missing = [n for n in (base, *wave) if n not in kf]
    if missing:
        raise ValueError(f"{arm} needs joint keyframes named {', '.join(missing)} for the wave.")
    b = kf[base]["values"]
    out = []
    for n in wave:
        v = kf[n]["values"]
        if max(abs(x - y) for x, y in zip(v[:WRIST], b[:WRIST])) > JOINT_MOVE_TOL_DEG:
            raise ValueError(f"'{n}' must differ from '{base}' only at the wrist (Joint {WRIST}).")
        out.append(float(v[WRIST] - b[WRIST]))
    return out


MIRROR_SIGNS = (-1, 1, 1, -1, 1, -1)   # xArm6: yaw and roll joints flip, the bending joints stay


def mirror_joints(values):
    """Joint angles for the mirror image of a pose across the plane halfway between the two arms. The arms' bases
    face the same way, so mirroring flips the joints that twist about vertical or along the arm (0, 3, 5) and keeps
    the bending ones (1, 2, 4). This also mirrors the wrist camera, right side up (within about 3 degrees: the camera
    mount itself is not mirrored). Mirroring arm's parking puts arm2 in the matching spot, facing the middle."""
    return [sign * float(v) for sign, v in zip(MIRROR_SIGNS, values)]


def half_turn(values):
    """The same pose with the final wrist turned half a turn (J5 +/- 180, whichever stays nearer zero): the two-finger
    gripper looks the same, but the wrist camera is on the other side. For arm2's mirrored pose the camera-upright
    version is refused by the motion planner (its model puts cam2 inside arm2's wrist_link), so arm2 is planned to
    this and then turns only its wrist into place."""
    v = [float(x) for x in values]
    j = v[WRIST] + 180.0 if v[WRIST] < 0 else v[WRIST] - 180.0
    return v[:WRIST] + [j] + v[WRIST + 1:]


def startup_waves(offsets, names=WAVE_KEYFRAMES):
    """Each arm's wave for Simultaneous Start Up: arm turns right1, left1, right2, left2 as taught; arm2 does the same
    turns in reverse order, left first (left1, right1, left2, right2), so the two mirror each other."""
    arm1 = list(zip(names, offsets))
    return {"arm": arm1, "arm2": [arm1[i] for i in (1, 0, 3, 2)]}


def simultaneous_startup(run, arms, grippers, mover, starts, waves, finish=None,
                         check=lambda: None, log=print):
    """Simultaneous Start Up.

    starts: {arm name: keyframe} where each arm does its wave, reached one at a time through the motion planner in
    the order given. waves: {arm name: [(label, wrist offset in degrees from that start), ...]}, the same number of
    turns per arm: turn i runs on every arm at the same time, as joint moves (fixed direction). Then every wrist goes
    back to its start, and every gripper closes and opens together. finish: {arm name: keyframe} each of those arms
    moves to afterwards (planned, one at a time); arms not in it stay at their start. Stops all arms on any error."""
    order = list(starts)
    finish = finish or {}
    turns = {len(waves[n]) for n in order}
    if len(turns) != 1:
        raise ValueError("Every arm needs the same number of wave turns.")

    async def together(coros):
        await asyncio.gather(*coros)

    try:
        for n in order:
            check()
            kf = starts[n]
            target = kf.get("values")
            now = [float(v) for v in run(arms[n].get_joint_positions()).values]
            if target is not None and max(abs(a - b) for a, b in zip(now, target)) <= JOINT_MOVE_TOL_DEG:
                log(f"{n} is already at its wave start, {kf['name']}")
                continue
            if target is not None and max(abs(a - b) for a, b in zip(now[:WRIST], target[:WRIST])) > JOINT_MOVE_TOL_DEG:
                approach = kf.get("approach_values")
                via = dict(kf, values=approach, name=f"{kf['name']} (wrist half a turn round)") if approach else kf
                log(f"Moving {n} to {via['name']} (motion planner, collision-checked)")
                mover.go(via)
                now = [float(v) for v in run(arms[n].get_joint_positions()).values]
            if target is not None:
                if max(abs(a - b) for a, b in zip(now, target)) > JOINT_MOVE_TOL_DEG:
                    log(f"Turning {n}'s wrist to its wave start, {kf['name']} "
                        f"({now[WRIST]:+.1f}° to {target[WRIST]:+.1f}°, joint move, wrist only)")
                    mover.joints_to(dict(kf, joint_move=True))
            else:
                log(f"Moving {n} to its wave start, {kf['name']} (motion planner, collision-checked)")
                mover.go(kf)
        base = {n: [float(v) for v in run(arms[n].get_joint_positions()).values] for n in order}
        log(f"Wave at {WAVE_SPEED:.0f}°/s and {WAVE_ACCEL:.0f}°/s² (normal {DEFAULT_SPEED:.0f}°/s, {DEFAULT_ACCEL:.0f}°/s²)")
        set_motion(run, [arms[n] for n in order], WAVE_MOTION)

        def wrists(offsets, text):
            check()
            log(text)
            run(together([arms[n].move_to_joint_positions(JointPositions(
                values=base[n][:WRIST] + [base[n][WRIST] + offsets[n]] + base[n][WRIST + 1:])) for n in order]))

        for i in range(turns.pop()):
            step = {n: waves[n][i][1] for n in order}
            what = ", ".join(f"{n} {waves[n][i][0]} {waves[n][i][1]:+.1f}°" for n in order)
            wrists(step, f"Wave {i + 1}: {what}, together")
        wrists({n: 0.0 for n in order}, "Every wrist back to its wave start, together")
        check()
        log("Closing every gripper, together")
        run(together([grippers[n].grab() for n in order]))
        check()
        log("Opening every gripper, together")
        run(together([grippers[n].open() for n in order]))
        set_motion(run, [arms[n] for n in order], NORMAL_MOTION)   # planned moves at normal speed
        for n in finish:
            check()
            log(f"Moving {n} to {finish[n]['name']} (motion planner, collision-checked)")
            mover.go(finish[n])
        log("Start up done: " + ", ".join(f"{n} at {finish[n]['name'] if n in finish else starts[n]['name']}"
                                          for n in order) + ". Ready for the sequence.")
    except Exception:
        for n in order:
            try:
                run(arms[n].stop())
            except Exception:
                pass
        try:
            set_motion(run, [arms[n] for n in order], NORMAL_MOTION)
        except Exception:
            pass
        raise


def apply_gripper(run, gripper, action):
    if action == "grab":
        run(gripper.grab())
    elif action == "open":
        run(gripper.open())


class BottleTask:
    def __init__(self, run, arms, grippers, unscrewer, mover):
        self.run, self.arms, self.grippers, self.unscrewer, self.mover = run, arms, grippers, unscrewer, mover
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = {"state": "idle", "step": "", "error": None}

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._lock:
            return dict(self._status)

    def start(self, keyframes, unscrew_params, drop, retreat=None, via=()):
        """keyframes: played in order first; unscrew_params: UnscrewParams; retreat: optional keyframe whose height
        the arm rises to right after the unscrew, straight up and keeping its current orientation (the wrist stays
        where the last twist left it; the keyframe's gripper action is skipped so the cap stays held); drop: the
        trash keyframe, played last. via: keyframes passed through between the retreat and the drop, motion only
        (their gripper actions and delays are skipped, so the cap stays held). One arm moves at a time: the
        motion service plans one move at a time, and each plan only sees the other arm standing still."""
        if self.running:
            return
        self._stop.clear()
        self._set(state="running", step="starting", error=None)
        self._thread = threading.Thread(target=self._sequence, args=(list(keyframes), unscrew_params, drop, retreat, list(via)),
                                        name="bottle-task", daemon=True)
        self._thread.start()

    def stop(self):
        """Interrupt the run (including an unscrew in progress) and stop every arm. Safe to call when idle."""
        self._stop.set()
        self.unscrewer.stop()
        self._halt()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None

    # ---------- internals ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _halt(self):
        for arm in self.arms.values():
            try:
                self.run(arm.stop())
            except Exception:
                pass

    def _check(self):
        if self._stop.is_set():
            raise InterruptedError("stopped")

    def _keyframe(self, label, kf, move_only=False):
        self._check()
        self._set(step=f"{label}: moving {kf['arm']} to {kf['name']}")
        self.mover.go(kf)
        if move_only:
            return
        if kf.get("gripper"):
            self._check()
            self._set(step=f"{label}: gripper {kf['gripper']} at {kf['name']}")
            apply_gripper(self.run, self.grippers[kf["arm"]], kf["gripper"])
        if kf.get("delay", 0) > 0 and self._stop.wait(kf["delay"]):
            raise InterruptedError("stopped")

    def _rise_to(self, kf):
        """Straight up to kf's height, same x, y and orientation as now (no re-orienting after the unscrew)."""
        self._check()
        arm = self.arms[kf["arm"]]
        e = self.run(arm.get_end_position())
        z = self.mover.target(kf)["z"]
        if z <= e.z:
            return  # already at/above it
        self._set(step=f"3/3 retreat: {kf['arm']} up {z - e.z:.0f} mm to {kf['name']} height, holding the cap")
        target = Pose(x=e.x, y=e.y, z=z, o_x=e.o_x, o_y=e.o_y, o_z=e.o_z, theta=e.theta)
        self.mover.to_pose(kf["arm"], target)

    def _sequence(self, keyframes, unscrew_params, drop, retreat, via):
        try:
            for n, kf in enumerate(keyframes, 1):
                self._keyframe(f"1/3 keyframes {n}/{len(keyframes)}", kf)

            self._check()
            self.unscrewer.start(unscrew_params)
            while self.unscrewer.running:
                self._set(step=f"2/3 unscrew: {self.unscrewer.status()['step']}")
                if self._stop.wait(0.2):
                    raise InterruptedError("stopped")
            u = self.unscrewer.status()
            if u["state"] != "done":
                raise RuntimeError(f"unscrew {u['state']} at '{u['step']}': {u['error']}")

            if retreat is not None:
                self._rise_to(retreat)
            for kf in via:
                self._keyframe("3/3 via (holding cap)", kf, move_only=True)
            self._keyframe("3/3 drop", drop)
            self._set(state="done", step="complete")
        except Exception as e:
            self._halt()
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e))
