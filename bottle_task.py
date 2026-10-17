"""Full bottle-opening run: keyframe sequence -> unscrew -> retreat up -> via poses -> drop the cap at the trash pose.

Runs in a background thread so the dashboard stays responsive and Stop / STOP ALL can interrupt it. Each phase
waits for the previous one to finish; if any step fails or is stopped, the arms are stopped and the run ends there.
"""
import threading

from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from kinematics import Kinematics

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


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
        self.to_pose(kf["arm"], self.target(kf))


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
