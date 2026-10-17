"""Full bottle-opening run: keyframe sequence -> unscrew -> retreat up -> via poses -> drop the cap at the trash pose.

Runs in a background thread so the dashboard stays responsive and Stop / STOP ALL can interrupt it. Each phase
waits for the previous one to finish; if any step fails or is stopped, the arms are stopped and the run ends there.
"""
import threading

from viam.components.arm import JointPositions
from viam.proto.common import Pose

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


def go_to_keyframe(run, arm, kf):
    """Joint keyframes ("values") move in joint space. XYZ keyframes ("pose": x, y, z in mm and the orientation
    vector o_x, o_y, o_z, theta, in the arm's base frame as get_end_position reports) go through the motion planner."""
    if "pose" in kf:
        run(arm.move_to_position(Pose(**{k: float(kf["pose"][k]) for k in POSE_KEYS})))
    else:
        run(arm.move_to_joint_positions(JointPositions(values=kf["values"])))


def apply_gripper(run, gripper, action):
    if action == "grab":
        run(gripper.grab())
    elif action == "open":
        run(gripper.open())


class BottleTask:
    def __init__(self, run, arms, grippers, unscrewer):
        self.run, self.arms, self.grippers, self.unscrewer = run, arms, grippers, unscrewer
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

    def start(self, keyframes, unscrew_params, drop, retreat=None, via=(), alongside=None):
        """keyframes: played in order first; unscrew_params: UnscrewParams; retreat: optional keyframe whose height
        the arm rises to right after the unscrew, straight up and keeping its current orientation (the wrist stays
        where the last twist left it; the keyframe's gripper action is skipped so the cap stays held); drop: the
        trash keyframe, played last. via: keyframes passed through between the retreat and the drop, motion only
        (their gripper actions and delays are skipped, so the cap stays held). alongside: optional keyframe for
        another arm, started at the same time as the first keyframe (e.g. arm2 to its start while arm goes to
        parking). After that first keyframe the run waits until the alongside move has finished too, so from then
        on only one arm moves at a time."""
        if self.running:
            return
        self._stop.clear()
        self._set(state="running", step="starting", error=None)
        self._thread = threading.Thread(target=self._sequence, args=(list(keyframes), unscrew_params, drop, retreat, list(via), alongside),
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
        go_to_keyframe(self.run, self.arms[kf["arm"]], kf)
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
        z = float(kf["pose"]["z"]) if "pose" in kf else None
        if z is None or z <= e.z:
            return  # joint keyframe (no height to read) or already at/above it
        self._set(step=f"3/3 retreat: {kf['arm']} up {z - e.z:.0f} mm to {kf['name']} height, holding the cap")
        target = Pose(x=e.x, y=e.y, z=z, o_x=e.o_x, o_y=e.o_y, o_z=e.o_z, theta=e.theta)
        self.run(arm.move_to_position(target))

    def _move_and_grip(self, kf, errors):
        """For the alongside move's thread: move + gripper action, recording any failure for the main thread."""
        try:
            go_to_keyframe(self.run, self.arms[kf["arm"]], kf)
            if kf.get("gripper") and not self._stop.is_set():
                apply_gripper(self.run, self.grippers[kf["arm"]], kf["gripper"])
        except Exception as e:
            errors.append(e)

    def _sequence(self, keyframes, unscrew_params, drop, retreat, via, alongside):
        side, errors = None, []

        def wait_for_side(reason):
            nonlocal side
            if side is not None:
                if side.is_alive():
                    self._set(step=f"waiting for {alongside['arm']} to reach {alongside['name']} ({reason})")
                side.join()
                side = None
            if errors:
                raise RuntimeError(f"{alongside['arm']} to {alongside['name']}: {errors[0]}")

        try:
            for n, kf in enumerate(keyframes, 1):
                label = f"1/3 keyframes {n}/{len(keyframes)}"
                if n == 1 and alongside is not None:
                    self._check()
                    side = threading.Thread(target=self._move_and_grip, args=(alongside, errors), daemon=True)
                    side.start()  # e.g. arm2 heads to its start pose while arm runs its sequence
                    label += f" (and {alongside['arm']} → {alongside['name']})"
                elif alongside is not None and kf["arm"] == alongside["arm"]:
                    wait_for_side(f"before its {kf['name']}")
                self._keyframe(label, kf)
                if n == 1:
                    wait_for_side(f"{kf['arm']} waits at {kf['name']}")  # nothing else moves until both have arrived
            if alongside is not None:
                wait_for_side("before the unscrew")

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
            if side is not None:
                side.join(timeout=10)
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e))
