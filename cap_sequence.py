"""The whole bottle-opening job as one list of retryable steps (runs in a background thread):

  1 arm 1: YOLO pick & place (resets first if needed; ends holding the bottle at the place spot)
  2 arm 2: open gripper2, look: planned move to the look pose (open_bottle_cap_up_pose, over the place spot)
  3 arm 2: lock on the cap: bottle_track tracks the bottle-top circle in x, y until locked
  4 arm 2: down to DOWN_MM below the look height (one straight planned move): fingers around the cap
  5 arm 2: unscrew with the given UnscrewParams from that pose
  6 arm 2: back up to the look height, straight up, keeping the orientation the unscrew left
  7 arm 2: to the trash drop pose (planned move), open the gripper: the cap drops

start(from_step) runs the steps from there to the end, so a step that stopped or failed can be retried and the job
continues. Heights are absolute (from the look pose), so retrying a half-finished move cannot overshoot.
Every move goes through the motion planner; straight moves use the linear constraint and fall back to a plain
planned move if no straight path exists. stop() halts the step, the tracker, the unscrew and both arms.
"""
import threading

from viam.proto.common import Pose

import bottle_track as BT

# arm speed (UFactory DoCommand; the module allows up to 180 deg/s and 1145 deg/s^2). Arm 2's moves in this job run at
# 80 % of that; arm 1's YOLO pick & place runs at the configured normal speed; arm 2 is put back to normal at the end.
FAST = {"set_speed": 144.0, "set_acceleration": 916.0}
NORMAL = {"set_speed": 60.0, "set_acceleration": 381.67}
DOWN_MM = 90.0  # look pose -> fingers around the cap (the look pose is 100 mm above the taught cap pose; stop 1 cm higher)
LINEAR = (5.0, 5.0)


class Job:
    """Steps for the full job. arm1_pick: callable(check, log) doing arm 1's YOLO pick & place, or None (arm 2 only)."""

    def __init__(self, run, mover, tracker, unscrewer, arm2, gripper2, arm1_pick, look_kf, drop_pose, unscrew_params,
                 halt_all, top_z=BT.BOTTLE_TOP_Z, top_d=BT.TOP_D_MM, arm1=None):
        self.run, self.mover, self.tracker, self.unscrewer = run, mover, tracker, unscrewer
        self.arm1 = arm1
        self.arm2, self.gripper2, self.arm1_pick, self.halt_all = arm2, gripper2, arm1_pick, halt_all
        self.look_kf, self.drop_pose, self.unscrew_params = look_kf, drop_pose, unscrew_params
        self.top_z, self.top_d = top_z, top_d
        self.look_z = float(look_kf["pose"]["z"])
        self.steps = [("Arm 1: YOLO pick & place", self._pick),
                      ("Arm 2: look at the bottle top", self._look),
                      ("Arm 2: lock on the cap", self._lock_on),
                      (f"Arm 2: down {DOWN_MM:.0f} mm onto the cap", self._down),
                      ("Arm 2: unscrew", self._unscrew),
                      ("Arm 2: back up holding the cap", self._up),
                      ("Arm 2: drop the cap in the trash", self._drop)]

    def _fast(self, r):
        """Arm 2 at FAST for its moves (re-applied every step: the unscrew sets its own speed and restores normal)."""
        self.run(self.arm2.do_command(FAST))
        r.log(f"arm2 speed {FAST['set_speed']:g} deg/s, acceleration {FAST['set_acceleration']:g} deg/s^2")

    def normal_speed(self):
        self.run(self.arm2.do_command(NORMAL))

    def _straight(self, pose):
        try:
            self.mover.to_pose("arm2", pose, linear=LINEAR)
        except Exception as e:
            if "no direct solutions" not in str(e) and "failed to find path" not in str(e):
                raise
            self.mover.to_pose("arm2", pose)

    def _pick(self, r):
        if self.arm1_pick is None:
            r.log("Skipped (arm 2 only)")
            return
        if self.arm1 is not None:
            self.run(self.arm1.do_command(NORMAL))  # YOLO pick & place at the normal speed
            r.log(f"arm speed {NORMAL['set_speed']:g} deg/s (normal) for the pick & place")
        self.arm1_pick(r.check, r.log)

    def _look(self, r):
        self._fast(r)
        r.log("Opening gripper2")
        self.run(self.gripper2.open())
        r.check()
        r.log(f"arm2 to {self.look_kf['name']}")
        self.mover.go(self.look_kf)

    def _lock_on(self, r):
        self._fast(r)
        self.tracker.start(self.top_z, self.top_d)
        r.wait(lambda: self.tracker.running, lambda: f"tracking: {self.tracker.status().get('step', '')}")
        t = self.tracker.status()
        if t.get("state") != "locked":
            raise RuntimeError(f"cap not locked ({t.get('state')}): {t.get('error') or t.get('step')}")
        r.log(t.get("step", "locked"))

    def _down(self, r):
        self._fast(r)
        e = self.run(self.arm2.get_end_position())
        z = self.look_z - DOWN_MM
        r.log(f"straight down to z {z:.1f} ({e.z - z:.0f} mm)")
        self._straight(Pose(x=e.x, y=e.y, z=z, o_x=e.o_x, o_y=e.o_y, o_z=e.o_z, theta=e.theta))

    def _unscrew(self, r):
        self.unscrewer.start(self.unscrew_params)
        r.wait(lambda: self.unscrewer.running, lambda: f"unscrew: {self.unscrewer.status()['step']}")
        u = self.unscrewer.status()
        if u["state"] != "done":
            raise RuntimeError(f"unscrew {u['state']} at '{u['step']}': {u['error']}")

    def _up(self, r):
        self._fast(r)
        e = self.run(self.arm2.get_end_position())
        if e.z < self.look_z - 1:
            r.log(f"straight up {self.look_z - e.z:.0f} mm to the look height, holding the cap")
            self._straight(Pose(x=e.x, y=e.y, z=self.look_z, o_x=e.o_x, o_y=e.o_y, o_z=e.o_z, theta=e.theta))

    def _drop(self, r):
        # normal speed: at FAST this long move tripped the xArm's own limit ("Linear speed exceeds limit in ServoJ mode")
        self.normal_speed()
        r.log(f"arm2 speed {NORMAL['set_speed']:g} deg/s (normal) for the move to the trash")
        r.log("to the trash drop pose")
        self.mover.to_pose("arm2", self.drop_pose)
        r.check()
        r.log("opening gripper2: the cap drops")
        self.run(self.gripper2.open())


class JobRunner:
    """Runs a Job's steps in a thread; remembers the step that stopped or failed so it can be retried."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.job = None
        self._status = {"state": "idle", "step": None, "names": [], "log": [], "error": None, "failed": None}

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._lock:
            return dict(self._status, log=list(self._status["log"]))

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def log(self, line):
        with self._lock:
            n = self._status["step"]
            self._status["log"].append(f"[{n + 1}] {line}" if n is not None else line)
            self._status["log"] = self._status["log"][-60:]

    def check(self):
        if self._stop.is_set():
            raise InterruptedError("stopped")

    def wait(self, busy, what):
        while busy():
            if self._stop.wait(0.2):
                raise InterruptedError("stopped")
            self._set(detail=what())

    def start(self, job, from_step=0):
        if self.running:
            return
        self.job = job
        self._stop.clear()
        self._set(state="running", step=from_step, names=[n for n, _ in job.steps], error=None, failed=None,
                  detail="", log=[] if from_step == 0 else self._status["log"])
        self._thread = threading.Thread(target=self._run, args=(from_step,), name="bottle-job", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.job is not None:
            self.job.tracker.stop()
            self.job.unscrewer.stop()
            self.job.halt_all()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None

    def _normal_speed(self):
        try:
            self.job.normal_speed()
            self.log("arm2 back to normal speed")
        except Exception as e:
            self.log(f"could not reset arm2's speed: {e}")

    def _run(self, from_step):
        n = from_step
        try:
            for n in range(from_step, len(self.job.steps)):
                self.check()
                name, fn = self.job.steps[n]
                self._set(step=n, detail="")
                self.log(f"start: {name}")
                fn(self)
                self.log("done")
            self._set(state="done", step=None, detail="")
            self._normal_speed()
        except Exception as e:
            try:
                self.job.halt_all()
            except Exception:
                pass
            self._normal_speed()
            self.log(f"{'stopped' if isinstance(e, InterruptedError) else 'error'}: {e}")
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e), failed=n)
