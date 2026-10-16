"""Regrasp unscrew: twist the cap off with the wrist joint, re-gripping between twists.

Start with the other arm holding the bottle. The arm first opens the gripper and moves to the start keyframe
(e.g. open_bottle_cap_pose, with the fingers around the cap); without one it starts from where it is. It then
levels the tool to point exactly straight down, pivoting about the gripper so the fingers stay over the cap, and
sets the grip force. Each twist: grab -> unscrew (rotate the last joint by `twist_deg` in the loosening direction)
-> open -> twist back, all at the same height. The last twist instead keeps hold of the cap and lifts it
`final_lift_mm` straight up, and the sequence ends there.

Runs in a background thread so the dashboard stays responsive and Stop / STOP ALL can interrupt it.
Nothing here runs until start() is called.
"""
import math
import threading
from dataclasses import dataclass

import numpy as np
from viam.components.arm import JointPositions
from viam.proto.common import Pose, PoseInFrame, Transform

WRIST = 5  # last joint (Joint 5 in the Viam app): rolls the gripper about the tool axis
_LEVEL_FRAME = "_level_target"  # virtual frame used to check a leveling move before making it


@dataclass
class UnscrewParams:
    twists: int = 4
    twist_deg: float = 90.0           # wrist roll per twist
    loosen_sign: int = -1             # tool pointing down: +J5 turns clockwise seen from above, so loosen is -J5
    release_pos: int = 500            # gripper opening between twists (0 closed .. 850 open), ~50 mm
    final_lift_mm: float = 30.0       # lift with the cap held after the last twist; the sequence ends there
    speed_deg_s: float = 20.0
    normal_speed_deg_s: float = 60.0  # arm2's configured speed_degs_per_sec, restored afterwards
    grip_force_pct: int = 50          # G2 grasp force (1-100 %)
    normal_grip_force_pct: int = 50   # gripper2's configured gripper_force (module default), restored afterwards
    max_level_tilt_deg: float = 15.0  # refuse to level a pose further than this from vertical
    start_pose: str = ""              # keyframe name to move to first ("" = start from the current pose)
    start_joints: tuple = ()          # that keyframe's joint angles (degrees) ...
    start_target: tuple = ()          # ... or its pose (x, y, z, o_x, o_y, o_z, theta) in the arm's base frame
    dry_run: bool = False             # full cycle without a cap: an empty grab is not an error


def describe(p: UnscrewParams):
    """Human-readable step list for the confirmation dialog."""
    turn = "counter-clockwise" if p.loosen_sign < 0 else "clockwise"
    steps = [f"Slow arm to {p.speed_deg_s:g}°/s, set grip force to {p.grip_force_pct}%",
             f"Open gripper to {p.release_pos}"]
    if p.start_joints:
        steps.append(f"Move to **{p.start_pose}** ({', '.join(f'{v:.1f}' for v in p.start_joints)}°)")
    elif p.start_target:
        x, y, z, ox, oy, oz, th = p.start_target
        steps.append(f"Move to **{p.start_pose}**: x {x:.1f}, y {y:.1f}, z {z:.1f} mm, "
                     f"orientation ({ox:.3f}, {oy:.3f}, {oz:.3f}) θ {th:.1f}° (motion planner)")
    steps.append(f"Level the tool to point straight down (pivot at the gripper, roll kept; abort if more than "
                 f"{p.max_level_tilt_deg:g}° off)")
    for i in range(1, p.twists):
        steps.append(f"{i}. Grab → unscrew {p.twist_deg:g}° {turn} (seen from above) → open to {p.release_pos} "
                     "→ twist back (same height)")
    steps.append(f"{p.twists}. Grab → unscrew {p.twist_deg:g}° {turn} → keep holding and lift {p.final_lift_mm:g} mm; "
                 "end there")
    if p.dry_run:
        steps.append("Dry run: empty grabs are allowed")
    steps.append(f"Restore speed to {p.normal_speed_deg_s:g}°/s and grip force to {p.normal_grip_force_pct}%")
    return steps


class Unscrewer:
    def __init__(self, run, robot, arm, gripper, arm_name="arm2", gripper_name="gripper2"):
        self.run, self.robot, self.arm, self.gripper = run, robot, arm, gripper
        self.arm_name, self.gripper_name = arm_name, gripper_name
        self.base = f"{arm_name}_origin"
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

    def check(self, max_tilt_deg=15.0):
        """Reason the unscrew cannot start from the current pose, or None."""
        tilt = self._tilt_deg(self.run(self.arm.get_end_position()))
        if tilt > max_tilt_deg:
            return (f"Tool is {tilt:.1f}° from pointing straight down (limit {max_tilt_deg:g}°); "
                    "re-teach the start pose closer to vertical.")
        return None

    # ---------- leveling geometry (frame system; nothing here moves) ----------
    def _tf(self, src, pose, dst, extra=None):
        return self.run(self.robot.transform_pose(PoseInFrame(reference_frame=src, pose=pose), dst, extra)).pose

    def _point(self, src, x=0.0, y=0.0, z=0.0, dst=None, extra=None):
        p = self._tf(src, Pose(x=x, y=y, z=z, o_z=1), dst or self.base, extra)
        return np.array([p.x, p.y, p.z])

    @staticmethod
    def _tilt_deg(end):
        return math.degrees(math.acos(max(-1.0, min(1.0, -end.o_z))))

    def tool_length(self):
        """Flange -> gripper point distance, from the frame system (gripper2's frame translation)."""
        return float(np.linalg.norm(self._point(self.gripper_name, dst=self.arm_name)))

    def level_plan(self, max_tilt_deg, tool=None, extra=None):
        """Smallest rotation that points the tool straight down, pivoting at the gripper point and keeping roll.

        Returns {"tilt": deg, "target": base-frame flange Pose or None if already level}. The rotation is built in
        the tool's own frame and converted by the frame system, then checked through a virtual frame: the new
        tool axis must point down, the gripper point must stay put and the roll must not change by more than the
        tilt itself. Raises if the pose is too far off or a check fails. `tool`/`extra` let tests use a virtual
        tool frame; by default it is the arm's end effector."""
        tool = tool or self.arm_name
        L = self.tool_length()
        flange = self._point(tool, extra=extra)
        below = self._point(self.base, *(flange - [0, 0, 100]), dst=tool, extra=extra)
        d = below / np.linalg.norm(below)  # straight down, in tool coordinates
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, d[2]))))
        if tilt > max_tilt_deg:
            raise RuntimeError(f"Tool is {tilt:.1f}° from pointing straight down (limit {max_tilt_deg:g}°); "
                               "re-teach the start pose closer to vertical.")
        if tilt < 1.0:  # also below where orientation vectors stop resolving direction reliably
            return {"tilt": tilt, "target": None}
        # Orientation vectors are Rz(lon) Ry(lat) Rz(theta); theta = -lon makes it the pure tilt from z to d.
        lon = math.degrees(math.atan2(d[1], d[0]))
        rel = Pose(x=-L * d[0], y=-L * d[1], z=L * (1 - d[2]), o_x=d[0], o_y=d[1], o_z=d[2], theta=-lon)
        target = self._tf(tool, rel, self.base, extra)
        vt = [Transform(reference_frame=_LEVEL_FRAME,
                        pose_in_observer_frame=PoseInFrame(reference_frame=tool, pose=rel))] + list(extra or [])
        origin_new = self._point(_LEVEL_FRAME, extra=vt)
        axis_new = self._point(_LEVEL_FRAME, z=100, extra=vt) - origin_new
        pivot_shift = np.linalg.norm(self._point(_LEVEL_FRAME, z=L, extra=vt) - self._point(tool, z=L, extra=extra))
        x_old = self._point(tool, x=100, extra=extra) - flange
        x_new = self._point(_LEVEL_FRAME, x=100, extra=vt) - origin_new
        roll = abs((math.degrees(math.atan2(x_new[1], x_new[0]) - math.atan2(x_old[1], x_old[0])) + 180) % 360 - 180)
        down_err = math.degrees(math.acos(max(-1.0, min(1.0, -axis_new[2] / 100))))
        if down_err > 0.5 or pivot_shift > 1.0 or roll > tilt + 1.0:
            raise RuntimeError(f"leveling check failed (axis {down_err:.2f}° off, pivot moved {pivot_shift:.2f} mm, "
                               f"roll changed {roll:.1f}°); not moving")
        return {"tilt": tilt, "target": target, "roll_change": roll}

    # ---------- control ----------
    def start(self, params: UnscrewParams):
        if self.running:
            return
        self._stop.clear()
        self._set(state="running", step="starting", error=None)
        self._thread = threading.Thread(target=self._sequence, args=(params,), name="cap-unscrew", daemon=True)
        self._thread.start()

    def stop(self):
        """Interrupt the sequence and stop the arm and gripper. Safe to call when idle."""
        self._stop.set()
        for part in (self.arm, self.gripper):
            try:
                self.run(part.stop())
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None

    # ---------- internals ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _do(self, label, coro_fn):
        if self._stop.is_set():
            raise InterruptedError("stopped")
        self._set(step=label)
        return self.run(coro_fn())

    def _joints(self):
        return list(self.run(self.arm.get_joint_positions()).values)

    def _rotate_wrist(self, label, delta_deg):
        j = self._joints()
        j[WRIST] += delta_deg
        self._do(label, lambda: self.arm.move_to_joint_positions(JointPositions(values=j)))

    def _lift(self, label, mm):
        if mm == 0:
            return
        e = self.run(self.arm.get_end_position())
        target = Pose(x=e.x, y=e.y, z=e.z + mm, o_x=e.o_x, o_y=e.o_y, o_z=e.o_z, theta=e.theta)
        self._do(label, lambda: self.arm.move_to_position(target))

    def _open(self, label, pos):
        self._do(label, lambda: self.gripper.open() if pos >= 850 else self.gripper.do_command({"set": float(pos)}))

    def _grab(self, label, require=True):
        holding = self._do(label, lambda: self.gripper.grab())
        if require and holding is False:
            raise RuntimeError(f"{label}: gripper closed on nothing; is it around the cap?")

    def _level(self, p):
        problem = self.check(p.max_level_tilt_deg)
        if problem:
            raise RuntimeError(problem)
        if self._tilt_deg(self.run(self.arm.get_end_position())) < 1.0:
            return  # already straight down (also skips the frame-system work)
        plan = self.level_plan(p.max_level_tilt_deg)
        if plan["target"] is not None:
            self._do(f"leveling tool ({plan['tilt']:.1f}° off vertical)",
                     lambda: self.arm.move_to_position(plan["target"]))

    def _sequence(self, p: UnscrewParams):
        twist = p.loosen_sign * p.twist_deg
        try:
            self._do("slowing arm", lambda: self.arm.do_command({"set_speed": p.speed_deg_s}))
            self._do("setting grip force", lambda: self.gripper.do_command({"set_gripper_torque": p.grip_force_pct}))
            self._open("opening gripper", p.release_pos)
            if p.start_joints:
                self._do(f"moving to {p.start_pose}",
                         lambda: self.arm.move_to_joint_positions(JointPositions(values=list(p.start_joints))))
            elif p.start_target:
                x, y, z, ox, oy, oz, th = p.start_target
                target = Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=th)
                self._do(f"moving to {p.start_pose}", lambda: self.arm.move_to_position(target))
            self._level(p)
            for i in range(1, p.twists + 1):
                tag = f"twist {i}/{p.twists}"
                self._grab(f"{tag}: grab", require=not p.dry_run)
                self._rotate_wrist(f"{tag}: unscrewing", twist)
                if i < p.twists:
                    self._open(f"{tag}: open", p.release_pos)
                    self._rotate_wrist(f"{tag}: twisting back", -twist)
            self._lift("lifting cap", p.final_lift_mm)  # still holding it; the sequence ends here
            self._set(state="done", step="complete")
        except Exception as e:
            for part in (self.arm, self.gripper):
                try:
                    self.run(part.stop())
                except Exception:
                    pass
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e))
        finally:
            for what, part, cmd in (("speed", self.arm, {"set_speed": p.normal_speed_deg_s}),
                                    ("grip force", self.gripper, {"set_gripper_torque": p.normal_grip_force_pct})):
                try:
                    self.run(part.do_command(cmd))
                except Exception as e:
                    self._set(error=f"could not restore {what}: {e}")
