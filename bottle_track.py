"""Top-down bottle tracking for arm2: from the look pose, keep the gripper over the bottle's top circle in x, y.

The look pose (open_bottle_cap_up_pose) has cam2 looking straight down from a fixed height. The bottle's top is at
a known height (arm2's base sits on the table, so the table is z = 0 and the top is at the bottle height), so a
pixel turns into a position by intersecting the camera ray with that plane: no depth and no size guessing. The
circle is found with plain Hough circles, limited to the radius the top must have at that distance.

Tracking moves the finger midpoint (TCP_MM out along the tool axis) over the circle's center in small steps,
height and orientation unchanged, until it is within `tol_mm` for `need` checks in a row ("locked").
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from viam.proto.common import Pose

from cap_vision import robust_median
from kinematics import ov_to_matrix

TCP_MM = 172.0  # finger midpoint, out from the flange along the tool axis (UFactory gripper TCP)


def tcp_of(end, tcp_mm=TCP_MM):
    """(finger midpoint position, rotation matrix) for an arm end pose."""
    m = ov_to_matrix(end.o_x, end.o_y, end.o_z, end.theta)
    return np.array([end.x, end.y, end.z]) + tcp_mm * m[:, 2], m

BOTTLE_TOP_Z = 300.0  # mm above the table = arm2 base frame z of the bottle's top (30 cm bottle)
TOP_D_MM = 25.0       # diameter of the top circle (the cap seen from above)
MAX_START_MM = 120.0  # refuse to track a circle farther than this from the gripper (it is not the bottle)
CALIBRATION_FILE = Path(__file__).with_name("track_calibration.json")
TRACK_LOG = Path(__file__).with_name("track_log.jsonl")  # one line per tracking check/move, for reviewing a run


# ---------- calibration: measured offset with the bottle known to be under the fingers ----------
def load_offset():
    """(dx, dy) mm added to every measured bottle position; (0, 0) if not calibrated."""
    try:
        return tuple(json.loads(CALIBRATION_FILE.read_text())["offset_xy"])
    except Exception:
        return (0.0, 0.0)


def save_offset(offset_xy, note=""):
    CALIBRATION_FILE.write_text(json.dumps({"offset_xy": [float(offset_xy[0]), float(offset_xy[1])], "note": note,
                                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")


# ---------- geometry ----------
def _base_point(vision, x, y, z):
    p = vision.to_frame(f"{vision.arm_name}_origin", x, y, z)
    return np.array([p.x, p.y, p.z])


def ray_to_plane(vision, u, v, z_plane):
    """Arm-base-frame point where the camera ray through pixel (u, v) meets the plane z = z_plane."""
    i = vision.intr
    o = _base_point(vision, 0, 0, 0)
    far = _base_point(vision, 1000 * (u - i.center_x_px) / i.focal_x_px, 1000 * (v - i.center_y_px) / i.focal_y_px, 1000)
    d = far - o
    if abs(d[2]) < 1e-6:
        raise ValueError("camera ray is parallel to the bottle-top plane")
    return o + (z_plane - o[2]) / d[2] * d


def plane_to_pixel(vision, point):
    """Pixel of an arm-base-frame point (None if behind the camera)."""
    base = f"{vision.arm_name}_origin"
    p = vision.to_frame(vision.cam, *point, src=base)
    if p.z <= 0:
        return None
    i = vision.intr
    return i.focal_x_px * p.x / p.z + i.center_x_px, i.focal_y_px * p.y / p.z + i.center_y_px


def expected_radius(vision, z_plane, diameter_mm):
    """Radius in pixels the top circle has at the plane, seen along the optical axis."""
    o = _base_point(vision, 0, 0, 0)
    a = _base_point(vision, 0, 0, 1000)
    d = a - o
    dist = (z_plane - o[2]) / d[2] * 1000 if abs(d[2]) > 1e-6 else 0
    if dist <= 0:
        raise ValueError("the bottle-top plane is not in front of the camera")
    return vision.intr.focal_x_px * diameter_mm / 2 / dist


# ---------- detection ----------
def rim_share(edges, u, v, r):
    h, w = edges.shape
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)[:, None]
    rad = r + np.arange(-3, 4)[None, :]
    xs = np.clip((u + rad * np.cos(ang)).astype(int), 0, w - 1)
    ys = np.clip((v + rad * np.sin(ang)).astype(int), 0, h - 1)
    return float(edges[ys, xs].any(axis=1).mean())


class TopCircleDetect:
    """Callable detector for cap_vision.find_caps: Hough circles with radius r_exp ± tol, best outline first.
    Returns (circles, []) with circles as (u, v, r, rim share)."""

    def __init__(self, r_exp, tol=0.3, strictness=30, min_rim=0.4, near=None, max_px=None):
        self.r_exp, self.tol, self.strictness, self.min_rim = r_exp, tol, strictness, min_rim
        self.near, self.max_px = near, max_px  # only circles within max_px of pixel `near` count

    def __call__(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lo, hi = max(3, int(self.r_exp * (1 - self.tol))), int(self.r_exp * (1 + self.tol)) + 1
        found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(lo, 10),
                                 param1=120, param2=self.strictness, minRadius=lo, maxRadius=hi)
        if found is None:
            return [], []
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        circles = [(float(u), float(v), float(r), rim_share(edges, u, v, r)) for u, v, r in found[0][:10]]
        if self.near is not None and self.max_px is not None:
            far = [c for c in circles if np.hypot(c[0] - self.near[0], c[1] - self.near[1]) > self.max_px]
            circles = [c for c in circles if c not in far]
        else:
            far = []
        good = sorted((c for c in circles if c[3] >= self.min_rim), key=lambda c: -c[3])
        return good, [c for c in circles if c[3] < self.min_rim] + far


# ---------- tracking ----------
class BottleTracker:
    """Background loop: detect the top circle (median of `frames`), turn it into x, y on the bottle-top plane,
    and step the finger midpoint toward it until locked. Stop() halts the loop and the arm."""

    def __init__(self, run, vision, mover, arm_name="arm2"):
        self.run, self.vision, self.mover, self.arm_name = run, vision, mover, arm_name
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = {"state": "idle"}
        self._last_px = None  # pixel of the circle picked last time: the next pick sticks to it

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._lock:
            return dict(self._status)

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def start(self, z_plane=BOTTLE_TOP_Z, diameter_mm=TOP_D_MM, gain=0.6, max_step=20.0, tol_mm=3.0, need=3,
              max_moves=25, frames=5, linear=(5.0, 5.0)):
        if self.running:
            return
        self._stop.clear()
        self._last_px = None
        self._set(state="running", step="starting", error=None, moves=0, err_mm=None, bottle=None, tcp=None)
        args = (z_plane, diameter_mm, gain, max_step, tol_mm, need, max_moves, frames, linear)
        self._thread = threading.Thread(target=self._loop, args=args, name="bottle-tracker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.run(self.vision.arm.stop())
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        if self.status().get("state") == "running":
            self._set(state="stopped", step="stopped by user")

    def _pick(self, circles, r_exp):
        """One circle per frame, the same one every time: near-concentric pairs (the cap's top edge and the
        separated ring below it) collapse to the inner, smaller one; then the circle nearest the previous pick
        wins (the best outline on the first check)."""
        keep = []
        for c in sorted(circles, key=lambda c: c[2]):  # smaller first, so the inner one of a pair survives
            if not any(np.hypot(c[0] - k[0], c[1] - k[1]) < 0.6 * r_exp for k in keep):
                keep.append(c)
        if not keep:
            return None
        if self._last_px is not None:
            return min(keep, key=lambda c: np.hypot(c[0] - self._last_px[0], c[1] - self._last_px[1]))
        return max(keep, key=lambda c: c[3])

    def measure(self, z_plane, diameter_mm, frames=5, calibrated=True):
        """(bottle center on the plane, finger midpoint, circle) or (None, tcp, None) if no steady circle.
        Only circles within MAX_START_MM (on the plane) of the spot under the fingers count; one circle is picked
        per frame (see _pick) and the median of the agreeing picks is used. The calibration offset is added
        unless calibrated=False."""
        v = self.vision
        end = self.run(v.arm.get_end_position())
        tcp, _ = tcp_of(end)
        r_exp = expected_radius(v, z_plane, diameter_mm)
        near = plane_to_pixel(v, (tcp[0], tcp[1], z_plane))
        max_px = r_exp * 2 * MAX_START_MM / diameter_mm  # MAX_START_MM on the plane, in pixels
        detect = TopCircleDetect(r_exp, near=near, max_px=max_px)
        picks = []
        for _ in range(frames):
            bgr, _ = v.capture()
            c = self._pick(detect(bgr)[0], r_exp)
            if c is not None:
                picks.append(c)
        best = robust_median(picks, max(2, frames - 2))
        if best is None:
            return None, tcp, None
        circle = best[0]
        self._last_px = circle[:2]
        bottle = ray_to_plane(v, circle[0], circle[1], z_plane)
        if calibrated:
            off = load_offset()
            bottle = bottle + np.array([off[0], off[1], 0.0])
        return bottle, tcp, circle

    def _log(self, **kw):
        row = {"t": time.strftime("%H:%M:%S")}
        row.update({k: (np.round(np.asarray(v, float), 2).tolist() if not isinstance(v, (int, str)) else v)
                    for k, v in kw.items()})
        try:
            with TRACK_LOG.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def _loop(self, z_plane, diameter_mm, gain, max_step, tol_mm, need, max_moves, frames, linear):
        ok, misses, moves, prev, worse = 0, 0, 0, None, 0
        try:
            while not self._stop.is_set():
                bottle, tcp, circle = self.measure(z_plane, diameter_mm, frames)
                if bottle is None:
                    misses += 1
                    self._set(step=f"no steady circle ({misses}/5)", err_mm=None)
                    if misses >= 5:
                        raise RuntimeError("lost the bottle top: no steady circle in 5 tries")
                    continue
                misses = 0
                err = bottle[:2] - tcp[:2]
                dist = float(np.hypot(*err))
                self._log(moves=moves, bottle=bottle[:2], tcp=tcp[:2], circle=circle[:3], err=err, dist=dist)
                if prev is not None and dist > prev + 2.0:  # moving should bring it closer
                    worse += 1
                    if worse >= 2:
                        raise RuntimeError(f"error grew twice in a row ({prev:.1f} -> {dist:.1f} mm): stopped. "
                                           "Check the camera frame / calibration.")
                else:
                    worse = 0
                prev = dist
                self._set(err_mm=dist, err_xy=err.tolist(), bottle=bottle.tolist(), tcp=tcp.tolist(), circle=circle[:3])
                if dist <= tol_mm:
                    ok += 1
                    self._set(step=f"within {tol_mm:g} mm ({ok}/{need})")
                    if ok >= need:
                        self._set(state="locked", step=f"locked: {dist:.1f} mm from the bottle center")
                        self._log(locked=dist, moves=moves)
                        return
                    continue
                ok = 0
                if moves >= max_moves:
                    raise RuntimeError(f"not locked after {max_moves} moves ({dist:.1f} mm off)")
                step = gain * err
                n = float(np.hypot(*step))
                if n > max_step:
                    step *= max_step / n
                end = self.run(self.vision.arm.get_end_position())
                target = Pose(x=end.x + float(step[0]), y=end.y + float(step[1]), z=end.z,
                              o_x=end.o_x, o_y=end.o_y, o_z=end.o_z, theta=end.theta)
                self._set(step=f"move {moves + 1}: ({step[0]:+.1f}, {step[1]:+.1f}) mm, {dist:.1f} mm off")
                if self._stop.is_set():
                    break
                try:
                    self.mover.to_pose(self.arm_name, target, linear=linear)
                except Exception as e:
                    if "no direct solutions" not in str(e):
                        raise
                    self.mover.to_pose(self.arm_name, target)  # tiny step: a plain planned move is fine
                moves += 1
                self._set(moves=moves)
                self._log(moved=step)
        except Exception as e:
            try:
                self.run(self.vision.arm.stop())
            except Exception:
                pass
            self._set(state="error", error=str(e))
