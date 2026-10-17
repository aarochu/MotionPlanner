"""Bottle-cap detection on a wrist RealSense and the geometry to put a gripper over it.

Shared by the dashboard (streamlit_app.py) and the tracker (cap_tracker.py). Nothing here moves the arm.
"""
import cv2
import numpy as np
from viam.proto.common import Pose, PoseInFrame


# ---------- pure image functions ----------
def circle_features(hsv, edges, u, v, r, ref):
    """Silver fraction and dark fraction inside the circle, and how much of its outline lies on an edge.

    Brightness is judged against `ref`, the scene's bright reference (the white table), not a fixed level, so a
    cap in shade or under different lighting still counts as silver: unsaturated and at least half as bright as
    the table. Dark means under 35 % of it (the dark bottle body)."""
    h, w = edges.shape
    yy, xx = np.ogrid[:h, :w]
    inner = (xx - u) ** 2 + (yy - v) ** 2 <= (0.8 * r) ** 2
    sat, val = hsv[..., 1][inner], hsv[..., 2][inner]
    silver = float(((sat < 60) & (val > 0.5 * ref)).mean())
    dark = float((val < 0.35 * ref).mean())
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)[:, None]
    rad = r + np.arange(-4, 5)[None, :]  # tolerate a few px of radius error
    xs = np.clip((u + rad * np.cos(ang)).astype(int), 0, w - 1)
    ys = np.clip((v + rad * np.sin(ang)).astype(int), 0, h - 1)
    rim = float(edges[ys, xs].any(axis=1).mean())
    return silver, dark, rim


def detect_circles(bgr, min_r, max_r, sensitivity, min_silver, min_rim):
    """Hough circles split into (caps, rejected), each as (u, v, r, score), caps best first.

    A cap is mostly silver (low saturation, bright relative to the table), has almost no dark pixels (the dark blue bottle body and
    neck fail here) and has a crisp outline most of the way round (partial circles inside the cap fail here).
    Circles overlapping a better-scoring cap are duplicates of it and are rejected too."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r, 40),
                             param1=120, param2=sensitivity, minRadius=min_r, maxRadius=max_r)
    if found is None:
        return [], []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ref = max(float(np.percentile(hsv[..., 2], 90)), 1.0)  # brightness of the white table in this frame
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    candidates, rejected = [], []
    for u, v, r in found[0][:12]:
        silver, dark, rim = circle_features(hsv, edges, u, v, r, ref)
        c = (float(u), float(v), float(r), silver * rim * float(r))  # radius term: full cap beats logo-sized circles
        (candidates if silver >= min_silver and dark <= 0.15 and rim >= min_rim else rejected).append(c)
    caps = []
    for c in sorted(candidates, key=lambda c: -c[3]):
        if any(np.hypot(c[0] - k[0], c[1] - k[1]) < max(c[2], k[2]) for k in caps):  # overlaps a better cap
            rejected.append(c)
        else:
            caps.append(c)
    return caps, rejected


def robust_median(hits, min_hits):
    """(circle, n_agreeing): median of the circles whose centers sit near the overall median, or None."""
    if len(hits) < min_hits:
        return None
    arr = np.array(hits)
    med = np.median(arr, axis=0)
    agree = arr[np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1]) < 0.3 * med[2]]
    if len(agree) < min_hits:
        return None
    return tuple(float(x) for x in np.median(agree, axis=0)), len(agree)


def colorize_depth(depth, near=150, far=600):
    scaled = np.clip((depth.astype(np.int32) - near) * 255 // (far - near), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(255 - scaled, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    rgb[depth == 0] = 0  # no reading (too close, too far, or specular)
    return rgb


# ---------- camera + frame system ----------
class CapVision:
    """cam2 capture, cap measurement and gripper geometry, bound to one robot connection.

    `run` executes a coroutine on the connection's event loop and returns its result (thread-safe)."""

    def __init__(self, run, robot, arm, camera, intrinsics, cam="cam2", arm_name="arm2", gripper="gripper2"):
        self.run, self.robot, self.arm, self.camera, self.intr = run, robot, arm, camera, intrinsics
        self.cam, self.arm_name, self.gripper = cam, arm_name, gripper

    def capture(self):
        """Color (BGR) and depth (uint16 mm, aligned to color when the camera has align_color_depth)."""
        imgs, _ = self.run(self.camera.get_images())
        by_name = {i.name: i for i in imgs}
        bgr = cv2.imdecode(np.frombuffer(by_name["color"].data, np.uint8), cv2.IMREAD_COLOR)
        depth = None
        if "depth" in by_name:  # image/vnd.viam.dep: 8-byte magic, width, height (big-endian u64), big-endian u16 mm
            data = by_name["depth"].data
            w, h = int.from_bytes(data[8:16], "big"), int.from_bytes(data[16:24], "big")
            depth = np.frombuffer(data[24:], ">u2").reshape(h, w)
        return bgr, depth

    def detect_stable(self, detect, frames=5, min_hits=3):
        """Best cap over several frames: median of the per-frame winners after dropping ones far from the median,
        so a single bad frame (blur, a partial circle inside the cap) cannot steer the arm.
        Returns (bgr, depth, circle, hits) from the last frame with a cap, or None if too few frames agree."""
        hits, last = [], None
        for _ in range(frames):
            bgr, depth = self.capture()
            caps, _ = detect_circles(bgr, *detect)
            if caps:
                hits.append(caps[0])
                last = (bgr, depth)
        best = robust_median(hits, min_hits)
        return None if best is None else (*last, *best)

    def measure(self, circle, depth, cap_diameter_mm):
        """Camera-frame point (mm) of the circle's center. Uses aligned depth when enough of the disk is valid,
        otherwise the pinhole size estimate Z = fx * D / (2r) (e.g. cap closer than the D435 min range)."""
        u, v, r, _ = circle
        i = self.intr
        z, method, valid = None, "size", 0.0
        if depth is not None:
            h, w = depth.shape
            yy, xx = np.ogrid[:h, :w]
            vals = depth[(xx - u) ** 2 + (yy - v) ** 2 <= (0.6 * r) ** 2]
            valid = float((vals > 0).mean()) if vals.size else 0.0
            if valid >= 0.6:  # below this the valid pixels are mostly background leaking in at the rim
                z, method = float(np.median(vals[vals > 0])), "depth"
        if z is None:
            z = i.focal_x_px * cap_diameter_mm / (2 * r)
        x = (u - i.center_x_px) * z / i.focal_x_px
        y = (v - i.center_y_px) * z / i.focal_y_px
        return {"x": x, "y": y, "z": z, "method": method, "valid": valid}

    def to_frame(self, frame, x=0.0, y=0.0, z=0.0, src=None):
        query = PoseInFrame(reference_frame=src or self.cam, pose=Pose(x=x, y=y, z=z, o_z=1))
        return self.run(self.robot.transform_pose(query, frame)).pose

    def gripper_pixel(self, point):
        """Pixel straight below the gripper at the cap's height: where the cap appears once XY-aligned."""
        cap = self.to_frame("world", point["x"], point["y"], point["z"])
        grip = self.to_frame("world", src=self.gripper)
        p = self.to_frame(self.cam, grip.x, grip.y, cap.z, src="world")
        if p.z <= 0:
            return None
        i = self.intr
        return (i.focal_x_px * p.x / p.z + i.center_x_px, i.focal_y_px * p.y / p.z + i.center_y_px)

    def cap_error(self, point):
        """(cap, grip): cap center and gripper position in the arm's base frame."""
        base = f"{self.arm_name}_origin"
        return self.to_frame(base, point["x"], point["y"], point["z"]), self.to_frame(base, src=self.gripper)

    def offset_target(self, delta):
        """Arm end pose shifted by delta (base frame, mm), orientation unchanged."""
        end = self.run(self.arm.get_end_position())
        return Pose(x=end.x + delta[0], y=end.y + delta[1], z=end.z + delta[2],
                    o_x=end.o_x, o_y=end.o_y, o_z=end.o_z, theta=end.theta)

    def plan_align(self, point, hover_mm, keep_height, max_mm):
        """Arm-base-frame target that puts the gripper over the circle center (orientation unchanged)."""
        cap, grip = self.cap_error(point)
        delta = np.array([cap.x - grip.x, cap.y - grip.y, 0.0 if keep_height else cap.z + hover_mm - grip.z])
        norm = float(np.linalg.norm(delta))
        clamped = norm > max_mm
        if clamped:
            delta *= max_mm / norm
        return {"cap": cap, "cap_world": self.to_frame("world", point["x"], point["y"], point["z"]),
                "grip_world": self.to_frame("world", src=self.gripper), "grip": grip, "delta": delta, "norm": norm,
                "clamped": clamped, "target": self.offset_target(delta)}

    def annotate(self, bgr, circles, rejected=(), grip_px=None):
        """Cyan +: optical center. Green: chosen cap. Magenta X: below the gripper at cap height (line = XY move)."""
        out = bgr.copy()
        cv2.drawMarker(out, (int(self.intr.center_x_px), int(self.intr.center_y_px)), (255, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
        for u, v, r, _ in rejected:
            cv2.circle(out, (int(u), int(v)), int(r), (0, 0, 255), 1)
        for n, (u, v, r, _) in enumerate(circles):
            color = (0, 255, 0) if n == 0 else (160, 160, 160)
            cv2.circle(out, (int(u), int(v)), int(r), color, 4 if n == 0 else 2)
            cv2.circle(out, (int(u), int(v)), 4, color, -1)
        if grip_px is not None:
            g = (int(grip_px[0]), int(grip_px[1]))
            if circles:
                cv2.line(out, g, (int(circles[0][0]), int(circles[0][1])), (255, 0, 255), 2)
            cv2.drawMarker(out, g, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
