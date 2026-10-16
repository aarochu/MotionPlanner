"""Bottle detection and pixel -> world conversion.

Pure functions with no Streamlit code, so the same detection can run in a standalone sequence script.
The robot connection is passed in (`run` executes a coroutine and returns its result).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from viam.proto.common import Pose, PoseInFrame

METHODS = ("Depth (camera looking level)", "Plane (camera looking down)")

DEFAULTS = {
    # color
    "hue_lo": 165, "hue_hi": 179, "sat_min": 100, "val_min": 80, "min_area": 1500,
    # shape gates (a candidate must pass both when use_shape is on)
    "use_shape": False, "min_rect": 0.8, "max_aspect": 1.6,
    # position
    "method": METHODS[0], "radius": 40.0, "depth_lo": 200, "depth_hi": 1500, "plane_z": -97.0,
}


# ---------- config ----------
def load_config(path):
    """Built-in defaults, overridden by whatever is in the JSON file."""
    cfg = dict(DEFAULTS)
    path = Path(path)
    if path.exists():
        try:
            cfg.update({k: v for k, v in json.loads(path.read_text()).items() if k in cfg})
        except (OSError, ValueError):
            pass
    if cfg["method"] not in METHODS:
        cfg["method"] = METHODS[0]
    return cfg


def save_config(path, cfg):
    Path(path).write_text(json.dumps({k: cfg[k] for k in DEFAULTS}, indent=2) + "\n")


# ---------- capture ----------
def world_point(run, robot, frame, x=0.0, y=0.0, z=0.0):
    """A point given in `frame` -> world coordinates (mm), using the machine's frame system."""
    q = PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0))
    r = run(robot.transform_pose(q, "world")).pose
    return np.array([r.x, r.y, r.z])


def capture_frame(run, robot, cam, cam_frame, arm_name):
    """One color frame (+ depth) plus the camera's pose and intrinsics at this instant. Read-only."""
    async def grab():
        images, _ = await cam.get_images()
        return images, await cam.get_properties()

    images, props = run(grab())
    pick = next((im for im in images if "color" in (getattr(im, "name", "") or "").lower()), None)
    if pick is None:
        pick = next(im for im in images if "jpeg" in im.mime_type or "png" in im.mime_type)
    bgr = cv2.imdecode(np.frombuffer(pick.data, np.uint8), cv2.IMREAD_COLOR)

    dep = next((im for im in images if "dep" in im.mime_type), None)
    depth = None
    if dep is not None:
        depth = np.asarray(dep.bytes_to_depth_array(), dtype=np.uint16)  # mm; 0 / 65535 = no reading
        if depth.shape != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    k = props.intrinsic_parameters
    # Camera axes expressed in world, so later pixel -> world needs no further robot calls
    # (and stays correct even if the arm moves after the capture).
    origin = world_point(run, robot, cam_frame)
    axes = np.stack([(world_point(run, robot, cam_frame, *a) - origin) / 1000.0
                     for a in ((1000, 0, 0), (0, 1000, 0), (0, 0, 1000))], axis=1)
    return {"bgr": bgr, "depth": depth, "arm": arm_name, "fx": k.focal_x_px, "fy": k.focal_y_px,
            "cx": k.center_x_px, "cy": k.center_y_px, "origin": origin, "R": axes,
            "time": time.strftime("%H:%M:%S")}


# ---------- detection: color, then shape ----------
def color_mask(bgr, cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((cfg["hue_lo"], cfg["sat_min"], cfg["val_min"])),
                       np.array((cfg["hue_hi"], 255, 255)))
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def shape_metrics(contour):
    """How rectangular a blob is.

    rectangularity: blob area / area of its tightest rotated rectangle (1.0 = perfect rectangle)
    aspect:         long side / short side of that rectangle (1.0 = square)
    solidity:       blob area / area of its convex hull (low = ragged or holey)
    """
    area = cv2.contourArea(contour)
    (_, (w, h), _) = cv2.minAreaRect(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    short, long_ = min(w, h), max(w, h)
    return {
        "area": area,
        "rectangularity": area / (w * h) if w * h > 0 else 0.0,
        "aspect": long_ / short if short > 0 else float("inf"),
        "solidity": area / hull_area if hull_area > 0 else 0.0,
    }


def detect(bgr, cfg):
    """Returns (mask, best, candidates).

    candidates: every color blob above min_area, each with its shape metrics, whether it passed the
    shape gates (`accepted`, always True when `use_shape` is off) and why not (`reason`). best: the accepted candidate with the highest
    area * rectangularity * solidity, or None.
    """
    mask = color_mask(bgr, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        if cv2.contourArea(c) < cfg["min_area"]:
            continue
        m = shape_metrics(c)
        reasons = []
        if cfg["use_shape"]:
            if m["rectangularity"] < cfg["min_rect"]:
                reasons.append(f"rectangularity {m['rectangularity']:.2f} < {cfg['min_rect']:.2f}")
            if m["aspect"] > cfg["max_aspect"]:
                reasons.append(f"aspect {m['aspect']:.2f} > {cfg['max_aspect']:.1f}")
        mom = cv2.moments(c)
        candidates.append({
            **m, "contour": c, "bbox": cv2.boundingRect(c),
            "px": (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]),
            "accepted": not reasons, "reason": "; ".join(reasons),
            "score": m["area"] * m["rectangularity"] * m["solidity"],
        })
    accepted = [c for c in candidates if c["accepted"]]
    best = max(accepted, key=lambda c: c["score"], default=None)
    if best is not None:
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [best["contour"]], -1, 255, -1)
        best["filled"] = filled
    return mask, best, candidates


def draw_detections(bgr, best, candidates):
    """Annotated copy: green = chosen, cyan = passed but lower score, red = rejected by shape."""
    out = bgr.copy()
    for c in candidates:
        x, y, w, h = c["bbox"]
        color = (0, 255, 0) if c is best else (255, 255, 0) if c["accepted"] else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3 if c is best else 2)
        cv2.putText(out, f"rect {c['rectangularity']:.2f}  asp {c['aspect']:.2f}", (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if best is not None:
        cv2.drawMarker(out, (int(best["px"][0]), int(best["px"][1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    return out


# ---------- position ----------
def pixel_to_world(cap, u, v, plane_z):
    """Cast the camera ray through pixel (u, v) and intersect it with the horizontal plane z = plane_z."""
    ray = cap["R"] @ np.array([(u - cap["cx"]) / cap["fx"], (v - cap["cy"]) / cap["fy"], 1.0])
    if abs(ray[2]) < 1e-6:
        return None, "Ray is parallel to the plane"
    t = (plane_z - cap["origin"][2]) / ray[2]
    if t <= 0:
        return None, "Plane is behind the camera along this ray: check the plane height and where the camera points"
    note = ""
    if abs(ray[2]) / np.linalg.norm(ray) < 0.2:
        note = "Camera looks nearly parallel to the plane, so x,y is very sensitive to errors. Aim it more downward."
    return cap["origin"] + t * ray, note


def depth_to_world(cap, det, radius, d_lo, d_hi):
    """Median depth over the detected target -> bottle axis position in world (mm).

    The target is the bottle's front surface, so the axis lies about one bottle radius further along the ray."""
    depth = cap.get("depth")
    if depth is None:
        return None, None, "This capture has no depth frame."
    on_blob = det["filled"] > 0
    ok = on_blob & (depth > 0) & (depth < 65535) & (depth >= d_lo) & (depth <= d_hi)
    vals = depth[ok].astype(float)
    stats = {"n": int(vals.size), "valid_frac": vals.size / max(int(on_blob.sum()), 1)}
    if vals.size < 50:
        return None, stats, ("Too few valid depth pixels on the target. Move the camera back (the sensor can't read "
                             "very close surfaces) or widen the valid depth range.")
    d = float(np.median(vals))
    stats.update(median=d, spread=float(np.percentile(vals, 75) - np.percentile(vals, 25)))
    u, v = det["px"]
    surface = np.array([(u - cap["cx"]) / cap["fx"] * d, (v - cap["cy"]) / cap["fy"] * d, d])
    axis = surface + radius * surface / np.linalg.norm(surface)
    note = ""
    if stats["valid_frac"] < 0.3:
        note = "Less than 30% of the target has valid depth, so the distance may be unreliable."
    elif stats["spread"] > 40:
        note = "Depth varies a lot across the target (spread > 40 mm), so part of it may be reading the background."
    return cap["origin"] + cap["R"] @ axis, stats, note


def depth_preview(depth, max_mm=2000):
    scaled = (np.clip(depth, 0, max_mm).astype(np.float32) / max_mm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[(depth == 0) | (depth == 65535)] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
