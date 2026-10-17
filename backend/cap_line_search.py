"""Line-search cap detection for arm 2 — an alternative to Hough circles.

Uses known bottle geometry to collapse the 2D circle search into a 1D line
search along the bottle axis:

Knowns
------
- Bottle height: ~1 ft (304 mm)
- The axis / vector through the gripper center passes through the bottle center
- Cap diameter: known (~30-40 mm, Saratoga cap)

Strategy
--------
1. Go ~1 m above the bottle top
2. Travel down the center line looking directly at the cap
3. Use cap size in pixels → distance:  Z = fx * D / (2 * r)
4. Optional template-match / OCR for the 'S' on the cap

Nothing here moves the arm — it's pure image + geometry logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:  # pytesseract or tesseract binary missing
    _HAS_TESSERACT = False


# Bottle height in mm (Saratoga 28 fl oz ≈ 1 ft / 304.8 mm)
BOTTLE_HEIGHT_MM = 304.8

# Saratoga cap diameter in mm
CAP_DIAMETER_MM = 34.0


def _render_letter_template(letter: str, size_px: int = 64) -> np.ndarray:
    """Render a grayscale template of `letter` for template matching."""
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size to center it
    (tw, th), _ = cv2.getTextSize(letter, font, 1.5, 3)
    scale = min((size_px - 10) / tw, (size_px - 10) / th)
    font_scale = scale * 1.5
    thickness = max(1, int(3 * scale))
    (tw, th), _ = cv2.getTextSize(letter, font, font_scale, thickness)
    org = ((size_px - tw) // 2, (size_px + th) // 2)
    cv2.putText(img, letter, org, font, font_scale, 255, thickness, cv2.LINE_AA)
    # Invert so letter is white on black (matches bright cap, dark letter)
    return 255 - img


@dataclass
class CapCandidate:
    u: float          # pixel u
    v: float          # pixel v
    r: float          # pixel radius
    z_mm: float       # estimated distance from camera (mm)
    x_mm: float       # camera-frame x (mm)
    y_mm: float       # camera-frame y (mm)
    score: float      # detection confidence (0-1)
    has_s: bool       # whether the 'S' was detected (OCR/template)
    s_confidence: float  # OCR/template confidence for the 'S'


def size_to_distance(fx_px: float, cap_d_mm: float, r_px: float) -> float:
    """Pinhole distance estimate: Z = fx * D / (2 * r)."""
    if r_px <= 0:
        return float('inf')
    return fx_px * cap_d_mm / (2.0 * r_px)


def detect_s_template(bgr: np.ndarray, roi: tuple[int, int, int, int],
                      template: np.ndarray) -> tuple[bool, float]:
    """Template-match for the letter 'S' inside `roi` (x, y, w, h).

    Returns (found, confidence)."""
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return False, 0.0
    crop = bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Normalize brightness
    gray = cv2.equalizeHist(gray)
    # Resize template to match ROI height proportion
    t_h, t_w = template.shape
    scale = h / t_h * 0.6  # letter ~60% of cap diameter
    t_w2, t_h2 = int(t_w * scale), int(t_h * scale)
    if t_w2 < 4 or t_h2 < 4:
        return False, 0.0
    tmpl = cv2.resize(template, (t_w2, t_h2))
    if t_w2 > w or t_h2 > h:
        return False, 0.0
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    found = max_val > 0.45
    return found, float(max_val)


def detect_s_ocr(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[bool, float]:
    """Use pytesseract to read text inside `roi` (x, y, w, h)."""
    if not _HAS_TESSERACT:
        return False, 0.0
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return False, 0.0
    crop = bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 10 -c tessedit_char_whitelist=Ss'
    try:
        text = pytesseract.image_to_string(thresh, config=config).strip()
        found = 'S' in text.upper()
        return found, 0.8 if found else 0.0
    except Exception:
        return False, 0.0


def line_search_cap(bgr: np.ndarray, depth: np.ndarray | None,
                    intrinsics, bottle_center_px: tuple[float, float],
                    axis_dir_px: tuple[float, float],
                    cap_d_mm: float = CAP_DIAMETER_MM,
                    search_range_px: int = 200,
                    step_px: int = 10,
                    use_ocr: bool = False,
                    template: Optional[np.ndarray] = None) -> list[CapCandidate]:
    """Search along a line in the image for the cap.

    Parameters
    ----------
    bgr : color image
    depth : aligned depth image (may be None)
    intrinsics : camera intrinsics (fx, fy, cx, cy)
    bottle_center_px : (u, v) pixel of bottle center where axis enters image
    axis_dir_px : (du, dv) unit direction along the bottle axis in pixels
    cap_d_mm : known cap diameter in mm
    search_range_px : how far to search along the line (px)
    step_px : step size along the line (px)
    use_ocr : whether to run OCR/template match for the 'S'
    template : pre-rendered 'S' template (if None, a default one is made)

    Returns
    -------
    list of CapCandidate, best first
    """
    if template is None:
        template = _render_letter_template('S')

    fx = intrinsics.focal_x_px
    fy = intrinsics.focal_y_px
    cx = intrinsics.center_x_px
    cy = intrinsics.center_y_px

    bu, bv = bottle_center_px
    du, dv = axis_dir_px
    norm = math.hypot(du, dv)
    if norm == 0:
        return []
    du, dv = du / norm, dv / norm

    candidates: list[CapCandidate] = []

    # Search along the line from the bottle center outward
    for t in range(0, search_range_px + 1, step_px):
        u = bu + du * t
        v = bv + dv * t
        ui, vi = int(round(u)), int(round(v))

        # Stay in image bounds with margin for the cap radius
        max_r_px = int(cap_d_mm / 2 * fx / 100.0) + 20  # rough guess at cap radius in px
        if ui < max_r_px or ui >= bgr.shape[1] - max_r_px:
            continue
        if vi < max_r_px or vi >= bgr.shape[0] - max_r_px:
            continue

        # Crop around this point and look for a circle
        crop_half = max_r_px + 10
        crop = bgr[vi-crop_half:vi+crop_half, ui-crop_half:ui+crop_half]

        # Convert to HSV: cap is silver (low saturation, bright)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        silver_mask = (sat < 80) & (val > 100)

        # Find circular blobs in the silver mask
        silver_u8 = silver_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(silver_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            (cu, cv_), radius = cv2.minEnclosingCircle(cnt)
            if radius < 5 or radius > max_r_px:
                continue

            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue

            # Convert to camera-frame coords using depth or size estimate
            cu_abs = ui - crop_half + cu
            cv_abs = vi - crop_half + cv_

            z_mm = None
            if depth is not None:
                h_d, w_d = depth.shape
                du_i, dv_i = int(round(cu_abs)), int(round(cv_abs))
                if 0 <= du_i < w_d and 0 <= dv_i < h_d:
                    vals = depth[max(0, dv_i-3):dv_i+4, max(0, du_i-3):du_i+4]
                    valid = vals[(vals > 0) & (vals < 5000)]
                    if valid.size >= 3:
                        z_mm = float(np.median(valid))

            if z_mm is None or z_mm <= 0:
                z_mm = size_to_distance(fx, cap_d_mm, radius)

            x_mm = (cu_abs - cx) * z_mm / fx
            y_mm = (cv_abs - cy) * z_mm / fy

            # OCR / template match for 'S' inside the circle
            has_s = False
            s_conf = 0.0
            if use_ocr:
                r_int = int(radius)
                s_roi = (int(cu_abs - r_int), int(cv_abs - r_int),
                         int(2 * r_int), int(2 * r_int))
                has_s, s_conf = detect_s_template(bgr, s_roi, template)
                if not has_s and _HAS_TESSERACT:
                    has_s2, s_conf2 = detect_s_ocr(bgr, s_roi)
                    if has_s2:
                        has_s, s_conf = True, s_conf2

            # Score: circularity * silver fraction, boosted if 'S' found
            silver_frac = float((silver_mask[
                max(0, int(cv - vi + crop_half) - 3):min(crop.shape[0], int(cv - vi + crop_half) + 4),
                max(0, int(cu - ui + crop_half) - 3):min(crop.shape[1], int(cu - ui + crop_half) + 4)
            ] > 0).mean()) if crop.size else 0.0
            score = circularity * silver_frac * (1.5 if has_s else 1.0)

            candidates.append(CapCandidate(
                u=float(cu_abs), v=float(cv_abs), r=float(radius),
                z_mm=z_mm, x_mm=x_mm, y_mm=y_mm,
                score=score, has_s=has_s, s_confidence=s_conf,
            ))

    # Sort by score descending, dedupe overlapping
    candidates.sort(key=lambda c: -c.score)
    deduped: list[CapCandidate] = []
    for c in candidates:
        if any(math.hypot(c.u - k.u, c.v - k.v) < max(c.r, k.r) for k in deduped):
            continue
        deduped.append(c)
    return deduped


def line_search_from_arm_pose(bgr: np.ndarray, depth: np.ndarray | None,
                              intrinsics, bottle_center_world: tuple[float, float, float],
                              gripper_world: tuple[float, float, float],
                              cap_d_mm: float = CAP_DIAMETER_MM,
                              use_ocr: bool = False,
                              template: Optional[np.ndarray] = None) -> list[CapCandidate]:
    """High-level: compute the axis from gripper to bottle, then line-search.

    bottle_center_world : (x, y, z) in arm2 base frame — center of bottle
    gripper_world : (x, y, z) in arm2 base frame — gripper center
    """
    bx, by, bz = bottle_center_world
    gx, gy, gz = gripper_world

    # Direction from gripper to bottle (the axis)
    dx, dy, dz = bx - gx, by - gy, bz - gz
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    if norm == 0:
        return []
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    # Point 1 m above the bottle top along the axis (looking down)
    start_z = bz + BOTTLE_HEIGHT_MM + 1000.0  # 1 m above bottle top
    start_x = bx - dx * 100.0  # small offset back from bottle
    start_y = by - dy * 100.0

    # Project this 3D start point into the camera image
    # We need the camera's pose relative to arm2 base... this is complex
    # For now, fall back to pixel-based line search using image center
    # as bottle center and an estimated axis direction

    # Simpler: use image center as bottle center approx, axis pointing down
    h, w = bgr.shape[:2]
    bottle_center_px = (w / 2, h / 2)
    axis_dir_px = (0.0, 1.0)  # looking down in image

    return line_search_cap(bgr, depth, intrinsics, bottle_center_px,
                           axis_dir_px, cap_d_mm=cap_d_mm,
                           use_ocr=use_ocr, template=template)
