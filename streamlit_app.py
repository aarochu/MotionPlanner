"""Streamlit keyframe recorder for Viam arms (arm + arm2), plus cam2 cap detection and alignment.

Config comes from environment variables (see env.sh / run.sh); nothing secret is stored in code.
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import vision as V
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
API_KEY = os.environ["VIAM_API_KEY"]
ARM_GRIPPERS = {"arm": "gripper", "arm2": "gripper2"}
POSES_FILE = Path(os.environ.get("POSES_FILE", "poses.json"))
MAX_JOG_DEG = 15.0
JOINTS = [f"Joint {i}" for i in range(6)]  # 0-based, matching the Viam app
# arm -> (its wrist camera, its gripper). The camera section follows the active arm, and each arm has its own
# detector: arm2 finds the bottle CAP (silver circle, Hough); arm finds the bottle's PINK marker (color blob).
RIGS = {"arm": ("cam", "gripper"), "arm2": ("cam2", "gripper2")}
VISION_CONFIG_FILE = Path(os.environ.get("VISION_CONFIG_FILE", "vision_config.json"))
MAX_ALIGN_MM = 100.0  # longest single alignment move; run Align again to close the rest

st.set_page_config(page_title="Arm Keyframes", page_icon="🦾", layout="wide")


@st.cache_resource
def get_robot():
    """One connection + one background event loop shared across reruns."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=120)

    async def connect():
        opts = RobotClient.Options.with_api_key(
            api_key=API_KEY, api_key_id=API_KEY_ID
        )
        robot = await RobotClient.at_address(ADDRESS, opts)
        arms, grippers = {}, {}
        for arm_name, grip_name in ARM_GRIPPERS.items():
            try:
                arms[arm_name] = Arm.from_robot(robot, arm_name)
                grippers[arm_name] = Gripper.from_robot(robot, grip_name)
            except Exception:  # arm2 may not be wired up yet
                pass
        cams = {}
        for arm_name, (cam_name, _) in RIGS.items():
            try:
                camera = Camera.from_robot(robot, cam_name)
                cams[arm_name] = (camera, (await camera.get_properties()).intrinsic_parameters)
            except Exception:
                pass
        return robot, arms, grippers, cams

    robot, arms, grippers, cams = run(connect())
    return run, robot, arms, grippers, cams


run, ROBOT, ARMS, GRIPPERS, CAM_RIGS = get_robot()


def load_poses():
    return json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else []


def save_poses(poses):
    POSES_FILE.write_text(json.dumps(poses, indent=2))


def joints(arm):
    return list(run(ARMS[arm].get_joint_positions()).values)


def move(arm, values):
    run(ARMS[arm].move_to_joint_positions(JointPositions(values=values)))


def apply_gripper(arm, action):
    if action == "grab":
        run(GRIPPERS[arm].grab())
    elif action == "open":
        run(GRIPPERS[arm].open())


def safely(fn, *args):
    try:
        fn(*args)
        return True
    except Exception as e:
        st.error(str(e))
        return False


# ---------- cam2 vision ----------
def capture():
    """Color (BGR) and depth (uint16 mm, aligned to color when cam2 has align_color_depth) from cam2."""
    imgs, _ = run(CAMERA.get_images())
    by_name = {i.name: i for i in imgs}
    bgr = cv2.imdecode(np.frombuffer(by_name["color"].data, np.uint8), cv2.IMREAD_COLOR)
    depth = None
    if "depth" in by_name:  # image/vnd.viam.dep: 8-byte magic, width, height (big-endian u64), then big-endian u16 mm
        data = by_name["depth"].data
        w, h = int.from_bytes(data[8:16], "big"), int.from_bytes(data[16:24], "big")
        depth = np.frombuffer(data[24:], ">u2").reshape(h, w)
    return bgr, depth


def circle_features(hsv, edges, u, v, r):
    """Silver fraction and dark fraction inside the circle, and how much of its outline lies on an edge."""
    h, w = edges.shape
    yy, xx = np.ogrid[:h, :w]
    inner = (xx - u) ** 2 + (yy - v) ** 2 <= (0.8 * r) ** 2
    sat, val = hsv[..., 1][inner], hsv[..., 2][inner]
    silver = float(((sat < 50) & (val > 110)).mean())
    dark = float((val < 70).mean())
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)[:, None]
    rad = r + np.arange(-4, 5)[None, :]  # tolerate a few px of radius error
    xs = np.clip((u + rad * np.cos(ang)).astype(int), 0, w - 1)
    ys = np.clip((v + rad * np.sin(ang)).astype(int), 0, h - 1)
    rim = float(edges[ys, xs].any(axis=1).mean())
    return silver, dark, rim


def detect_circles(bgr, min_r, max_r, sensitivity, min_silver, min_rim):
    """Hough circles split into (caps, rejected), each as (u, v, r, score), caps best first.

    A cap is mostly silver (low saturation, bright), has almost no dark pixels (the dark blue bottle body and
    neck fail here) and has a crisp outline most of the way round (partial circles inside the cap fail here).
    Caps whose center falls inside a better cap are duplicates of it and are rejected too."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r, 40),
                             param1=120, param2=sensitivity, minRadius=min_r, maxRadius=max_r)
    if found is None:
        return [], []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    candidates, rejected = [], []
    for u, v, r in found[0][:12]:
        silver, dark, rim = circle_features(hsv, edges, u, v, r)
        c = (float(u), float(v), float(r), silver * rim)
        (candidates if silver >= min_silver and dark <= 0.15 and rim >= min_rim else rejected).append(c)
    caps = []
    for c in sorted(candidates, key=lambda c: -c[3]):
        if any((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 < k[2] ** 2 for k in caps):
            rejected.append(c)
        else:
            caps.append(c)
    return caps, rejected


def detect_stable(detect, frames=5, min_hits=3):
    """Best cap over several frames: median of the per-frame winners after dropping ones far from the median,
    so a single bad frame (blur, a partial circle inside the cap) cannot steer the arm.
    Returns (bgr, depth, circle, hits) from the last frame with a cap, or None if too few frames agree."""
    hits, last = [], None
    for _ in range(frames):
        bgr, depth = capture()
        caps, _ = detect_circles(bgr, *detect)
        if caps:
            hits.append(caps[0])
            last = (bgr, depth)
    if len(hits) < min_hits:
        return None
    arr = np.array(hits)
    med = np.median(arr, axis=0)
    agree = arr[np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1]) < 0.3 * med[2]]
    if len(agree) < min_hits:
        return None
    return (*last, tuple(float(x) for x in np.median(agree, axis=0)), len(agree))


def measure(circle, depth, cap_diameter_mm):
    """Camera-frame point (mm) of the circle's center. Uses aligned depth when enough of the disk is valid,
    otherwise falls back to the pinhole size estimate Z = fx * D / (2r) (e.g. cap closer than the D435 min range)."""
    u, v, r, _ = circle
    i = INTRINSICS
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


def to_frame(frame, x=0.0, y=0.0, z=0.0, src=None):
    return run(ROBOT.transform_pose(PoseInFrame(reference_frame=src or CAM, pose=Pose(x=x, y=y, z=z, o_z=1)), frame)).pose


def plan_align(point, hover_mm, keep_height):
    """Arm-base-frame target that puts gripper2 over the circle center (orientation unchanged)."""
    cap = to_frame(f"{CAM_ARM}_origin", point["x"], point["y"], point["z"])
    grip = to_frame(f"{CAM_ARM}_origin", src=CAM_GRIPPER)
    delta = np.array([cap.x - grip.x, cap.y - grip.y, 0.0 if keep_height else cap.z + hover_mm - grip.z])
    norm = float(np.linalg.norm(delta))
    clamped = norm > MAX_ALIGN_MM
    if clamped:
        delta *= MAX_ALIGN_MM / norm
    end = run(ARMS[CAM_ARM].get_end_position())
    target = Pose(x=end.x + delta[0], y=end.y + delta[1], z=end.z + delta[2],
                  o_x=end.o_x, o_y=end.o_y, o_z=end.o_z, theta=end.theta)
    return {"cap": cap, "cap_world": to_frame("world", point["x"], point["y"], point["z"]),
            "grip_world": to_frame("world", src=CAM_GRIPPER), "grip": grip, "delta": delta, "norm": norm,
            "clamped": clamped, "target": target}


def annotate(bgr, circles, rejected=()):
    out = bgr.copy()
    cv2.drawMarker(out, (int(INTRINSICS.center_x_px), int(INTRINSICS.center_y_px)), (255, 255, 0),
                   cv2.MARKER_CROSS, 30, 2)
    for u, v, r, _ in rejected:
        cv2.circle(out, (int(u), int(v)), int(r), (0, 0, 255), 1)
    for n, (u, v, r, _) in enumerate(circles):
        color = (0, 255, 0) if n == 0 else (160, 160, 160)
        cv2.circle(out, (int(u), int(v)), int(r), color, 4 if n == 0 else 2)
        cv2.circle(out, (int(u), int(v)), 4, color, -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def colorize_depth(depth, near=150, far=600):
    scaled = np.clip((depth.astype(np.int32) - near) * 255 // (far - near), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(255 - scaled, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    rgb[depth == 0] = 0  # no reading (too close, too far, or specular)
    return rgb


# ---------- sidebar ----------
with st.sidebar:
    st.title("🦾 Arm Keyframes")
    arm = st.selectbox("Active arm", list(ARMS), help="Jog, gripper and Save act on this arm.")
    if st.button("🛑 STOP ALL", type="primary", use_container_width=True):
        for a in ARMS.values():
            run(a.stop())
        st.toast("Stopped all arms")
    st.caption(f"Connected to `{ADDRESS}`")
    st.caption("Keep the e-stop within reach.")


# the camera helpers below read these; they follow the active arm
CAM_ARM = arm
CAM, CAM_GRIPPER = RIGS[arm]
CAMERA, INTRINSICS = CAM_RIGS.get(arm, (None, None))


# ---------- live joints ----------
@st.fragment(run_every=1.0)
def live_joints(arm_name):
    st.subheader(f"Live joints — {arm_name}")
    try:
        vals = joints(arm_name)
        for col, name, v in zip(st.columns(6), JOINTS, vals):
            col.metric(name, f"{v:.1f}°")
        p = run(ARMS[arm_name].get_end_position())
        st.caption(
            f"End effector — x {p.x:.1f}, y {p.y:.1f}, z {p.z:.1f} mm · "
            f"orientation o=({p.o_x:.3f}, {p.o_y:.3f}, {p.o_z:.3f}), θ {p.theta:.1f}°"
        )
    except Exception as e:
        st.warning(f"Could not read joints: {e}")


live_joints(arm)

# ---------- controls ----------
jog_col, grip_col, save_col = st.columns([3, 2, 3])

with jog_col:
    with st.container(border=True):
        st.subheader("Jog")
        step = st.select_slider("Step (degrees)", [0.5, 1, 2, 5, 10], value=2)
        for i, name in enumerate(JOINTS):
            label, minus, plus = st.columns([2, 1, 1])
            label.markdown(f"**{name}**")
            for btn_col, sign, sym in ((minus, -1, "−"), (plus, 1, "+")):
                if btn_col.button(sym, key=f"jog{i}{sym}", use_container_width=True):
                    delta = sign * step
                    if abs(delta) > MAX_JOG_DEG:
                        st.error(f"Max jog is {MAX_JOG_DEG}°")
                    else:
                        def do_jog(i=i, delta=delta):
                            v = joints(arm)
                            v[i] += delta
                            move(arm, v)
                        safely(do_jog)

with grip_col:
    with st.container(border=True):
        st.subheader("Gripper")
        g_open, g_grab = st.columns(2)
        if g_open.button("Open", use_container_width=True):
            safely(apply_gripper, arm, "open")
        if g_grab.button("Grab", use_container_width=True):
            safely(apply_gripper, arm, "grab")

with save_col:
    with st.container(border=True):
        st.subheader("Save keyframe")
        name = st.text_input("Name", placeholder="e.g. holder_hold_pose")
        g_action = st.selectbox(
            "Gripper action after arriving", ["none", "grab", "open"], help="Played when this keyframe runs."
        )
        delay = st.number_input("Delay after (seconds)", min_value=0.0, max_value=600.0, step=0.5, value=0.0,
                                help="Pause after this keyframe's move and gripper action, before the next one.")
        if st.button(f"💾 Save current pose of {arm}", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Give the keyframe a name.")
            else:
                def do_save():
                    pose = {"name": name.strip(), "arm": arm, "values": joints(arm),
                            "gripper": "" if g_action == "none" else g_action, "delay": delay}
                    save_poses([p for p in load_poses() if p["name"] != pose["name"]] + [pose])
                if safely(do_save):
                    st.session_state[f"delay_{name.strip()}"] = delay
                    st.toast(f"Saved '{name.strip()}'")
                    st.rerun()


# ---------- confirmation dialogs ----------
@st.dialog("Confirm move")
def confirm_go(idx):
    p = load_poses()[idx]
    st.write(f"Move **{p['arm']}** to **{p['name']}**?")
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, move", type="primary"):
        def go():
            move(p["arm"], p["values"])
            apply_gripper(p["arm"], p.get("gripper"))
        with st.spinner("Moving…"):
            ok = safely(go)
        if ok:
            st.rerun()


@st.dialog("Run sequence")
def confirm_run():
    poses = load_poses()
    st.write(f"Play all **{len(poses)}** keyframes in order, one at a time?")
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, run", type="primary"):
        progress = st.progress(0.0)
        for n, p in enumerate(poses, 1):
            st.write(f"{n}/{len(poses)}: {p['arm']} → {p['name']}")
            def step_fn(p=p):
                move(p["arm"], p["values"])
                apply_gripper(p["arm"], p.get("gripper"))
            if not safely(step_fn):
                return
            wait = p.get("delay", 0.0)
            if wait > 0 and n < len(poses):
                st.write(f"⏱ waiting {wait:g} s")
                time.sleep(wait)
            progress.progress(n / len(poses))
        st.success("Sequence complete")


# ---------- typed joint targets ----------
@st.dialog("Confirm move")
def confirm_targets(arm_name, targets):
    now = joints(arm_name)
    st.write(f"Move **{arm_name}** to these joint angles?")
    st.table({
        "Joint": JOINTS,
        "Now (°)": [f"{v:.2f}" for v in now],
        "Target (°)": [f"{v:.2f}" for v in targets],
        "Change (°)": [f"{t - n:+.2f}" for t, n in zip(targets, now)],
    })
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, move", type="primary"):
        with st.spinner("Moving…"):
            ok = safely(move, arm_name, targets)
        if ok:
            st.rerun()


def load_current_into_targets():
    for i, v in enumerate(joints(arm)):
        st.session_state[f"tgt{i}"] = round(v, 2)


if st.session_state.get("tgt_arm") != arm:  # arm switched (or first load): start from its current pose
    load_current_into_targets()
    st.session_state["tgt_arm"] = arm

with st.container(border=True):
    st.subheader("Move to joint angles")
    cols = st.columns(6)
    targets = [
        col.number_input(name, min_value=-360.0, max_value=360.0, step=1.0, format="%.2f", key=f"tgt{i}")
        for i, (col, name) in enumerate(zip(cols, JOINTS))
    ]
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Load current", on_click=load_current_into_targets, use_container_width=True)
    if b2.button("Move", type="primary", use_container_width=True):
        confirm_targets(arm, targets)

# ---------- cam2 feed, circle detection and alignment ----------
@st.dialog("Align to circle", width="large")
def confirm_align(settings):
    if st.session_state.get("align_yes") and "align_plan" in st.session_state:
        # "Yes, move" reruns this dialog: execute the plan that was shown, not a fresh detection
        with st.spinner("Moving…"):
            ok = safely(lambda: run(ARMS[CAM_ARM].move_to_position(st.session_state.pop("align_plan")["target"])))
        if ok:
            st.rerun()
        return
    try:
        found = detect_stable(settings["detect"])
        if found is None:
            st.error("No cap found consistently across 5 frames. Adjust the detection settings and try again.")
            return
        bgr, depth, circle, hits = found
        point = measure(circle, depth, settings["cap_d"])
        plan = plan_align(point, settings["hover"], settings["keep_height"])
    except Exception as e:
        st.error(str(e))
        return
    st.image(annotate(bgr, [circle]), use_container_width=True)
    st.caption(f"Cap at ({circle[0]:.0f}, {circle[1]:.0f}) px, r {circle[2]:.0f} px · median of {hits}/5 frames")
    dx, dy, dz = plan["delta"]
    cw, g = plan["cap_world"], plan["grip_world"]
    st.table({
        "": ["Circle center (world)", f"{CAM_GRIPPER} now (world)", "Move", "Distance from"],
        "x (mm)": [f"{cw.x:.1f}", f"{g.x:.1f}", f"{dx:+.1f}", ""],
        "y (mm)": [f"{cw.y:.1f}", f"{g.y:.1f}", f"{dy:+.1f}", ""],
        "z (mm)": [f"{cw.z:.1f}", f"{g.z:.1f}", f"{dz:+.1f}", "depth" if point["method"] == "depth" else "cap size"],
    })
    if point["method"] == "size":
        st.warning(f"Depth only covers {point['valid']:.0%} of the circle (likely closer than the camera's "
                   f"minimum range), so distance is estimated from a {settings['cap_d']:g} mm cap diameter.")
    if plan["clamped"]:
        st.info(f"Full correction is {plan['norm']:.0f} mm; this move is limited to {MAX_ALIGN_MM:.0f} mm. "
                "Run Align again afterwards.")
    if not settings["keep_height"] and plan["grip"].z + dz < plan["cap"].z + 20:
        st.error("Target would put the gripper within 20 mm of the circle. Increase the hover height.")
        return
    st.caption("Keep the e-stop in reach. Orientation is kept; only the position changes.")
    st.session_state["align_plan"] = plan
    st.button("Yes, move", type="primary", key="align_yes")


def use_gripper_height():
    st.session_state["m_plane"] = float(V.world_point(run, ROBOT, ARM_GRIPPERS[arm])[2])


def locate_result(cap, cfg):
    """Position of the pink marker from the last Locate capture (read-only)."""
    if cap is None or cap["arm"] != CAM_ARM:
        st.info("Press **Locate bottle** to find the pink marker and get its world position.")
        return
    _, hit, cands = V.detect(cap["bgr"], cfg)
    st.caption(f"Located from the frame captured at {cap['time']}. Locate again after moving the arm.")
    if not hit:
        st.warning("No pink marker in that frame. Adjust the pink detection settings.")
        return
    depth_mode = cfg["method"].startswith("Depth")
    stats = None
    if depth_mode:
        world, stats, note = V.depth_to_world(cap, hit, cfg["radius"], cfg["depth_lo"], cfg["depth_hi"])
    else:
        world, note = V.pixel_to_world(cap, *hit["px"], cfg["plane_z"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pixel (u, v)", f"{hit['px'][0]:.0f}, {hit['px'][1]:.0f}")
    m2.metric("Blob size (px)", f"{hit['bbox'][2]} × {hit['bbox'][3]}")
    if world is None:
        st.error(note)
    else:
        m3.metric("World x, y (mm)", f"{world[0]:.1f}, {world[1]:.1f}")
        m4.metric("World z (mm)", f"{world[2]:.1f}")
        if note:
            st.warning(note)
    if stats and "median" in stats:
        st.caption(f"Marker depth: median {stats['median']:.0f} mm, spread {stats['spread']:.0f} mm, "
                   f"{stats['valid_frac']:.0%} of pixels valid ({stats['n']} px).")
    if depth_mode:
        st.caption("Depth mode needs `align_color_depth: true` on this camera in the Viam config "
                   "(cam2 has it; cam did not when last checked), otherwise depth won't line up with the image.")


if CAMERA is not None and CAM_ARM in ARMS:
    st.divider()
    st.subheader(f"Camera — {CAM} on {CAM_ARM}")
    marker_mode = CAM_ARM == "arm"   # arm: pink bottle marker (read-only)   arm2: bottle cap (+ Align)
    feed_col, ctrl_col = st.columns([3, 1])
    with ctrl_col:
        live = st.toggle("Live feed", value=True)
        show_depth = st.toggle("Show depth", value=False)
        if not marker_mode:
            with st.expander("Detection", expanded=False):
                min_r, max_r = st.slider("Radius (px)", 10, 400, (40, 250))
                sensitivity = st.slider("Strictness", 20, 120, 50, help="Hough accumulator threshold; higher finds fewer circles.")
                min_silver = st.slider("Min silver", 0.0, 1.0, 0.5, 0.05,
                                       help="Share of the circle that is bright and unsaturated. Cap ≈ 0.8, bottle ≈ 0.15.")
                min_rim = st.slider("Min outline", 0.0, 1.0, 0.55, 0.05,
                                    help="Share of the circle's outline on an image edge. Cap ≈ 0.7–0.8; partial circles ≈ 0.4.")
                show_rejected = st.toggle("Show rejected circles", value=False)
                cap_d = st.number_input("Cap diameter (mm)", 5.0, 150.0, 30.0, 0.5,
                                        help="Used to estimate distance when depth has no reading on the circle.")
            keep_height = st.radio("Align", ["XY only (keep height)", "XY + hover above"], index=0) == "XY only (keep height)"
            hover = st.number_input("Hover above circle (mm)", 20.0, 300.0, 80.0, 5.0, disabled=keep_height)
            settings = {"detect": (min_r, max_r, sensitivity, min_silver, min_rim),"cap_d": cap_d, "hover": hover, "keep_height": keep_height}
            if st.button(f"🎯 Align {CAM_ARM} to circle", type="primary", use_container_width=True):
                confirm_align(settings)
        else:
            cfg = V.load_config(VISION_CONFIG_FILE)
            with st.expander("Pink detection", expanded=False):
                hue = st.slider("Hue range", 0, 179, (cfg["hue_lo"], cfg["hue_hi"]), key="m_hue",
                                help="OpenCV hue 0–179. The pink note measures about 173–175.")
                sat_min = st.slider("Min saturation", 0, 255, cfg["sat_min"], key="m_sat")
                val_min = st.slider("Min brightness", 0, 255, cfg["val_min"], key="m_val")
                min_area = st.slider("Min blob area (px)", 100, 100000, cfg["min_area"], step=100, key="m_area")
                use_shape = st.checkbox("Enable shape check", value=bool(cfg["use_shape"]), key="m_shape",
                                        help="Off: the largest pink blob wins. On: the blob must also look like a rectangle.")
                min_rect = st.slider("Min rectangularity", 0.0, 1.0, float(cfg["min_rect"]), step=0.05, key="m_rect",
                                     disabled=not use_shape)
                max_aspect = st.slider("Max aspect ratio", 1.0, 5.0, float(cfg["max_aspect"]), step=0.1,
                                       key="m_aspect", disabled=not use_shape)
            method = st.radio("Position method", list(V.METHODS), key="m_method", index=list(V.METHODS).index(cfg["method"]),
                              help="Depth uses the marker's measured distance. Plane assumes a known grasp height and "
                                   "needs the camera to look downward.")
            radius, (d_lo, d_hi), plane_z = cfg["radius"], (cfg["depth_lo"], cfg["depth_hi"]), cfg["plane_z"]
            if method.startswith("Depth"):
                radius = st.number_input("Bottle radius (mm)", 5.0, 150.0, float(cfg["radius"]), 1.0, key="m_radius",
                                         help="Distance from the marker surface to the bottle's axis.")
                d_lo, d_hi = st.slider("Valid depth (mm)", 100, 3000, (cfg["depth_lo"], cfg["depth_hi"]), step=10,
                                       key="m_depth")
            else:
                st.session_state.setdefault("m_plane", cfg["plane_z"])
                plane_z = st.number_input("Grasp point z (mm, world)", step=1.0, format="%.1f", key="m_plane")
                st.button("Use current gripper height", on_click=use_gripper_height)
            live_cfg = {"hue_lo": hue[0], "hue_hi": hue[1], "sat_min": sat_min, "val_min": val_min,
                        "min_area": min_area, "use_shape": use_shape, "min_rect": min_rect, "max_aspect": max_aspect,
                        "method": method, "radius": radius, "depth_lo": d_lo, "depth_hi": d_hi, "plane_z": plane_z}
            if st.button("💾 Save as default", use_container_width=True,
                         help=f"Writes these pink-detection settings to {VISION_CONFIG_FILE}."):
                V.save_config(VISION_CONFIG_FILE, live_cfg)
                st.toast(f"Saved to {VISION_CONFIG_FILE}")
            settings = {"cfg": live_cfg}
            if st.button("📍 Locate bottle", type="primary", use_container_width=True,
                         help="Read-only: grabs a frame, finds the pink marker and reports its world position. "
                              "Nothing moves."):
                try:
                    with st.spinner("Capturing…"):
                        st.session_state["locate"] = V.capture_frame(run, ROBOT, CAMERA, CAM, CAM_ARM)
                except Exception as e:
                    st.error(f"Capture failed: {e}")

    @st.fragment(run_every=1.0 if live else None)
    def cam_feed_cap(settings):
        try:
            bgr, depth = capture()
            circles, rejected = detect_circles(bgr, *settings["detect"])
        except Exception as e:
            st.warning(f"Could not read {CAM}: {e}")
            return
        img_col, depth_col = st.columns(2) if show_depth and depth is not None else (st.container(), None)
        img_col.image(annotate(bgr, circles, rejected if show_rejected else ()), use_container_width=True)
        if depth_col is not None:
            depth_col.image(colorize_depth(depth), use_container_width=True, caption="Depth (black = no reading)")
        if circles:
            u, v, r, _ = circles[0]
            p = measure(circles[0], depth, settings["cap_d"])
            st.caption(f"Circle at ({u:.0f}, {v:.0f}) px, r {r:.0f} px · camera frame x {p['x']:.1f}, "
                       f"y {p['y']:.1f}, z {p['z']:.1f} mm ({'depth' if p['method'] == 'depth' else 'from cap size'}"
                       f", depth valid {p['valid']:.0%}) · {len(rejected)} other circles rejected")
        else:
            st.caption(f"No cap found ({len(rejected)} circles rejected).")

    @st.fragment(run_every=1.0 if live else None)
    def cam_feed_marker(settings):
        try:
            bgr, depth = capture()
            _, hit, cands = V.detect(bgr, settings["cfg"])
        except Exception as e:
            st.warning(f"Could not read {CAM}: {e}")
            return
        img_col, depth_col = st.columns(2) if show_depth and depth is not None else (st.container(), None)
        img_col.image(cv2.cvtColor(V.draw_detections(bgr, hit, cands), cv2.COLOR_BGR2RGB), use_container_width=True)
        if depth_col is not None:
            depth_col.image(colorize_depth(depth), use_container_width=True, caption="Depth (black = no reading)")
        if hit:
            st.caption(f"Pink marker at ({hit['px'][0]:.0f}, {hit['px'][1]:.0f}) px, "
                       f"{hit['bbox'][2]} × {hit['bbox'][3]} px · {len(cands) - 1} other blob(s)")
        else:
            st.caption(f"No pink marker found ({len(cands)} blob(s) rejected by the shape check)." if cands
                       else "No pink marker found.")

    with feed_col:
        if marker_mode:
            cam_feed_marker(settings)
            locate_result(st.session_state.get("locate"), settings["cfg"])
        else:
            cam_feed_cap(settings)


# ---------- keyframe table ----------
def set_delay(pose_name):
    all_poses = load_poses()
    for q in all_poses:
        if q["name"] == pose_name:
            q["delay"] = st.session_state[f"delay_{pose_name}"]
    save_poses(all_poses)



st.divider()
st.subheader("Keyframes (play order, top → bottom)")
poses = load_poses()
if not poses:
    st.info("No keyframes yet. Jog the arm to a pose and save it above.")
else:
    COLS = [0.4, 2, 1, 4, 1, 1.4, 3]
    header = st.columns(COLS)
    for col, text in zip(header, ["#", "Name", "Arm", "Joints (deg)", "Gripper", "Delay after (s)", ""]):
        col.markdown(f"**{text}**")
    for i, p in enumerate(poses):
        row = st.columns(COLS)
        row[0].write(i + 1)
        row[1].write(p["name"])
        row[2].write(p.get("arm", "arm"))
        row[3].code("  ".join(f"{v:7.1f}" for v in p["values"]), language=None)
        row[4].write(p.get("gripper") or "–")
        row[5].number_input(
            "Delay after (s)", min_value=0.0, max_value=600.0, step=0.5, value=float(p.get("delay", 0.0)),
            key=f"delay_{p['name']}", label_visibility="collapsed", on_change=set_delay, args=(p["name"],),
        )
        go_c, up_c, down_c, del_c = row[6].columns(4)
        if go_c.button("Go", key=f"go{i}"):
            confirm_go(i)
        if up_c.button("↑", key=f"up{i}", disabled=i == 0):
            poses[i - 1], poses[i] = poses[i], poses[i - 1]
            save_poses(poses)
            st.rerun()
        if down_c.button("↓", key=f"dn{i}", disabled=i == len(poses) - 1):
            poses[i + 1], poses[i] = poses[i], poses[i + 1]
            save_poses(poses)
            st.rerun()
        if del_c.button("✕", key=f"del{i}"):
            st.session_state.pop(f"delay_{p['name']}", None)
            save_poses(poses[:i] + poses[i + 1:])
            st.rerun()

    st.caption(f"Total programmed delay: {sum(q.get('delay', 0.0) for q in poses[:-1]):g} s "
               "(the last keyframe's delay is ignored)")
    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True):
        confirm_run()

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {[round(v, 3) for v in p['values']]}, {p.get('delay', 0.0):g}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")
