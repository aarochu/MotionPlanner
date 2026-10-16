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
import bottle_task
import poses_io
import vision as V
from bottle_task import POSE_KEYS, BottleTask
from cap_unscrew import UnscrewParams, Unscrewer, describe
from cap_vision import circle_features, detect_circles  # cap test judged against the table's brightness
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
# What each xArm6 joint does, in plain words
JOINT_DESC = [
    "Base spin: turns the whole arm left/right",
    "Shoulder: tilts the arm forward/back",
    "Elbow: bends the arm to reach in/out",
    "Forearm roll: twists the forearm along its length",
    "Wrist bend: tilts the gripper up/down",
    "Wrist twist: spins the gripper (the unscrew motion)",
]
# arm -> (its wrist camera, its gripper). The camera section follows the active arm, and each arm has its own
# detector: arm2 finds the bottle CAP (silver circle, Hough); arm finds the bottle's PINK marker (color blob).
RIGS = {"arm": ("cam", "gripper"), "arm2": ("cam2", "gripper2")}
VISION_CONFIG_FILE = Path(os.environ.get("VISION_CONFIG_FILE", "vision_config.json"))
# Corrected grab (arm): scan the bottle from SCAN_KEYFRAME, compare with a reference taken with the bottle placed
# correctly, and shift the GRAB_KEYFRAME grab by the difference.
GRAB_REF_FILE = Path(os.environ.get("GRAB_REF_FILE", "grab_reference.json"))
SCAN_KEYFRAME, GRAB_KEYFRAME = "NewMid", "grab_pos"
MAX_GRAB_FIX_MM = 30.0   # bigger misplacement aborts instead of chasing the bottle
PRE_GRAB_MM = 60.0       # pre-grab point this far back along the gripper's approach direction
AT_POSE_TOL_DEG = 1.0    # how close (every joint) the arm must be to a keyframe to count as "at" it
# Coarse first look from PARK_KEYFRAME: confirms the bottle is there and roughly where before the arm leaves,
# then the fine scan at SCAN_KEYFRAME must agree with it.
PARK_KEYFRAME = "parking"
MAX_PARK_FIX_MM = 50.0   # coarse offset beyond this: re-place the bottle before going anywhere
CROSS_CHECK_MM = 15.0    # parking and mid-grab offsets must agree within this
PARK_SCAN_MAX_AGE_S = 300
MAX_ALIGN_MM = 100.0  # longest single alignment move; run Align again to close the rest
# arm2 unscrew and the full run (cap_unscrew.py, bottle_task.py); keyframe names in poses.json
UNSCREW_START_POSE = "open_bottle_cap_pose"  # arm2 keyframe the unscrew sequence moves to first
TRASH_DROP_POSE = "trash_drop_pose"  # full run ends here (cap dropped); not part of its keyframe phase
RETREAT_POSE = "open_bottle_cap_up_pose"  # full run: after unscrewing, rise to this pose's height before the drop
VIA_POSES = ["safe_bottle_start_pose"]  # full run: then pass through these (move only, cap held) on the way to the drop
ALONGSIDE_POSE = "safe_bottle_start_pose"  # full run: arm2 moves here together with arm's first keyframe; arm waits for it

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


@st.cache_resource
def get_unscrewer():
    if "arm2" not in ARMS or "arm2" not in GRIPPERS:
        return None
    return Unscrewer(run, ROBOT, ARMS["arm2"], GRIPPERS["arm2"], "arm2", "gripper2")


UNSCREWER = get_unscrewer()


@st.cache_resource
def get_bottle_task():
    return BottleTask(run, ARMS, GRIPPERS, UNSCREWER) if UNSCREWER is not None else None


BOTTLE_TASK = get_bottle_task()
FULL_RUNNING = BOTTLE_TASK is not None and BOTTLE_TASK.running


def load_poses():
    """Flat list of keyframes (each with its "arm"), in play order. File format: see poses_io.py."""
    return poses_io.load(POSES_FILE)


def save_poses(poses):
    poses_io.save(POSES_FILE, poses)


def joints(arm):
    return list(run(ARMS[arm].get_joint_positions()).values)


def move(arm, values):
    run(ARMS[arm].move_to_joint_positions(JointPositions(values=values)))


def go_to_keyframe(p):
    bottle_task.go_to_keyframe(run, ARMS[p["arm"]], p)  # joints: joint move; pose: motion planner


def keyframe_text(p):
    if "pose" in p:
        q = p["pose"]
        return (f"x {q['x']:.1f}  y {q['y']:.1f}  z {q['z']:.1f}  "
                f"o ({q['o_x']:.3f}, {q['o_y']:.3f}, {q['o_z']:.3f})  θ {q['theta']:.1f}")
    return "  ".join(f"{v:7.1f}" for v in p["values"])


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
        if BOTTLE_TASK is not None:
            BOTTLE_TASK.stop()
        if UNSCREWER is not None:
            UNSCREWER.stop()
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
        for col, name, v, desc in zip(st.columns(6), JOINTS, vals, JOINT_DESC):
            col.metric(name, f"{v:.1f}°")
            col.caption(desc)
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
            label.markdown(f"**{name}**  \n{JOINT_DESC[i]}")
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
        save_as = st.radio("Save as", ["Joints", "XYZ + orientation"], horizontal=True,
                           help="XYZ + orientation stores the end pose in the arm's base frame and moves there "
                                "with the motion planner.")
        if st.button(f"💾 Save current pose of {arm}", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Give the keyframe a name.")
            else:
                def do_save():
                    pose = {"name": name.strip(), "arm": arm}
                    if save_as == "Joints":
                        pose["values"] = joints(arm)
                    else:
                        e = run(ARMS[arm].get_end_position())
                        pose["pose"] = {k: round(getattr(e, k), 4) for k in POSE_KEYS}
                    pose.update({"gripper": "" if g_action == "none" else g_action, "delay": delay})
                    save_poses([p for p in load_poses() if p["name"] != pose["name"]] + [pose])
                if safely(do_save):
                    st.session_state.pop(f"delay_{name.strip()}", None)  # table reads the saved delay
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
            go_to_keyframe(p)
            apply_gripper(p["arm"], p.get("gripper"))
        with st.spinner("Moving…"):
            ok = safely(go)
        if ok:
            st.rerun()


@st.dialog("Run sequence")
def confirm_run(selection):
    poses = poses_io.sequence(load_poses(), selection)
    label = poses_io.RUN_OPTIONS[selection]
    if not poses:
        st.warning(f"No keyframes saved for {label}.")
        return
    counts = {a: sum(p["arm"] == a for p in poses) for a in poses_io.ARMS}
    st.write(f"Play **{len(poses)}** keyframe(s) for **{label}**, one move at a time?")
    if selection == "arm_then_arm2":
        st.caption(f"First all {counts['arm']} arm keyframe(s), then all {counts['arm2']} arm2 keyframe(s). "
                   "The two arms never move at the same time.")
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, run", type="primary"):
        progress = st.progress(0.0)
        for n, p in enumerate(poses, 1):
            if n > 1 and p["arm"] != poses[n - 2]["arm"]:
                st.write(f"— {poses[n - 2]['arm']} done, starting {p['arm']} —")
            st.write(f"{n}/{len(poses)}: {p['arm']} → {p['name']}")
            def step_fn(p=p):
                go_to_keyframe(p)
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
        col.number_input(name, min_value=-360.0, max_value=360.0, step=1.0, format="%.2f", key=f"tgt{i}",
                         help=JOINT_DESC[i])
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
    world, info = V.locate(cap, hit, cfg)       # depth, falling back to marker size when depth is unreliable
    note, stats, depth_mode = info["note"], info["stats"], info["method"] != "plane"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pixel (u, v)", f"{hit['px'][0]:.0f}, {hit['px'][1]:.0f}")
    m2.metric("Blob size (px)", f"{hit['bbox'][2]} × {hit['bbox'][3]}")
    if world is None:
        st.error(note)
    else:
        m3.metric("World x, y (mm)", f"{world[0]:.1f}, {world[1]:.1f}")
        m4.metric("World z (mm)", f"{world[2]:.1f}")
        if depth_mode:
            st.caption(f"Distance {info['z']:.0f} mm from "
                       f"{'depth' if info['method'] == 'depth' else 'marker size (fallback)'}.")
        if note:
            st.warning(note)
    if stats and "median" in stats:
        st.caption(f"Marker depth: median {stats['median']:.0f} mm, spread {stats['spread']:.0f} mm, "
                   f"{stats['valid_frac']:.0%} of pixels valid ({stats['n']} px).")
    if depth_mode:
        st.caption("Depth needs `align_color_depth: true` on this camera in the Viam config so depth lines up "
                   "with the image (both cam and cam2 have it as of the last check).")


def load_grab_ref():
    return json.loads(GRAB_REF_FILE.read_text()) if GRAB_REF_FILE.exists() else {}


def save_grab_ref(ref):
    GRAB_REF_FILE.write_text(json.dumps(ref, indent=2) + "\n")


def at_keyframe(name, arm_name="arm"):
    """(is the arm at this keyframe?, largest joint difference in degrees, the keyframe or None)."""
    kf = next((p for p in load_poses() if p["name"] == name and p.get("arm", "arm") == arm_name), None)
    if kf is None:
        return False, None, None
    if "values" not in kf:  # an XYZ keyframe: this check compares joint angles
        return False, None, kf
    diff = max(abs(a - b) for a, b in zip(joints(arm_name), kf["values"]))
    return diff <= AT_POSE_TOL_DEG, diff, kf


def pose_dict(p):
    return {k: float(getattr(p, k)) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def plan_corrected_grab(ref, measured_xy):
    """Shift the recorded grab pose by (measured - reference) in x, y; keep height and orientation.
    The pre-grab point is PRE_GRAB_MM back along the gripper's approach direction (its orientation vector)."""
    dx, dy = measured_xy[0] - ref["bottle_xy"][0], measured_xy[1] - ref["bottle_xy"][1]
    g = ref["grab_pose"]
    grab = dict(g, x=g["x"] + dx, y=g["y"] + dy)
    o = np.array([g["o_x"], g["o_y"], g["o_z"]])
    o = o / np.linalg.norm(o)
    pre = dict(grab, x=grab["x"] - PRE_GRAB_MM * o[0], y=grab["y"] - PRE_GRAB_MM * o[1],
               z=grab["z"] - PRE_GRAB_MM * o[2])
    return {"dx": dx, "dy": dy, "norm": float(np.hypot(dx, dy)), "grab": grab, "pre": pre}


def measure_bottle_now(cfg):
    return V.measure_stable(run, ROBOT, CAM_RIGS["arm"][0], RIGS["arm"][0], "arm", cfg)


@st.dialog("Corrected grab", width="large")
def confirm_corrected_grab(cfg):
    if st.session_state.get("cgrab_yes") and "cgrab_plan" in st.session_state:
        # "Yes, move" reruns this dialog: execute the plan that was shown, not a fresh measurement
        plan = st.session_state.pop("cgrab_plan")
        def go():
            arm_ = ARMS["arm"]
            run(arm_.move_to_position(Pose(**plan["pre"])))
            run(arm_.move_to_position(Pose(**plan["grab"])))
            apply_gripper("arm", plan["gripper"])
        with st.spinner("Moving: pre-grab, then straight in…"):
            ok = safely(go)
        if ok:
            st.rerun()
        return
    ref = load_grab_ref()
    at_scan, diff, _ = at_keyframe(SCAN_KEYFRAME)
    if not at_scan:
        st.error(f"The arm must be at '{SCAN_KEYFRAME}' to scan (it is {diff:.1f}° away). "
                 f"Press Go on '{SCAN_KEYFRAME}' first." if diff is not None
                 else f"No keyframe named '{SCAN_KEYFRAME}' for arm.")
        return
    with st.spinner("Measuring the bottle over 5 frames…"):
        try:
            m = measure_bottle_now(cfg)
        except Exception as e:
            st.error(f"Measurement failed: {e}")
            return
    if m is None:
        st.error("The pink marker wasn't found consistently (needs 3 of 5 frames agreeing within 15 mm). "
                 "Nothing will move.")
        return
    plan = plan_corrected_grab(ref, m["point"][:2])
    cap, det = m["last"]
    st.image(cv2.cvtColor(V.draw_detections(cap["bgr"], det, [det]), cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"{m['hits']}/{m['frames']} frames agreed (within {m['spread_mm']:.1f} mm) · distance from "
               f"{', '.join(m['methods'])}")
    st.table({
        "": ["Bottle (reference)", "Bottle (now)", "Correction"],
        "x (mm)": [f"{ref['bottle_xy'][0]:.1f}", f"{m['point'][0]:.1f}", f"{plan['dx']:+.1f}"],
        "y (mm)": [f"{ref['bottle_xy'][1]:.1f}", f"{m['point'][1]:.1f}", f"{plan['dy']:+.1f}"],
    })
    ps = st.session_state.get("park_scan")
    if ps and ps.get("offset") is not None and time.time() - ps["time"] <= PARK_SCAN_MAX_AGE_S:
        disagree = float(np.hypot(plan["dx"] - ps["offset"][0], plan["dy"] - ps["offset"][1]))
        if disagree > CROSS_CHECK_MM:
            st.error(f"The parking scan and the {SCAN_KEYFRAME} scan disagree by {disagree:.0f} mm "
                     f"(limit {CROSS_CHECK_MM:.0f} mm), so one of them is wrong. Nothing will move.")
            return
        st.success(f"Parking scan agrees with this scan within {disagree:.0f} mm.")
    else:
        st.info(f"No recent parking scan to cross-check (optional): press Scan at {PARK_KEYFRAME} before going "
                f"to {SCAN_KEYFRAME} for an extra check.")
    if plan["norm"] > MAX_GRAB_FIX_MM:
        st.error(f"The bottle is {plan['norm']:.0f} mm from where it should be, more than the "
                 f"{MAX_GRAB_FIX_MM:.0f} mm limit. Re-place the bottle. Nothing will move.")
        return
    _, _, grab_kf = at_keyframe(GRAB_KEYFRAME)
    plan["gripper"] = (grab_kf or {}).get("gripper", "grab") or "grab"
    st.write(f"Plan: move to a pre-grab point {PRE_GRAB_MM:.0f} mm back, go straight in to the corrected grab "
             f"({plan['norm']:.1f} mm from the taught one), then **{plan['gripper']}**. Height and orientation "
             "stay as taught.")
    st.caption("Keep the e-stop in reach.")
    st.session_state["cgrab_plan"] = plan
    st.button("Yes, move", type="primary", key="cgrab_yes")


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
            if st.button(f"🎯 Align {CAM_ARM} to circle", type="primary", use_container_width=True,
                         disabled=FULL_RUNNING or (UNSCREWER is not None and UNSCREWER.running)):
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
            marker_h = cfg["marker_h"]
            if method.startswith("Depth"):
                radius = st.number_input("Bottle radius (mm)", 5.0, 150.0, float(cfg["radius"]), 1.0, key="m_radius",
                                         help="Distance from the marker surface to the bottle's axis.")
                d_lo, d_hi = st.slider("Valid depth (mm)", 100, 3000, (cfg["depth_lo"], cfg["depth_hi"]), step=10,
                                       key="m_depth")
                marker_h = st.number_input("Marker height (mm)", 10.0, 300.0, float(cfg["marker_h"]), 1.0,
                                           key="m_markerh",
                                           help="Fallback when depth fails: distance from how tall the marker looks. "
                                                "Measure it; a 3-inch sticky note is about 76 mm.")
            else:
                st.session_state.setdefault("m_plane", cfg["plane_z"])
                plane_z = st.number_input("Grasp point z (mm, world)", step=1.0, format="%.1f", key="m_plane")
                st.button("Use current gripper height", on_click=use_gripper_height)
            live_cfg = {"hue_lo": hue[0], "hue_hi": hue[1], "sat_min": sat_min, "val_min": val_min,
                        "min_area": min_area, "use_shape": use_shape, "min_rect": min_rect, "max_aspect": max_aspect,
                        "method": method, "radius": radius, "depth_lo": d_lo, "depth_hi": d_hi, "plane_z": plane_z,
                        "marker_h": marker_h, "min_valid": cfg["min_valid"], "max_spread": cfg["max_spread"],
                        "use_dot": cfg["use_dot"], "dot_v_max": cfg["dot_v_max"], "dot_min_area": cfg["dot_min_area"]}
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
            ref = load_grab_ref()
            ref_ready = "bottle_xy" in ref and "grab_pose" in ref
            with st.expander("Grab correction", expanded=not ref_ready):
                st.caption(f"Teach once with the bottle placed correctly: at **{SCAN_KEYFRAME}** press Set reference, "
                           f"then Go to **{GRAB_KEYFRAME}** and press Record grab pose.")
                if st.button(f"Set reference (at {SCAN_KEYFRAME})", use_container_width=True):
                    ok_pose, diff, _ = at_keyframe(SCAN_KEYFRAME)
                    if not ok_pose:
                        st.error(f"Go to '{SCAN_KEYFRAME}' first" + (f" ({diff:.1f}° away)." if diff is not None
                                                                    else f": no keyframe named '{SCAN_KEYFRAME}'."))
                    else:
                        with st.spinner("Measuring over 5 frames…"):
                            m = measure_bottle_now(live_cfg)
                        if m is None:
                            st.error("Pink marker not found consistently. Reference not saved.")
                        else:
                            ref.update(bottle_xy=[float(m["point"][0]), float(m["point"][1])],
                                       bottle_methods=m["methods"], scan_keyframe=SCAN_KEYFRAME,
                                       reference_time=time.strftime("%Y-%m-%d %H:%M:%S"))
                            save_grab_ref(ref)
                            st.toast(f"Reference saved: bottle at ({m['point'][0]:.1f}, {m['point'][1]:.1f}) mm")
                if st.button(f"Set parking reference (at {PARK_KEYFRAME})", use_container_width=True):
                    ok_pose, diff, _ = at_keyframe(PARK_KEYFRAME)
                    if not ok_pose:
                        st.error(f"Go to '{PARK_KEYFRAME}' first" + (f" ({diff:.1f}° away)." if diff is not None
                                                                    else f": no keyframe named '{PARK_KEYFRAME}'."))
                    else:
                        with st.spinner("Measuring over 5 frames…"):
                            m = measure_bottle_now(live_cfg)
                        if m is None:
                            st.error("Pink marker not found consistently. Parking reference not saved.")
                        else:
                            ref.update(park_bottle_xy=[float(m["point"][0]), float(m["point"][1])],
                                       park_keyframe=PARK_KEYFRAME)
                            save_grab_ref(ref)
                            st.toast(f"Parking reference saved: ({m['point'][0]:.1f}, {m['point'][1]:.1f}) mm")
                if st.button(f"Record grab pose (at {GRAB_KEYFRAME})", use_container_width=True):
                    ok_pose, diff, _ = at_keyframe(GRAB_KEYFRAME)
                    if not ok_pose:
                        st.error(f"Go to '{GRAB_KEYFRAME}' first" + (f" ({diff:.1f}° away)." if diff is not None
                                                                    else f": no keyframe named '{GRAB_KEYFRAME}'."))
                    else:
                        ref.update(grab_pose=pose_dict(run(ARMS["arm"].get_end_position())),
                                   grab_keyframe=GRAB_KEYFRAME)
                        save_grab_ref(ref)
                        st.toast("Grab pose recorded")
                st.caption(("Reference: bottle at ({:.1f}, {:.1f}) mm, set {}.".format(*ref["bottle_xy"], ref.get("reference_time", ""))
                            if "bottle_xy" in ref else "Reference: not set.")
                           + (" Grab pose: recorded." if "grab_pose" in ref else " Grab pose: not recorded.")
                           + (" Parking reference: set." if "park_bottle_xy" in ref else " Parking reference: not set."))
            if st.button(f"📡 Scan at {PARK_KEYFRAME}", use_container_width=True, disabled="park_bottle_xy" not in ref,
                         help=f"Read-only first look from {PARK_KEYFRAME}: is the bottle there, and how far from "
                              f"where it should be (max {MAX_PARK_FIX_MM:.0f} mm)?"):
                ok_pose, diff, _ = at_keyframe(PARK_KEYFRAME)
                if not ok_pose:
                    st.error(f"Go to '{PARK_KEYFRAME}' first" + (f" ({diff:.1f}° away)." if diff is not None else "."))
                else:
                    with st.spinner("Measuring over 5 frames…"):
                        m = measure_bottle_now(live_cfg)
                    if m is None:
                        st.session_state["park_scan"] = {"offset": None, "time": time.time()}
                        st.error("No bottle found from parking. Don't proceed.")
                    else:
                        dx = m["point"][0] - ref["park_bottle_xy"][0]
                        dy = m["point"][1] - ref["park_bottle_xy"][1]
                        norm = float(np.hypot(dx, dy))
                        st.session_state["park_scan"] = {"offset": (float(dx), float(dy)), "norm": norm,
                                                         "time": time.time()}
                        if norm > MAX_PARK_FIX_MM:
                            st.error(f"Bottle is about {norm:.0f} mm from where it should be (dx {dx:+.0f}, "
                                     f"dy {dy:+.0f}). Re-place it before going anywhere.")
                        else:
                            st.success(f"Bottle found, about {norm:.0f} mm from where it should be (dx {dx:+.0f}, "
                                       f"dy {dy:+.0f}). OK to Go to {SCAN_KEYFRAME}.")
            if st.button("🎯 Corrected grab", type="primary", use_container_width=True, disabled=not ref_ready or FULL_RUNNING,
                         help=f"At {SCAN_KEYFRAME}: measure the bottle, shift the {GRAB_KEYFRAME} grab by its offset "
                              f"(max {MAX_GRAB_FIX_MM:.0f} mm), confirm, then move."):
                confirm_corrected_grab(live_cfg)

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


# ---------- unscrew (cap_unscrew.py) ----------
@st.dialog("Unscrew cap", width="large")
def confirm_unscrew(params):
    problem = None if (params.start_joints or params.start_target) else UNSCREWER.check()  # start pose: checked on arrival
    if problem:
        st.error(problem)
        return
    st.write(f"Run this on **arm2** / **gripper2**?")
    for line in describe(params):
        st.markdown(f"- {line}")
    st.caption("The other arm must already be holding the bottle. Keep the e-stop in reach.")
    if st.button("Yes, unscrew", type="primary"):
        UNSCREWER.start(params)
        st.rerun()


if UNSCREWER is not None:
    st.divider()
    st.subheader(f"Unscrew cap — arm2")
    busy = UNSCREWER.running or FULL_RUNNING
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        twists = c1.number_input("Twists", 1, 20, 4)
        twist_deg = c2.number_input("Degrees per twist", 10.0, 170.0, 120.0, 5.0)
        final_lift = c3.number_input("Final lift (mm)", 0.0, 100.0, 30.0, 5.0,
                                     help="After the last twist, lift this far while holding the cap; it ends there.")
        grip_force = c4.number_input("Grip force (%)", 1, 100, 60, 5,
                                     help="G2 gripper grasp force for every grab; reset to 50% afterwards.")
        c1, c2, c3, c4 = st.columns(4)
        speed = c1.number_input("Speed (°/s)", 3.0, 60.0, 60.0, 1.0)
        release = c2.number_input("Release opening (0–850)", 100, 850, 500, 25,
                                  help="How far the gripper opens between twists; ~10 units per mm.")
        direction = c3.selectbox("Direction (seen from above)", ["counter-clockwise (loosen)", "clockwise (tighten)"])
        starts = [q["name"] for q in load_poses() if q.get("arm", "arm") == "arm2"]
        start_pose = st.selectbox("Start pose", starts + ["(current pose)"],
                                  index=starts.index(UNSCREW_START_POSE) if UNSCREW_START_POSE in starts else len(starts),
                                  help="arm2 keyframe to move to first (gripper open), fingers around the cap.")
        start_kf = next((q for q in load_poses() if q["name"] == start_pose), {})
        start_joints = tuple(start_kf.get("values", ()))
        start_target = tuple(float(start_kf["pose"][k]) for k in POSE_KEYS) if "pose" in start_kf else ()
        dry_run = c4.toggle("Dry run (no cap)", value=True, help="Full cycle including open/close, but an empty grab "
                            "is not an error. Use this first, without a cap, to check the wrist turns the right way.")
        params = UnscrewParams(twists=twists, twist_deg=twist_deg, final_lift_mm=final_lift,
                               speed_deg_s=speed, release_pos=release, dry_run=dry_run,
                               start_pose=start_pose if start_kf else "", start_joints=start_joints,
                               start_target=start_target,
                               grip_force_pct=int(grip_force),
                               loosen_sign=-1 if direction.startswith("counter") else 1)
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("🔓 Unscrew", type="primary", use_container_width=True, disabled=busy):
            confirm_unscrew(params)
        if b2.button("⏹ Stop", use_container_width=True, disabled=not UNSCREWER.running):
            UNSCREWER.stop()
            st.rerun()

        @st.fragment(run_every=1.0)
        def unscrew_status():
            s = UNSCREWER.status()
            if s["state"] == "running":
                st.info(f"Running: {s['step']}")
            elif s["state"] == "done":
                st.success("Unscrew complete." + (f" ({s['error']})" if s.get("error") else ""))
            elif s["state"] in ("error", "stopped"):
                st.error(f"Unscrew {s['state']} at '{s['step']}': {s['error']}")

        unscrew_status()


# ---------- full run (bottle_task.py): arm's sequence, arm2's keyframes -> unscrew -> trash drop ----------
@st.dialog("Full run", width="large")
def confirm_full_run(keyframes, params, drop, retreat, via, alongside):
    st.write("Run all three phases, one after another?")
    st.markdown("**1. Keyframes** (all of arm's, then arm2's)")
    for n, kf in enumerate(keyframes):
        st.markdown(f"- {kf['arm']} → **{kf['name']}**" + (f", then gripper {kf['gripper']}" if kf.get("gripper") else "")
                    + (f", wait {kf['delay']:g} s" if kf.get("delay") else ""))
        if n == 0 and alongside is not None:
            st.markdown(f"  - **starting at the same time:** {alongside['arm']} → **{alongside['name']}**"
                        + (f", then gripper {alongside['gripper']}" if alongside.get("gripper") else "")
                        + f" — {kf['arm']} waits at {kf['name']} until {alongside['arm']} has arrived, then continues")
    st.markdown(f"**2. Unscrew** (arm2, settings from the Unscrew panel)")
    for line in describe(params):
        st.markdown(f"- {line}")
    st.markdown("**3. Retreat and drop**")
    if retreat is not None:
        st.markdown(f"- {retreat['arm']} straight up to **{retreat['name']}**'s height, holding the cap "
                    "(orientation kept as the unscrew left it; gripper unchanged)")
    for kf in via:
        st.markdown(f"- {kf['arm']} → **{kf['name']}** holding the cap (move only; gripper unchanged)")
    st.markdown(f"- {drop['arm']} → **{drop['name']}**" + (f", then gripper {drop['gripper']}" if drop.get("gripper") else ""))
    if params.dry_run:
        st.warning("Unscrew is in dry-run mode (empty grabs allowed). Turn it off in the Unscrew panel for a real cap.")
    st.caption("arm's sequence must end holding the bottle (its last keyframe has no gripper action). Keep the "
               "e-stop in reach; STOP ALL or Stop halts every phase.")
    if st.button("Yes, run all", type="primary"):
        BOTTLE_TASK.start(keyframes, params, drop, retreat, via, alongside)
        st.rerun()


if BOTTLE_TASK is not None:
    st.divider()
    st.subheader("Full run — arm's sequence → arm2: keyframes → unscrew → retreat → trash drop")
    all_kf = load_poses()
    drop_kf = next((k for k in all_kf if k["name"] == TRASH_DROP_POSE), None)
    pre_kf = [k for k in poses_io.sequence(all_kf, "arm_then_arm2") if k["name"] != TRASH_DROP_POSE]
    # arm2 goes to its start pose while arm does its first keyframe, then skips it in its own part
    alongside_kf = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == ALONGSIDE_POSE), None)
    if alongside_kf is not None and pre_kf and pre_kf[0]["arm"] != "arm2":
        pre_kf = [k for k in pre_kf if k is not alongside_kf]
    else:
        alongside_kf = None
    retreat_kf = next((k for k in all_kf if k["name"] == RETREAT_POSE), None)
    via_kf = [k for name in VIA_POSES for k in all_kf if k["name"] == name]
    with st.container(border=True):
        if drop_kf is None:
            st.warning(f"Save a keyframe named `{TRASH_DROP_POSE}` (where the cap is dropped) to enable the full run.")
        else:
            st.caption(f"1. Keyframes: {' → '.join(k['name'] for k in pre_kf) or '(none)'}"
                       + (f" (arm2 → {alongside_kf['name']} starts together with {pre_kf[0]['name']})" if alongside_kf else "") + " · "
                       f"2. Unscrew with the settings above (start **{params.start_pose or 'current pose'}**, "
                       f"{'dry run' if params.dry_run else 'real cap'}) · "
                       + (f"3. straight up to **{RETREAT_POSE}**'s height holding the cap, then " if retreat_kf else "3. ")
                       + "".join(f"via **{k['name']}**, then " for k in via_kf)
                       + f"{drop_kf['arm']} to **{TRASH_DROP_POSE}**, then gripper {drop_kf.get('gripper') or 'unchanged'}.")
        f1, f2, _ = st.columns([1, 1, 2])
        if f1.button("▶ Full run", type="primary", use_container_width=True,
                     disabled=drop_kf is None or busy or FULL_RUNNING):
            confirm_full_run(pre_kf, params, drop_kf, retreat_kf, via_kf, alongside_kf)
        if f2.button("⏹ Stop full run", use_container_width=True, disabled=not FULL_RUNNING):
            BOTTLE_TASK.stop()
            st.rerun()

        @st.fragment(run_every=1.0)
        def full_run_status():
            s = BOTTLE_TASK.status()
            if s["state"] == "running":
                st.info(f"Running: {s['step']}")
            elif s["state"] == "done":
                st.success("Full run complete.")
            elif s["state"] in ("error", "stopped"):
                st.error(f"Full run {s['state']} at '{s['step']}': {s['error']}")
            if s["state"] != "running" and FULL_RUNNING:
                st.rerun()  # finished: refresh so the buttons unlock

        full_run_status()


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
    for col, text in zip(header, ["#", "Name", "Arm", "Joints (deg) or pose", "Gripper", "Delay after (s)", ""]):
        col.markdown(f"**{text}**")
    for i, p in enumerate(poses):
        row = st.columns(COLS)
        row[0].write(i + 1)
        row[1].write(p["name"])
        row[2].write(p.get("arm", "arm"))
        row[3].code(keyframe_text(p), language=None)
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

    selection = st.radio("Run which keyframes", list(poses_io.RUN_OPTIONS), format_func=poses_io.RUN_OPTIONS.get,
                         horizontal=True, key="run_selection",
                         help="arm, then arm2: all of arm's keyframes first, then all of arm2's, whatever the "
                              "table order.")
    chosen = poses_io.sequence(poses, selection)
    st.caption(f"{len(chosen)} keyframe(s) selected · total programmed delay "
               f"{sum(q.get('delay', 0.0) for q in chosen[:-1]):g} s (the last keyframe's delay is ignored)")
    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True, disabled=FULL_RUNNING):
        confirm_run(selection)

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            target = p["pose"] if "pose" in p else [round(v, 3) for v in p["values"]]
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {target}, {p.get('delay', 0.0):g}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")
