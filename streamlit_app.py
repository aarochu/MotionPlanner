"""Streamlit keyframe recorder for Viam arms (arm + arm2), plus cam2 cap detection and alignment.

Config comes from environment variables (see env.sh / run.sh); nothing secret is stored in code.
"""
import asyncio
import dataclasses
import functools
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import poses_io
import vision as V
import yolo_grab as YG
import bottle_track as BT
from cap_vision import CapVision
from bottle_task import POSE_KEYS, BottleTask, Mover
from cap_unscrew import UnscrewParams, Unscrewer, describe
from cap_vision import circle_features, detect_circles  # cap test judged against the table's brightness
from kinematics import ov_to_matrix
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
AT_POSE_TOL_DEG = 1.0    # how close (every joint, or the tool orientation) the arm must be to a keyframe to count as "at" it
AT_POSE_TOL_MM = 2.0     # and for an XYZ keyframe, how close the end position must be
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
ARM2_FIRST_POSE = "safe_bottle_start_pose"  # full run: arm2 moves here first (out of arm's way), then arm's sequence

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
def get_mover():
    return Mover(run, ROBOT, ARMS)


MOVER = get_mover()  # keyframe, corrected-grab and Align moves: motion service, collision-checked


@st.cache_resource
def get_unscrewer():
    if "arm2" not in ARMS or "arm2" not in GRIPPERS:
        return None
    return Unscrewer(run, ROBOT, ARMS["arm2"], GRIPPERS["arm2"], "arm2", "gripper2")


UNSCREWER = get_unscrewer()


@st.cache_resource
def get_bottle_task():
    return BottleTask(run, ARMS, GRIPPERS, UNSCREWER, MOVER) if UNSCREWER is not None else None


BOTTLE_TASK = get_bottle_task()
FULL_RUNNING = BOTTLE_TASK is not None and BOTTLE_TASK.running


def load_poses():
    """Flat list of keyframes (each with its "arm"), in play order. File format: see poses_io.py."""
    return poses_io.load(POSES_FILE)


def save_poses(poses):
    poses_io.save(POSES_FILE, poses)


def rpc(fn, *args, **kwargs):
    """run(fn(*args)) with one retry when the robot session lapsed. The server rejects a call that carries an expired
    session before running it, and the SDK starts a fresh session on the next call, so the retry is safe."""
    try:
        return run(fn(*args, **kwargs))
    except Exception as e:
        if "SESSION_EXPIRED" not in str(e):
            raise
        return run(fn(*args, **kwargs))


def joints(arm):
    return list(rpc(ARMS[arm].get_joint_positions).values)


def move(arm, values):
    rpc(ARMS[arm].move_to_joint_positions, JointPositions(values=values))


@st.cache_resource
def action_stop_event():
    return threading.Event()


def check_action_stop():
    if action_stop_event().is_set():
        raise InterruptedError("Sequence stopped")


def go_to_keyframe(p):
    check_action_stop()
    MOVER.go(p)  # motion planner, collision-checked; a joint keyframe is converted to its end pose first


def keyframe_text(p):
    if "pose" in p:
        q = p["pose"]
        return (f"x {q['x']:.1f}  y {q['y']:.1f}  z {q['z']:.1f}  "
                f"o ({q['o_x']:.3f}, {q['o_y']:.3f}, {q['o_z']:.3f})  θ {q['theta']:.1f}")
    return "  ".join(f"{v:7.1f}" for v in p["values"])


def apply_gripper(arm, action):
    if action == "grab":
        return rpc(GRIPPERS[arm].grab)
    elif action == "open":
        rpc(GRIPPERS[arm].open)


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
        action_stop_event().set()
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

# Two tabs: "Simple" (two buttons for arm 1, filled at the end of this file) and "Full controls" (everything else).
# Entering the Full tab here instead of a `with` block keeps the rest of the file at its original indentation.
TAB_SIMPLE, TAB_FULL = st.tabs(["Simple", "Full controls"])
TAB_FULL.__enter__()


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
    st.write(f"Run **{p['name']}** on **{p['arm']}**?")
    st.caption(keyframe_text(p))
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, move", type="primary"):
        action_stop_event().clear()
        def go():
            go_to_keyframe(p)
            check_action_stop()
            apply_gripper(p["arm"], p.get("gripper"))
        with st.spinner("Moving…"):
            ok = safely(go)
        if ok:
            st.rerun()


@st.dialog("Run sequence")
def confirm_run(selection, first=None, last=None):
    """Play the steps of `selection`, optionally only steps first..last (0-based, inclusive)."""
    steps = poses_io.sequence(load_poses(), selection)
    label = poses_io.RUN_OPTIONS[selection]
    if not steps:
        st.warning(f"No keyframes saved for {label}.")
        return
    first = 0 if first is None else first
    last = len(steps) - 1 if last is None else last
    poses = steps[first:last + 1]
    counts = {a: sum(p["arm"] == a for p in poses) for a in poses_io.ARMS}
    if len(poses) < len(steps):
        st.write(f"Play steps **{first + 1}–{last + 1}** of {len(steps)} for **{label}** "
                 f"({poses[0]['name']} → {poses[-1]['name']}), one move at a time?")
    else:
        st.write(f"Play **{len(poses)}** step(s) for **{label}**, one move at a time?")
    if selection == "arm_then_arm2":
        st.caption(f"First all {counts['arm']} arm keyframe(s), then all {counts['arm2']} arm2 keyframe(s). "
                   "The two arms never move at the same time.")
    st.caption("Keep the e-stop in reach.")
    if st.button("Yes, run", type="primary"):
        action_stop_event().clear()
        progress = st.progress(0.0)
        for n, p in enumerate(poses, 1):
            if n > 1 and p["arm"] != poses[n - 2]["arm"]:
                st.write(f"— {poses[n - 2]['arm']} done, starting {p['arm']} —")
            st.write(f"{n}/{len(poses)}: {p['arm']} → {p['name']}")
            def step_fn(p=p):
                go_to_keyframe(p)
                check_action_stop()
                apply_gripper(p["arm"], p.get("gripper"))
            if not safely(step_fn):
                return
            wait = p.get("delay", 0.0)
            if wait > 0 and n < len(poses):
                st.write(f"⏱ waiting {wait:g} s")
                if action_stop_event().wait(wait):
                    st.warning("Sequence stopped")
                    return
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
            ok = safely(lambda: MOVER.to_pose(CAM_ARM, st.session_state.pop("align_plan")["target"]))
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


def at_keyframe(name, arm_name="arm"):
    """(is the arm at this keyframe?, how far off as text, the keyframe or None)."""
    kf = next((p for p in load_poses() if p["name"] == name and p.get("arm", "arm") == arm_name), None)
    if kf is None:
        return False, None, None
    if "values" in kf:
        diff = max(abs(a - b) for a, b in zip(joints(arm_name), kf["values"]))
        return diff <= AT_POSE_TOL_DEG, f"{diff:.1f}°", kf
    e, q = run(ARMS[arm_name].get_end_position()), kf["pose"]
    mm = float(np.linalg.norm([e.x - q["x"], e.y - q["y"], e.z - q["z"]]))
    r = ov_to_matrix(e.o_x, e.o_y, e.o_z, e.theta).T @ ov_to_matrix(q["o_x"], q["o_y"], q["o_z"], q["theta"])
    deg = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    return mm <= AT_POSE_TOL_MM and deg <= AT_POSE_TOL_DEG, f"{mm:.1f} mm, {deg:.1f}°", kf


def pose_dict(p):
    return {k: float(getattr(p, k)) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


# ---------- arm: YOLO head-on grab of the leftmost bottle (yolo_grab.py) ----------
def end_pose(arm_name="arm"):
    return pose_dict(rpc(ARMS[arm_name].get_end_position))


def pose_dist(a, b):
    return float(np.linalg.norm([a[k] - b[k] for k in ("x", "y", "z")]))


def same_orientation(a, b):
    o = ("o_x", "o_y", "o_z")
    return (np.allclose([a[k] for k in o], [b[k] for k in o], atol=0.01)
            and abs((a["theta"] - b["theta"] + 180) % 360 - 180) < 1.0)


def move_straight(goal, step_mm):
    """Short straight steps from where arm is now to goal, same orientation. STOP ALL interrupts between steps."""
    for w in YG.steps(end_pose(), goal, step_mm):
        check_action_stop()
        rpc(ARMS["arm"].move_to_position, Pose(**w))


NUDGE_MAX_MM = 20.0


def nudge_dirs(e):
    """Unit directions for nudging, from the arm's current orientation. Left/right as seen looking along the gripper
    (the camera's view); in/out along the gripper's approach axis; up/down vertical."""
    a = YG._unit([e["o_x"], e["o_y"], e["o_z"]])
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, YG._unit(a - a.dot(up) * up))
    return {"left": left, "right": -left, "up": up, "down": -up, "in": a, "out": -a}


def nudge(direction, mm):
    """One straight move of at most NUDGE_MAX_MM, orientation kept. Moves immediately, like the jog buttons."""
    e = end_pose()
    d = nudge_dirs(e)[direction] * min(float(mm), NUDGE_MAX_MM)
    rpc(ARMS["arm"].move_to_position, Pose(**dict(e, x=e["x"] + d[0], y=e["y"] + d[1], z=e["z"] + d[2])))


PIVOT_MM = 105.0 + YG.DEFAULTS["pad_mm"]   # finger pads: gripper frame (105 mm from the flange) + pad offset


def turn_to(target):
    rpc(ARMS["arm"].move_to_position, Pose(**target))


@st.dialog("Level the gripper")
def confirm_level():
    e = end_pose()
    try:
        target, turn = YG.level_target(e, PIVOT_MM)
    except ValueError as err:
        st.error(str(err))
        return
    p0, r0, h0 = YG.attitude(e)
    st.write(f"Make the approach axis horizontal and the fingers close horizontally, pivoting on the finger pads "
             f"({PIVOT_MM:.0f} mm out from the flange) so the gripper stays where it is.")
    st.table({"": ["now", "after"], "pitch (°)": [f"{p0:+.1f}", "0.0"], "roll (°)": [f"{r0:+.1f}", "0.0"],
              "heading (°)": [f"{h0:+.1f}", f"{h0:+.1f}"]})
    st.caption(f"{turn:.1f}° turn. Arm end moves {pose_dist(e, target):.1f} mm. Collision stop is off: keep the e-stop in reach.")
    if st.session_state.get("lvl_yes"):
        with st.spinner("Levelling…"):
            ok = safely(turn_to, target)
        if ok:
            st.rerun()
        return
    st.button("Yes, level it", type="primary", key="lvl_yes")


@st.dialog("YOLO head-on grab", width="large")
def confirm_yolo_move(kind):
    """kind: "pre" (open, go to the pre-grasp), "grab" (straight in, grab) or "back" (straight out to the pre-grasp line)."""
    plan, m = st.session_state.get("yg_plan"), st.session_state.get("yg_meas")
    cfg = st.session_state.get("yg_cfg_used", YG.DEFAULTS)
    if plan is None:
        st.error("Locate the bottle first.")
        return
    now, a = end_pose(), plan["a"]
    if kind == "pre":
        if pose_dist(now, m["end"]) > 2 or not same_orientation(now, m["end"]):
            st.error("The arm moved since the bottle was located. Press Locate leftmost bottle again first.")
            return
        goal, step = plan["pre"], cfg["pre_step_mm"]
        title = f"Open the gripper, then move to the pre-grasp {cfg['pre_mm']:.0f} mm straight in front of the bottle"
    elif kind == "grab":
        off = pose_dist(now, plan["pre"])
        if off > 5 or not same_orientation(now, plan["pre"]):
            st.error(f"Go to the pre-grasp first (the arm is {off:.0f} mm from it).")
            return
        goal, step = plan["grab"], cfg["in_step_mm"]
        title = f"Open the gripper, go {cfg['pre_mm']:.0f} mm straight in along the gripper's axis, then grab"
    else:
        k = float(np.dot([now[q] - plan["pre"][q] for q in ("x", "y", "z")], a))
        if k < 2:
            st.info("Already at or behind the pre-grasp; nothing to back out of.")
            return
        goal, step = dict(now, x=now["x"] - k * a[0], y=now["y"] - k * a[1], z=now["z"] - k * a[2]), cfg["in_step_mm"]
        title = f"Back straight out {k:.0f} mm along the approach (gripper unchanged)"
    wps = YG.steps(now, goal, step)
    st.write(f"**{title}.**")
    st.table({"": ["arm end now", "target", "move"],
              "x (mm)": [f"{now['x']:.1f}", f"{goal['x']:.1f}", f"{goal['x'] - now['x']:+.1f}"],
              "y (mm)": [f"{now['y']:.1f}", f"{goal['y']:.1f}", f"{goal['y'] - now['y']:+.1f}"],
              "z (mm)": [f"{now['z']:.1f}", f"{goal['z']:.1f}", f"{goal['z'] - now['z']:+.1f}"]})
    st.caption(f"{len(wps)} straight step(s) of at most {step:.0f} mm; orientation kept. STOP ALL stops between "
               "steps. Collision stop is off on this arm: keep the e-stop in reach.")
    if st.session_state.get(f"yg_yes_{kind}"):
        action_stop_event().clear()

        def go():
            if kind in ("pre", "grab"):
                apply_gripper("arm", "open")
            move_straight(goal, step)
            if kind == "grab":
                held = apply_gripper("arm", "grab")
                st.session_state["yg_status"] = ("Grabbed: the gripper reports it is holding something." if held else
                                                 "The gripper closed but reports it is NOT holding anything.")
        with st.spinner("Moving…"):
            ok = safely(go)
        if ok:
            st.rerun()
        return
    st.button("Yes, move", type="primary", key=f"yg_yes_{kind}")


def yolo_result():
    m, p = st.session_state.get("yg_meas"), st.session_state.get("yg_plan")
    if m is None or p is None:
        st.info("Press **Locate leftmost bottle**: YOLO + depth find the leftmost bottle in this view and plan a "
                "head-on grab. Nothing moves until you confirm a move.")
        return
    st.image(YG.annotate(m, p), use_container_width=True,
             caption=f"Located at {m['cap']['time']} · red cross: grasp point on the bottle axis · "
                     "magenta: open fingertips")
    A = p["A"]
    c = st.columns(4)
    c[0].metric("Bottle axis x, y (mm)", f"{A[0]:.0f}, {A[1]:.0f}")
    c[1].metric("Grasp z (mm)", f"{p['zg']:.0f}")
    c[2].metric("Width at grasp / height (mm)", f"{p.get('grip_width', m['width_mm']):.0f} / {p['height_mm']:.0f}")
    c[3].metric("Camera to bottle (mm)", f"{m['z']:.0f}")
    if m.get("profile"):
        st.caption("Bottle width by height (fraction of height from the bottom → mm): " + " · ".join(
            f"{f:.2f} → {w:.0f}" for f, w in sorted(m["profile"].items())) + f" · YOLO box {m['width_mm']:.0f} mm")
    rel = m["cap"]["origin"] - m["gripper"]["origin"]
    st.caption(f"Camera mount (frame system), relative to the gripper frame: {rel.dot(p['lat']):+.0f} mm sideways "
               f"(left +), {rel[2]:+.0f} mm up, {rel.dot(p['a']):+.0f} mm along the approach; optical axis "
               f"{np.degrees(np.arccos(np.clip(m['cap']['R'][:, 2].dot(p['a']), -1, 1))):.1f}° from the gripper axis.")
    lat, vert, along = p["offset_now"]
    st.caption(f"{m['frames']}/{m['of']} frames agree within {m['spread_mm']:.1f} mm · depth valid {m['valid']:.0%}, "
               f"spread {m['depth_spread']:.0f} mm · YOLO {m['conf']:.0%} · bottles per frame {m['counts']}")
    st.caption(f"Finger pads now vs. grasp point: {lat:+.0f} mm sideways (left +), {vert:+.0f} mm up, "
               f"{along:.0f} mm ahead.")
    for level, text in p["checks"]:
        (st.error if level == "error" else st.warning if level == "warn" else st.success)(text)
    if st.session_state.get("yg_status"):
        st.info(st.session_state["yg_status"])


if CAMERA is not None and CAM_ARM in ARMS:
    st.divider()
    st.subheader(f"Camera — {CAM} on {CAM_ARM}")
    marker_mode = CAM_ARM == "arm"   # arm: YOLO head-on grab of the leftmost bottle   arm2: bottle cap (+ Align)
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
            d = YG.DEFAULTS
            with st.expander("YOLO grab settings", expanded=False):
                ycfg = dict(d,
                            conf=st.slider("Min YOLO confidence", .1, .95, d["conf"], .05, key="yg_conf"),
                            radius=st.number_input("Bottle radius (mm)", 10.0, 80.0, d["radius"], 1.0, key="yg_radius",
                                                   help="Front surface to axis. Saratoga 28 oz is about 38-40 mm."),
                            grasp_frac=st.slider("Grasp height (fraction of bottle height)", .1, .7, d["grasp_frac"],
                                                 .05, key="yg_frac", help="Measured up from the bottle's bottom."),
                            pad_mm=st.number_input("Finger pads beyond gripper frame (mm)", 20.0, 100.0, d["pad_mm"],
                                                   1.0, key="yg_pad",
                                                   help="The bottle axis ends up this far beyond the Viam gripper "
                                                        "frame (105 mm from the flange). Larger stops earlier."),
                            pre_mm=st.number_input("Pre-grasp distance (mm)", 60.0, 200.0, d["pre_mm"], 5.0, key="yg_pre"),
                            stroke_mm=st.number_input("Gripper opening (mm)", 50.0, 90.0, d["stroke_mm"], 1.0,
                                                      key="yg_stroke"),
                            lat_trim_mm=st.number_input("Lateral trim, + = right (mm)", -50.0, 50.0, d["lat_trim_mm"],
                                                        1.0, key="yg_lat",
                                                        help="Correction for the camera's centre estimate, as seen "
                                                             "looking along the gripper."),
                            depth_trim_mm=st.number_input("Depth trim, + = deeper (mm)", -40.0, 40.0,
                                                          d["depth_trim_mm"], 1.0, key="yg_depth",
                                                          help="0 puts the finger pads on the bottle's axis."),
                            level=st.toggle("Level the gripper on approach", value=d["level"], key="yg_level"))
            settings = {"yolo": ycfg}
            busy = FULL_RUNNING or (UNSCREWER is not None and UNSCREWER.running)
            if st.button("🔍 Locate leftmost bottle", type="primary", use_container_width=True,
                         help="Read-only: YOLO + aligned depth over a few frames, then plans a head-on grab. Nothing moves."):
                try:
                    with st.spinner(f"Measuring over {int(ycfg['frames'])} frames…"):
                        m = YG.measure(run, ROBOT, ARMS["arm"], CAMERA, CAM, CAM_GRIPPER, ycfg)
                    st.session_state.update(yg_meas=m, yg_plan=YG.plan(m, ycfg), yg_cfg_used=ycfg, yg_status="")
                except Exception as e:
                    st.session_state.pop("yg_plan", None)
                    st.error(f"Locate failed: {e}")
            plan = st.session_state.get("yg_plan")
            ready = plan is not None and not any(level == "error" for level, _ in plan["checks"])
            if st.button("1 · Move to pre-grasp", use_container_width=True, disabled=not ready or busy,
                         help="Open the gripper, then go straight to a point in front of the bottle (confirm first)."):
                confirm_yolo_move("pre")
            if st.button("2 · Go in and grab", use_container_width=True, disabled=not ready or busy,
                         help="From the pre-grasp: straight in along the gripper's axis, then grab (confirm first)."):
                confirm_yolo_move("grab")
            if st.button("↩ Back out", use_container_width=True, disabled=plan is None or busy,
                         help="Straight back out along the approach to the pre-grasp line (confirm first)."):
                confirm_yolo_move("back")
            st.markdown("**Nudge** (straight, orientation kept; moves immediately)")
            nmm = st.select_slider("Nudge step (mm)", options=[1, 2, 5, 10, 20], value=5, key="yg_nudge_mm")
            labels = [("← Left", "left"), ("↑ Up", "up"), ("Right →", "right"),
                      ("Out", "out"), ("↓ Down", "down"), ("In", "in")]
            for col, (label, key) in zip(st.columns(3) + st.columns(3), labels):
                if col.button(label, key=f"yg_n_{key}", use_container_width=True, disabled=busy,
                              help=f"Move the gripper {key} by the step. Left/right as seen looking along the "
                                   "gripper (the camera's view); in/out along the gripper's axis."):
                    safely(nudge, key, nmm)
            try:
                pitch, roll, heading = YG.attitude(end_pose())
                st.caption(f"Gripper attitude: pitch {pitch:+.1f}°, roll {roll:+.1f}°, heading {heading:+.1f}° "
                           "(0° = straight along world +x).")
            except Exception:
                pass
            if st.button("⚖ Level gripper", use_container_width=True, disabled=busy,
                         help="Approach axis horizontal, fingers closing horizontally; pivots on the finger pads (confirm first)."):
                confirm_level()
            ycols = st.columns(2)
            for col, (label, deg) in zip(ycols, [("⟲ Yaw left 1°", 1.0), ("Yaw right 1° ⟳", -1.0)]):
                if col.button(label, key=f"yg_yaw_{deg}", use_container_width=True, disabled=busy,
                              help="Turn about the vertical through the finger pads; moves immediately."):
                    safely(lambda d=deg: turn_to(YG.yaw_target(end_pose(), d, PIVOT_MM)[0]))

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
    def cam_feed_plain(settings):
        try:
            bgr, depth = capture()
        except Exception as e:
            st.warning(f"Could not read {CAM}: {e}")
            return
        plan, m = st.session_state.get("yg_plan"), st.session_state.get("yg_meas")
        note = ""
        if plan is not None and m is not None:
            try:   # orientation is kept by every YOLO move, so the camera has only translated since Locate
                now = end_pose()
                if same_orientation(now, m["end"]):
                    cap = dict(m["cap"], origin=m["cap"]["origin"] + np.array([now[k] - m["end"][k] for k in ("x", "y", "z")]))
                    for pt, col in ((plan["A"], (0, 0, 255)), (plan["A"] - plan["lat"] * 42, (255, 0, 255)),
                                    (plan["A"] + plan["lat"] * 42, (255, 0, 255))):
                        u, v = YG.project(cap, pt)
                        cv2.drawMarker(bgr, (int(u), int(v)), col, cv2.MARKER_CROSS, 40, 3)
                    note = "Red cross: the planned grasp point projected from where the arm is now (it should sit on the bottle)."
            except Exception:
                pass
        img_col, depth_col = st.columns(2) if show_depth and depth is not None else (st.container(), None)
        img_col.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True, caption=note or None)
        if depth_col is not None:
            depth_col.image(colorize_depth(depth), use_container_width=True, caption="Depth (black = no reading)")

    with feed_col:
        if marker_mode:
            cam_feed_plain(settings)
            yolo_result()
        else:
            cam_feed_cap(settings)


# ---------- standalone YOLO viewer (read-only, independent of sequence/calibration) ----------
st.divider()
st.subheader("YOLO bottle preview — Arm 1 camera")
st.caption("Watch bottle detection from cam at any arm position. Green marks the leftmost visible bottle; "
           "yellow marks other bottles. This viewer never moves the arm or runs a sequence.")
yolo_live = st.toggle("Live YOLO preview", value=True, key="yolo_live")
yolo_confidence = st.slider("Preview confidence threshold", .1, 1., .65, .05, key="yolo_confidence",
                            help="Only changes this preview. Sequence detection stays at 0.65.")
yolo_show_all = st.toggle("Show other detected objects", value=False, key="yolo_show_all",
                          help="Also show other classes in blue to see whether YOLO labels a bottle as something else.")


@st.fragment(run_every=2.0 if yolo_live else None)
def yolo_preview_panel(confidence, show_all):
    st.button("Refresh YOLO preview", key="yolo_refresh", help="Capture another frame without moving the arm.")
    if "arm" not in CAM_RIGS:
        st.warning("Arm 1 camera is unavailable.")
        return
    try:
        image, count = YG.preview(run, ROBOT, CAM_RIGS["arm"][0], confidence=confidence, show_all=show_all)
    except Exception as exc:
        st.warning(f"YOLO preview unavailable: {exc}")
        return
    st.image(image, width="stretch")
    st.caption(f"{count} bottle(s) at confidence ≥ {confidence:.0%} · "
               f"captured {time.strftime('%H:%M:%S')} · {'Live: refreshes every 2 seconds' if yolo_live else 'Paused: click Refresh for a new frame'}")
    if count == 0:
        st.info("No bottles detected at this threshold. Try lowering it or showing other detected objects.")


yolo_preview_panel(yolo_confidence, yolo_show_all)


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
def confirm_full_run(keyframes, params, drop, retreat, via):
    st.write("Run all three phases, one after another?")
    st.markdown("**1. Keyframes**, one arm at a time, each move planned around obstacles and the other arm")
    for kf in keyframes:
        st.markdown(f"- {kf['arm']} → **{kf['name']}**" + (f", then gripper {kf['gripper']}" if kf.get("gripper") else "")
                    + (f", wait {kf['delay']:g} s" if kf.get("delay") else ""))
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
        if safely(BOTTLE_TASK.start, keyframes, params, drop, retreat, via):
            st.rerun()


if BOTTLE_TASK is not None:
    st.divider()
    st.subheader("Full run — arm's sequence → arm2: keyframes → unscrew → retreat → trash drop")
    all_kf = load_poses()
    drop_kf = next((k for k in all_kf if k["name"] == TRASH_DROP_POSE), None)
    pre_kf = [k for k in poses_io.sequence(all_kf, "arm_then_arm2") if k["name"] != TRASH_DROP_POSE]
    # arm2 goes to its start pose first (out of arm's way), then arm's sequence, then the rest of arm2's
    first_kf = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == ARM2_FIRST_POSE), None)
    if first_kf is not None:
        pre_kf = [first_kf] + [k for k in pre_kf if k is not first_kf]
    retreat_kf = next((k for k in all_kf if k["name"] == RETREAT_POSE), None)
    via_kf = [k for name in VIA_POSES for k in all_kf if k["name"] == name]
    with st.container(border=True):
        if drop_kf is None:
            st.warning(f"Save a keyframe named `{TRASH_DROP_POSE}` (where the cap is dropped) to enable the full run.")
        else:
            st.caption(f"1. Keyframes: {' → '.join(k['name'] for k in pre_kf) or '(none)'}"
                       + " · "
                       f"2. Unscrew with the settings above (start **{params.start_pose or 'current pose'}**, "
                       f"{'dry run' if params.dry_run else 'real cap'}) · "
                       + (f"3. straight up to **{RETREAT_POSE}**'s height holding the cap, then " if retreat_kf else "3. ")
                       + "".join(f"via **{k['name']}**, then " for k in via_kf)
                       + f"{drop_kf['arm']} to **{TRASH_DROP_POSE}**, then gripper {drop_kf.get('gripper') or 'unchanged'}.")
        f1, f2, _ = st.columns([1, 1, 2])
        if f1.button("▶ Full run", type="primary", use_container_width=True,
                     disabled=drop_kf is None or busy or FULL_RUNNING):
            confirm_full_run(pre_kf, params, drop_kf, retreat_kf, via_kf)
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
st.subheader("Sequence steps (play order, top → bottom)")
poses = load_poses()
if not poses:
    st.info("No keyframes yet. Jog the arm to a pose and save it above.")
else:
    COLS = [0.4, 2, 1, 4, 1, 1.4, 3]
    header = st.columns(COLS)
    for col, text in zip(header, ["#", "Name", "Arm", "Action / joints / pose", "Gripper", "Delay after (s)", ""]):
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

    selection = st.radio("Run which steps", list(poses_io.RUN_OPTIONS), format_func=poses_io.RUN_OPTIONS.get,
                         horizontal=True, key="run_selection",
                         help="arm, then arm2: all of arm's keyframes first, then all of arm2's, whatever the "
                              "table order.")
    all_steps = poses_io.sequence(poses, selection)
    first, last = 0, len(all_steps) - 1
    if len(all_steps) > 1:
        labels = [f"{n}. {p['arm']} · {p['name']}" for n, p in enumerate(all_steps, 1)]
        # the key changes with the step list, so a stale range never points at steps that moved or were deleted
        start_label, end_label = st.select_slider(
            "Run from … to …", options=labels, value=(labels[0], labels[-1]),
            key=f"run_range_{selection}_{hash(tuple(labels))}",
            help="Drag either end to run only part of the sequence. Both ends are included.")
        first, last = labels.index(start_label), labels.index(end_label)
    chosen = all_steps[first:last + 1]
    st.caption(f"{len(chosen)} of {len(all_steps)} step(s) selected · total programmed delay "
               f"{sum(q.get('delay', 0.0) for q in chosen[:-1]):g} s (the last keyframe's delay is ignored)")
    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True, disabled=FULL_RUNNING):
        confirm_run(selection, first, last)

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            target = p["pose"] if "pose" in p else [round(v, 3) for v in p["values"]]
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {target}, {p.get('delay', 0.0):g}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")


TAB_FULL.__exit__(None, None, None)


# ---------- Simple tab: reset arm 1, and the YOLO pick & place (yolo_grab.reset_arm / pick_and_place) ----------
def plan_arm1_joints(q):
    """yolo_grab's moves to parking / NewMid / above the place spot: the end pose those joints give, reached through
    the motion planner (collision-checked), like every keyframe move."""
    MOVER.go({"arm": "arm", "name": "yolo", "values": list(q)})


def smooth_arm1(goal):
    """yolo_grab's straight moves (reset back-out and rise, approach, grab, lift, set-down): one continuous planned
    straight-line move instead of short stop-and-go steps."""
    MOVER.to_pose("arm", goal, linear=(5.0, 5.0))


def arm1_keyframe(name):
    kf = next((p for p in load_poses() if p["arm"] == "arm" and p["name"] == name and "values" in p), None)
    return kf["values"] if kf else None


def run_logged(title, fn, *args):
    """Run fn(*args, check=..., log=...) inside the open dialog, streaming its steps; keep the log for the tab."""
    lines, out, ok = [], None, False
    action_stop_event().clear()
    with st.status(title, expanded=True) as status:
        def log(text):
            lines.append(f"{time.strftime('%H:%M:%S')}  {text}")
            status.write(text)
        try:
            out, ok = fn(*args, check=check_action_stop, log=log), True
            status.update(label=f"{title}: done", state="complete")
        except Exception as e:
            log(f"Stopped: {e}")
            status.update(label=f"{title}: stopped", state="error")
    st.session_state["simple_log"] = {"title": title, "ok": ok, "lines": lines}
    return out


@st.dialog("Reset arm 1")
def confirm_reset_arm1():
    parking = arm1_keyframe("parking")
    if parking is None:
        st.error("No joint keyframe named 'parking' for arm.")
        return
    st.write(f"Open arm 1's gripper. Unless it is already at parking: back straight out {YG.RESET_RETREAT_MM:.0f} mm "
             f"along the gripper, rise to z {YG.RESET_CLEAR_Z:.0f} mm if lower, then move to **parking**.")
    st.caption("Lets go of anything arm 1 holds. Arm 2 is not moved. Keep the e-stop in reach.")
    if st.session_state.get("reset1_yes"):
        run_logged("Reset arm 1", functools.partial(YG.reset_arm, move_joints=plan_arm1_joints, smooth=smooth_arm1),
                   rpc, ARMS["arm"], GRIPPERS["arm"], parking)
        st.rerun()
    st.button("Yes, reset arm 1", type="primary", key="reset1_yes")


@st.dialog("Run YOLO pick & place", width="large")
def confirm_pick_place():
    newmid, parking = arm1_keyframe("NewMid"), arm1_keyframe("parking")
    if newmid is None or parking is None:
        st.error("Arm needs joint keyframes named 'NewMid' and 'parking'.")
        return
    cfg = dict(YG.DEFAULTS)
    st.markdown(
        "1. If arm 1 is not at parking or NewMid, reset it first: open the gripper, back straight out "
        f"{YG.RESET_RETREAT_MM:.0f} mm, rise to z {YG.RESET_CLEAR_Z:.0f} mm, go to **parking**. Then open the gripper "
        "and go to **NewMid**.\n"
        f"2. YOLO + depth locate the leftmost bottle ({int(cfg['frames'])} frames). If it is not fully in view, back up "
        f"{cfg['backup_mm']:.0f} mm in x and look again (up to {int(cfg['max_backups'])} times); any failed check "
        "stops here.\n"
        f"3. Level pre-grasp {cfg['pre_mm']:.0f} mm in front, straight in, grab (opens and backs out on a miss).\n"
        f"4. Lift {YG.LIFT_MM:.0f} mm, carry to above the place spot, set it down at a height worked out from the "
        "measured bottle, and keep holding it.")
    st.caption(f"Known bottle {cfg['bottle_diam_mm']:.0f} mm in the {cfg['stroke_mm']:.0f} mm gripper. "
               f"Grasp trims: {cfg['lat_trim_mm']:+.0f} mm right, {cfg['depth_trim_mm']:+.0f} mm deeper than the "
               f"bottle axis. Table at z {YG.TABLE_Z:+.1f} mm. Only arm 1 moves: make sure arm 2 is clear of the place "
               "spot. STOP ALL stops between steps; keep the e-stop in reach.")
    if st.session_state.get("pickplace_yes"):
        result = run_logged("YOLO pick & place", functools.partial(YG.pick_and_place, move_joints=plan_arm1_joints, reset_first=True, smooth=smooth_arm1),
                            run, rpc, ROBOT, ARMS["arm"], GRIPPERS["arm"],
                            CAM_RIGS["arm"][0], RIGS["arm"][0], RIGS["arm"][1], newmid, parking, cfg)
        if result:
            st.session_state.update(yg_meas=result[0], yg_plan=result[1], yg_cfg_used=cfg)
        st.rerun()
    st.button("Yes, run it", type="primary", key="pickplace_yes")


def simple_view():
    st.subheader("Arm 1")
    st.caption("**Reset** parks arm 1 with the gripper open. **Run** picks the leftmost bottle with YOLO and sets it "
               "down at the place spot, still holding it. Both ask for confirmation first.")
    busy = FULL_RUNNING or (UNSCREWER is not None and UNSCREWER.running) or JOB_RUNNER.running
    c1, c2 = st.columns(2)
    if c1.button("🏠 Reset arm 1", use_container_width=True, disabled=busy or "arm" not in ARMS, key="simple_reset"):
        confirm_reset_arm1()
    if c2.button("▶ Run YOLO pick & place", type="primary", use_container_width=True,
                 disabled=busy or "arm" not in ARMS or "arm" not in CAM_RIGS, key="simple_run"):
        confirm_pick_place()
    if busy:
        st.info("The full run or the unscrew is running; wait for it to finish.")
    try:
        q = joints("arm")
        where = next((name for name in ("parking", "NewMid") if YG._near(q, arm1_keyframe(name))), None)
        st.caption(f"Arm 1 is {'at **' + where + '**' if where else 'away from parking and NewMid'}.")
    except Exception as e:
        st.caption(f"Could not read arm 1: {e}")
    log = st.session_state.get("simple_log")
    if log:
        (st.success if log["ok"] else st.error)(f"Last: {log['title']} — {'done' if log['ok'] else 'stopped'}")
        st.code("\n".join(log["lines"]) or "(no steps)", language=None)
    m, p = st.session_state.get("yg_meas"), st.session_state.get("yg_plan")
    if m is not None and p is not None:
        st.image(YG.annotate(m, p), use_container_width=True,
                 caption=f"Last YOLO locate ({m['cap']['time']}): red cross = grasp point, magenta = open fingertips")


# ---------- Simple tab: the whole job in one shot, retryable per step, and the cam2 feed (cap_sequence.py) ----------
from cap_sequence import DOWN_MM as CAP_DOWN_MM, FAST, NORMAL, Job, JobRunner
CAP_LOOK_POSE = "open_bottle_cap_up_pose"  # arm2 keyframe over the place spot: gripper down, cam2 sees the bottle top


@st.cache_resource
def get_cam2_vision():
    if "arm2" not in ARMS or "arm2" not in CAM_RIGS:
        return None
    return CapVision(run, ROBOT, ARMS["arm2"], *CAM_RIGS["arm2"], cam="cam2", arm_name="arm2", gripper="gripper2")


CAM2_VISION = get_cam2_vision()


@st.cache_resource
def get_job_runner():
    return JobRunner(), (BT.BottleTracker(run, CAM2_VISION, MOVER) if CAM2_VISION is not None else None)


JOB_RUNNER, TRACKER = get_job_runner()


def halt_all_arms():
    for a in ARMS.values():
        try:
            run(a.stop())
        except Exception:
            pass


def build_job(with_arm1, unscrew_params):
    kfs = {k["name"]: k for k in load_poses() if k.get("arm") == "arm2" and "pose" in k}
    if CAP_LOOK_POSE not in kfs or TRASH_DROP_POSE not in kfs:
        raise ValueError(f"Needs arm2 XYZ keyframes '{CAP_LOOK_POSE}' and '{TRASH_DROP_POSE}'.")
    pick = None
    if with_arm1:
        newmid, parking = arm1_keyframe("NewMid"), arm1_keyframe("parking")
        if newmid is None or parking is None:
            raise ValueError("Arm needs joint keyframes named 'NewMid' and 'parking'.")
        cfg = dict(YG.DEFAULTS)

        def pick(check, log):
            YG.pick_and_place(run, rpc, ROBOT, ARMS["arm"], GRIPPERS["arm"], CAM_RIGS["arm"][0], RIGS["arm"][0],
                              RIGS["arm"][1], newmid, parking, cfg, check=check, log=log, move_joints=plan_arm1_joints,
                              reset_first=True, smooth=smooth_arm1)
    return Job(run, MOVER, TRACKER, UNSCREWER, ARMS["arm2"], GRIPPERS["arm2"], pick, kfs[CAP_LOOK_POSE],
               dict(kfs[TRASH_DROP_POSE]["pose"]), unscrew_params, halt_all_arms, arm1=ARMS.get("arm"))


@st.dialog("Run the job", width="large")
def confirm_job(with_arm1, unscrew_params):
    try:
        job = build_job(with_arm1, unscrew_params)
    except ValueError as e:
        st.error(str(e))
        return
    names = [n for n, _ in job.steps]
    st.markdown("\n".join(f"{i}. {n}" + (" *(skipped: arm 2 only)*" if i == 1 and not with_arm1 else "")
                          for i, n in enumerate(names, 1)))
    st.caption(f"Speed: arm 2 at {FAST['set_speed']:g}°/s, {FAST['set_acceleration']:g}°/s² (80 % of max) for its moves; "
               f"arm 1's pick & place at the normal {NORMAL['set_speed']:g}°/s. Arm 2 goes back to normal at the end.")
    st.caption(f"Lock: within 3 mm of the bottle top (z {BT.BOTTLE_TOP_Z:.0f}); down {CAP_DOWN_MM:.0f} mm from the look "
               f"height; drop at {TRASH_DROP_POSE}. Unscrew settings (Full controls → Unscrew cap):")
    for line in describe(unscrew_params):
        st.markdown(f"- {line}")
    if unscrew_params.dry_run:
        st.warning("Unscrew is in dry-run mode (empty grabs allowed): set it in Full controls → Unscrew cap.")
    if not with_arm1:
        st.caption("Arm 1 must already be holding the bottle at the place spot.")
    st.caption("Every move goes through the motion planner. Stop or STOP ALL halts it; a stopped or failed step can "
               "be retried. Keep the e-stop in reach.")
    if st.button("Yes, run it", type="primary", key="job_yes"):
        JOB_RUNNER.start(job, 0 if with_arm1 else 1)
        st.rerun()


@st.dialog("Run from a step", width="large")
def confirm_job_from(k, unscrew_params):
    try:
        job = build_job(k == 0, unscrew_params)
    except ValueError as e:
        st.error(str(e))
        return
    names = [n for n, _ in job.steps]
    st.markdown("\n".join(f"{i + 1}. {n}" for i, n in enumerate(names) if i >= k))
    st.caption("Starts from where the arms are now. Make sure the earlier steps are really done (for step 7: arm 2 is "
               "holding the cap up at the look height). Keep the e-stop in reach.")
    if st.button("Yes, run", type="primary", key="job_from_yes"):
        JOB_RUNNER.start(job, k)
        st.rerun()


def cam2_feed():
    """Live cam2 image: the bottle-top circles found and, with the gripper pointing down, the spot under the fingers."""
    try:
        bgr, _ = CAM2_VISION.capture()
        e = run(ARMS["arm2"].get_end_position())
        out = bgr.copy()
        caption = "cam2"
        if e.o_z < -0.9:  # looking down: show what the tracker would see
            r_exp = BT.expected_radius(CAM2_VISION, BT.BOTTLE_TOP_Z, BT.TOP_D_MM)
            tcp, _ = BT.tcp_of(e)
            under = BT.plane_to_pixel(CAM2_VISION, (tcp[0], tcp[1], BT.BOTTLE_TOP_Z))
            circles, _ = BT.TopCircleDetect(r_exp)(bgr)
            for i, (u, v, r, _) in enumerate(circles[:3]):
                cv2.circle(out, (int(u), int(v)), int(r), (0, 255, 0) if i == 0 else (0, 165, 255), 3 if i == 0 else 2)
            if under is not None:
                cv2.drawMarker(out, (int(under[0]), int(under[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
            caption = (f"cam2 looking down: green = bottle-top circle ({len(circles)} found), magenta = under the "
                       "fingers")
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True, caption=caption)
    except Exception as ex:
        st.caption(f"cam2 not available: {ex}")


def job_view():
    st.subheader("Full job — one shot")
    st.caption("Arm 1 picks the bottle with YOLO and holds it; arm 2 locks on the cap, unscrews it and drops it in "
               "the trash. If a step stops or fails, retry it and the job carries on from there.")
    running = JOB_RUNNER.running
    busy = FULL_RUNNING or UNSCREWER.running or running
    s = JOB_RUNNER.status()
    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Full job (arm 1 + arm 2)", type="primary", use_container_width=True, disabled=busy, key="job_full"):
        confirm_job(True, dataclasses.replace(params, start_pose="", start_joints=(), start_target=()))
    if c2.button("▶ Open the cap only (arm 2)", use_container_width=True, disabled=busy, key="job_cap"):
        confirm_job(False, dataclasses.replace(params, start_pose="", start_joints=(), start_target=()))
    if c3.button("⏹ Stop", use_container_width=True, disabled=not running, key="job_stop"):
        JOB_RUNNER.stop()
        st.rerun()
    with st.expander("Run from a chosen step (e.g. to finish a job after a dashboard restart)", expanded=False):
        step_names = ["Arm 1: YOLO pick & place", "Arm 2: look at the bottle top", "Arm 2: lock on the cap",
                      f"Arm 2: down {CAP_DOWN_MM:.0f} mm onto the cap", "Arm 2: unscrew", "Arm 2: back up holding the cap",
                      "Arm 2: drop the cap in the trash"]
        pick = st.selectbox("Start at", [f"{i}. {n}" for i, n in enumerate(step_names, 1)], index=6, key="job_from")
        k = int(pick.split(".")[0]) - 1
        if st.button(f"▶ Run from step {k + 1}", disabled=busy, key="job_from_go"):
            confirm_job_from(k, dataclasses.replace(params, start_pose="", start_joints=(), start_target=()))
    if not running and s["state"] in ("error", "stopped") and s.get("failed") is not None and JOB_RUNNER.job is not None:
        k = s["failed"]
        if st.button(f"↻ Retry step {k + 1}: {s['names'][k]} (then continue)", type="primary", disabled=busy,
                     key="job_retry"):
            JOB_RUNNER.start(JOB_RUNNER.job, k)
            st.rerun()

    @st.fragment(run_every=1.0)
    def job_status():
        s = JOB_RUNNER.status()
        if s["names"]:
            marks = []
            for i, n in enumerate(s["names"]):
                if s["state"] == "running" and i == s["step"]:
                    mark = "▶"
                elif s.get("failed") == i:
                    mark = "✖"
                elif (s["state"] == "done") or (s["step"] is not None and i < s["step"]) or \
                        (s.get("failed") is not None and i < s["failed"]):
                    mark = "✔"
                else:
                    mark = "·"
                marks.append(f"{mark} {i + 1}. {n}")
            st.code("\n".join(marks), language=None)
        if s["state"] == "running":
            st.info(s.get("detail") or (s["log"][-1] if s["log"] else "running"))
        elif s["state"] == "done":
            st.success("Done: the cap is in the trash.")
        elif s["state"] in ("error", "stopped"):
            st.error(f"{s['state'].capitalize()} at step {s['failed'] + 1}: {s['error']}")
        if s["log"]:
            with st.expander("Log", expanded=False):
                st.code("\n".join(s["log"]), language=None)

    job_status()
    st.markdown("**cam2** (arm 2's camera)")

    @st.fragment(run_every=1.5)
    def cam2_live():
        cam2_feed()

    cam2_live()


with TAB_SIMPLE:
    simple_view()
    if TRACKER is not None and UNSCREWER is not None:
        st.divider()
        job_view()
