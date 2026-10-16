"""Streamlit keyframe recorder for Viam arms (arm + arm2).

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
from viam.components.camera import Camera
from viam.components.arm import Arm, JointPositions
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
ARM_GRIPPERS = {"arm": "gripper", "arm2": "gripper2"}
CAMERAS = {"arm": "cam", "arm2": "cam2"}
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
            api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
        )
        robot = await RobotClient.at_address(ADDRESS, opts)
        arms, grippers, cams = {}, {}, {}
        for arm_name, grip_name in ARM_GRIPPERS.items():
            try:
                arms[arm_name] = Arm.from_robot(robot, arm_name)
                grippers[arm_name] = Gripper.from_robot(robot, grip_name)
            except Exception:  # arm2 may not be wired up yet
                continue
            try:
                cams[arm_name] = Camera.from_robot(robot, CAMERAS[arm_name])
            except Exception:
                pass
        return robot, arms, grippers, cams

    robot, arms, grippers, cams = run(connect())
    return run, robot, arms, grippers, cams


run, ROBOT, ARMS, GRIPPERS, CAMS = get_robot()


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
def confirm_run(selection):
    poses = [q for q in load_poses() if selection == "both" or q.get("arm", "arm") == selection]
    label = "both arms" if selection == "both" else selection
    if not poses:
        st.warning(f"No keyframes saved for {label}.")
        return
    st.write(f"Play **{len(poses)}** keyframe(s) for **{label}** in order, one at a time?")
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
        col.number_input(name, min_value=-360.0, max_value=360.0, step=1.0, format="%.2f", key=f"tgt{i}",
                         help=JOINT_DESC[i])
        for i, (col, name) in enumerate(zip(cols, JOINTS))
    ]
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Load current", on_click=load_current_into_targets, use_container_width=True)
    if b2.button("Move", type="primary", use_container_width=True):
        confirm_targets(arm, targets)

# ---------- vision (detection logic lives in vision.py) ----------
VISION_CONFIG_FILE = Path(os.environ.get("VISION_CONFIG_FILE", "vision_config.json"))
VISION_KEYS = ("hue_rng", "sat_min", "val_min", "min_area", "use_shape", "min_rect", "max_aspect", "vision_method",
               "bottle_radius", "depth_rng", "plane_z")


def use_gripper_height():
    st.session_state["plane_z"] = float(V.world_point(run, ROBOT, ARM_GRIPPERS[arm])[2])


st.divider()
st.subheader(f"Vision — {CAMERAS[arm]} on {arm}")
st.caption("Read-only: captures a camera frame, finds the bottle by color, keeps only blobs that look like a "
           "rectangle, and converts the result to world x, y, z. Nothing here moves an arm.")
if arm not in CAMS:
    st.warning(f"No camera available for {arm}.")
else:
    img_col, vis_col = st.columns([3, 2])
    cfg = V.load_config(VISION_CONFIG_FILE)
    with vis_col:
        with st.container(border=True):
            st.markdown("**1. Color (HSV)**")
            hue = st.slider("Hue range", 0, 179, (cfg["hue_lo"], cfg["hue_hi"]), key="hue_rng",
                            help="OpenCV hue 0–179. The pink note measures about 173–175; blue is about 100–130.")
            sat_min = st.slider("Min saturation", 0, 255, cfg["sat_min"], key="sat_min")
            val_min = st.slider("Min brightness", 0, 255, cfg["val_min"], key="val_min")
            min_area = st.slider("Min blob area (px)", 100, 100000, cfg["min_area"], step=100, key="min_area")
        with st.container(border=True):
            st.markdown("**2. Shape**")
            use_shape = st.checkbox("Enable shape check", value=bool(cfg["use_shape"]), key="use_shape",
                                    help="Off: any color blob counts (the largest wins). On: blobs must also look "
                                         "like a rectangle.")
            min_rect = st.slider("Min rectangularity", 0.0, 1.0, float(cfg["min_rect"]), step=0.05, key="min_rect",
                                 disabled=not use_shape,
                                 help="Blob area ÷ area of its tightest rotated rectangle. 1.0 is a perfect "
                                      "rectangle; a circle is about 0.79.")
            max_aspect = st.slider("Max aspect ratio", 1.0, 5.0, float(cfg["max_aspect"]), step=0.1,
                                   key="max_aspect", disabled=not use_shape,
                                   help="Long side ÷ short side. 1.0 is square.")
        method = st.radio("Position method", list(V.METHODS), key="vision_method",
                          index=list(V.METHODS).index(cfg["method"]),
                          help="Depth uses the target's measured distance. Plane assumes the bottle's grasp point is "
                               "at a known height and needs the camera to look downward.")
        use_depth = method.startswith("Depth")
        radius, (d_lo, d_hi), plane_z = cfg["radius"], (cfg["depth_lo"], cfg["depth_hi"]), cfg["plane_z"]
        with st.container(border=True):
            if use_depth:
                st.markdown("**Depth settings**")
                radius = st.number_input("Bottle radius (mm)", min_value=5.0, max_value=150.0, value=float(cfg["radius"]),
                                         step=1.0, key="bottle_radius",
                                         help="Distance from the target surface to the bottle's axis.")
                d_lo, d_hi = st.slider("Valid depth range (mm)", 100, 3000, (cfg["depth_lo"], cfg["depth_hi"]),
                                       step=10, key="depth_rng",
                                       help="Readings outside this range are ignored (background, glare).")
            else:
                st.markdown("**Bottle plane height**")
                st.session_state.setdefault("plane_z", cfg["plane_z"])
                plane_z = st.number_input("z of the bottle's grasp point (mm, world frame)", step=1.0, format="%.1f",
                                          key="plane_z")
                st.button("Use current gripper height", on_click=use_gripper_height,
                          help="Jog the gripper to the height where it should grab the bottle, then click.")

        live_cfg = {"hue_lo": hue[0], "hue_hi": hue[1], "sat_min": sat_min, "val_min": val_min,
                    "min_area": min_area, "use_shape": use_shape, "min_rect": min_rect, "max_aspect": max_aspect,
                    "method": method,
                    "radius": radius, "depth_lo": d_lo, "depth_hi": d_hi, "plane_z": plane_z}
        save_c, reset_c = st.columns(2)
        if save_c.button("💾 Save as default", use_container_width=True,
                         help=f"Writes these settings to {VISION_CONFIG_FILE} so they load every time."):
            V.save_config(VISION_CONFIG_FILE, live_cfg)
            st.toast(f"Saved to {VISION_CONFIG_FILE}")
        if reset_c.button("↩ Reset to saved", use_container_width=True,
                          help="Discard unsaved slider changes and reload the saved defaults."):
            for k in VISION_KEYS:
                st.session_state.pop(k, None)
            st.rerun()
        if st.button("📷 Capture & detect", type="primary", use_container_width=True):
            try:
                with st.spinner("Capturing…"):
                    st.session_state["capture"] = V.capture_frame(run, ROBOT, CAMS[arm], CAMERAS[arm], arm)
            except Exception as e:
                st.error(f"Capture failed: {e}")

    cap = st.session_state.get("capture")
    with img_col:
        if cap is None or cap["arm"] != arm:
            st.info("Press **Capture & detect** to grab a frame from this arm's camera.")
        else:
            mask, hit, candidates = V.detect(cap["bgr"], live_cfg)
            shown = V.draw_detections(cap["bgr"], hit, candidates)
            tab_img, tab_mask, tab_depth = st.tabs(["Detection", "Mask", "Depth"])
            tab_img.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width="stretch")
            tab_img.caption("Green = chosen · cyan = passed the shape check but scored lower · red = rejected by shape.")
            tab_mask.image(mask, width="stretch")
            if cap.get("depth") is not None:
                tab_depth.image(V.depth_preview(cap["depth"]), width="stretch")
                tab_depth.caption("Blue = near, red = far (0–2 m). Black = no reading.")
            st.caption(f"Frame captured at {cap['time']}. Re-capture after moving the arm.")

            if candidates:
                with st.expander(f"{len(candidates)} color blob(s) found", expanded=not hit):
                    st.dataframe([{
                        "area (px)": int(c["area"]), "rectangularity": round(c["rectangularity"], 2),
                        "aspect": round(c["aspect"], 2), "solidity": round(c["solidity"], 2),
                        "result": "chosen" if c is hit else ("ok, lower score" if c["accepted"] else c["reason"]),
                    } for c in sorted(candidates, key=lambda c: -c["area"])], hide_index=True)
            if not candidates:
                st.warning("No color match. Adjust the HSV sliders (check the Mask tab) or the minimum area.")
            elif not hit:
                st.warning("Color blobs were found but none passed the shape check. See the table above, "
                           "or relax the rectangularity / aspect sliders.")
            else:
                stats = None
                if use_depth:
                    world, stats, note = V.depth_to_world(cap, hit, radius, d_lo, d_hi)
                else:
                    world, note = V.pixel_to_world(cap, *hit["px"], plane_z)
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
                    st.caption(f"Target depth: median {stats['median']:.0f} mm, spread {stats['spread']:.0f} mm, "
                               f"{stats['valid_frac']:.0%} of pixels valid ({stats['n']} px).")

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

    RUN_OPTIONS = {"both": "Both arms", "arm": "arm only", "arm2": "arm2 only"}
    selection = st.radio("Run which keyframes", list(RUN_OPTIONS), format_func=RUN_OPTIONS.get,
                         horizontal=True, key="run_selection")
    chosen = [q for q in poses if selection == "both" or q.get("arm", "arm") == selection]
    st.caption(f"{len(chosen)} keyframe(s) selected · total programmed delay "
               f"{sum(q.get('delay', 0.0) for q in chosen[:-1]):g} s (the last keyframe's delay is ignored)")
    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True):
        confirm_run(selection)

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {[round(v, 3) for v in p['values']]}, {p.get('delay', 0.0):g}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")
