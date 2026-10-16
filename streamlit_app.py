"""Streamlit keyframe recorder for Viam arms (arm + arm2), plus cam2 cap detection and alignment.

Config comes from environment variables (see env.sh / run.sh); nothing secret is stored in code.
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

from cap_tracker import CapTracker
import bottle_task
from bottle_task import POSE_KEYS, BottleTask
from cap_unscrew import UnscrewParams, Unscrewer, describe
from cap_vision import CapVision, colorize_depth, detect_circles, robust_median

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
API_KEY = os.environ["VIAM_API_KEY"]
ARM_GRIPPERS = {"arm": "gripper", "arm2": "gripper2"}
POSES_FILE = Path(os.environ.get("POSES_FILE", "poses.json"))
MAX_JOG_DEG = 15.0
JOINTS = [f"Joint {i}" for i in range(6)]  # 0-based, matching the Viam app
CAM, CAM_ARM, CAM_GRIPPER = "cam2", "arm2", "gripper2"  # wrist camera, the arm it rides on, and its gripper
MAX_ALIGN_MM = 100.0  # longest single alignment move; run Align again to close the rest
UNSCREW_START_POSE = "open_bottle_cap_pose"  # keyframe the unscrew sequence moves to first
TRASH_DROP_POSE = "trash_drop_pose"  # keyframe the full run ends at (cap dropped there), not part of its phase 1
RETREAT_POSE = "open_bottle_cap_up_pose"  # full run: after unscrewing, rise to this pose's height before the drop
VIA_POSES = ["safe_bottle_start_pose"]  # full run: then pass through these (move only, cap held) on the way to the drop

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
        try:
            camera = Camera.from_robot(robot, CAM)
            intrinsics = (await camera.get_properties()).intrinsic_parameters
        except Exception:
            camera, intrinsics = None, None
        return robot, arms, grippers, camera, intrinsics

    robot, arms, grippers, camera, intrinsics = run(connect())
    return run, robot, arms, grippers, camera, intrinsics


run, ROBOT, ARMS, GRIPPERS, CAMERA, INTRINSICS = get_robot()


def load_poses():
    return json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else []


def save_poses(poses):
    POSES_FILE.write_text(json.dumps(poses, indent=2))


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


# ---------- cam2 vision (cap_vision.py) and tracking (cap_tracker.py) ----------
VISION = (CapVision(run, ROBOT, ARMS[CAM_ARM], CAMERA, INTRINSICS, CAM, CAM_ARM, CAM_GRIPPER)
          if CAMERA is not None and CAM_ARM in ARMS else None)


@st.cache_resource
def get_tracker():
    """One tracker thread shared by every browser session, so the toggle always reflects the arm's real state."""
    return CapTracker(VISION, ARMS[CAM_ARM]) if VISION is not None else None


TRACKER = get_tracker()


@st.cache_resource
def get_unscrewer():
    if CAM_ARM not in ARMS or CAM_ARM not in GRIPPERS:
        return None
    return Unscrewer(run, ROBOT, ARMS[CAM_ARM], GRIPPERS[CAM_ARM], CAM_ARM, CAM_GRIPPER)


UNSCREWER = get_unscrewer()


@st.cache_resource
def get_bottle_task():
    return BottleTask(run, ARMS, GRIPPERS, UNSCREWER) if UNSCREWER is not None else None


BOTTLE_TASK = get_bottle_task()
FULL_RUNNING = BOTTLE_TASK is not None and BOTTLE_TASK.running


# ---------- sidebar ----------
with st.sidebar:
    st.title("🦾 Arm Keyframes")
    arm = st.selectbox("Active arm", list(ARMS), help="Jog, gripper and Save act on this arm.")
    if st.button("🛑 STOP ALL", type="primary", use_container_width=True):
        if TRACKER is not None:
            TRACKER.stop()
        if BOTTLE_TASK is not None:
            BOTTLE_TASK.stop()
        if UNSCREWER is not None:
            UNSCREWER.stop()
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
            go_to_keyframe(p)
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
        found = VISION.detect_stable(settings["detect"])
        if found is None:
            st.error("No cap found consistently across 5 frames. Adjust the detection settings and try again.")
            return
        bgr, depth, circle, hits = found
        point = VISION.measure(circle, depth, settings["cap_d"])
        plan = VISION.plan_align(point, settings["hover"], settings["keep_height"], MAX_ALIGN_MM)
    except Exception as e:
        st.error(str(e))
        return
    st.image(VISION.annotate(bgr, [circle], grip_px=VISION.gripper_pixel(point)), use_container_width=True)
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


if VISION is not None:
    st.divider()
    st.subheader(f"Camera — {CAM} on {CAM_ARM}")
    feed_col, ctrl_col = st.columns([3, 1])
    with ctrl_col:
        live = st.toggle("Live feed", value=True)
        show_depth = st.toggle("Show depth", value=False)
        with st.expander("Detection", expanded=False):
            min_r, max_r = st.slider("Radius (px)", 10, 400, (40, 250))
            sensitivity = st.slider("Strictness", 20, 120, 50, help="Hough accumulator threshold; higher finds fewer circles.")
            min_silver = st.slider("Min silver", 0.0, 1.0, 0.5, 0.05,
                                   help="Share of the circle that is unsaturated and at least half as bright as the table. "
                                        "Cap ≈ 0.7–0.8, bottle ≈ 0.15–0.35.")
            min_rim = st.slider("Min outline", 0.0, 1.0, 0.55, 0.05,
                                help="Share of the circle's outline on an image edge. Cap ≈ 0.7–0.8; partial circles ≈ 0.4.")
            show_rejected = st.toggle("Show rejected circles", value=False)
            cap_d = st.number_input("Cap diameter (mm)", 5.0, 150.0, 30.0, 0.5,
                                    help="Used to estimate distance when depth has no reading on the circle.")
        keep_height = st.radio("Align", ["XY only (keep height)", "XY + hover above"], index=0) == "XY only (keep height)"
        hover = st.number_input("Hover above circle (mm)", 20.0, 300.0, 80.0, 5.0, disabled=keep_height)
        settings = {"detect": (min_r, max_r, sensitivity, min_silver, min_rim), "cap_d": cap_d, "hover": hover,
                    "keep_height": keep_height}
        unscrewing = (UNSCREWER is not None and UNSCREWER.running) or FULL_RUNNING
        if st.button(f"🎯 Align {CAM_ARM} to circle", type="primary", use_container_width=True,
                     disabled=TRACKER.running or unscrewing, help="Not available while tracking or unscrewing."):
            confirm_align(settings)

        st.markdown("**Tracking**")
        tracking = st.toggle(f"Track cap with {CAM_ARM}", value=TRACKER.running, disabled=unscrewing,
                             help=f"Slowly follows the cap in XY at the current height (≤{TRACKER.max_step_mm:g} mm "
                                  f"per {TRACKER.period_s:g} s). Stops when the cap leaves the view and resumes "
                                  f"when it is back; never goes more than {TRACKER.max_travel_mm:g} mm from where "
                                  "it started.")
        if tracking and not TRACKER.running:
            TRACKER.start(settings["detect"], cap_d)
        elif not tracking and TRACKER.running:
            TRACKER.stop()
        elif TRACKER.running:
            TRACKER.update(settings["detect"], cap_d)  # detection sliders apply live

        @st.fragment(run_every=1.0)
        def tracker_status():
            s = TRACKER.status()
            if s["state"] == "error":
                st.error(f"Tracking stopped: {s.get('error')}")
            elif s["state"] != "off":
                err = s.get("error_mm")
                st.caption(f"**{s['state']}** · error {'–' if err is None else f'{err:.1f} mm'} · "
                           f"{s.get('moves', 0)} moves")
            if s["state"] in ("off", "error") and TRACKER.running is False and tracking:
                st.rerun()  # tracker stopped itself: refresh so the toggle shows off

        tracker_status()

    @st.fragment(run_every=1.0 if live else None)
    def cam_feed(settings):
        try:
            bgr, depth = VISION.capture()
            circles, rejected = detect_circles(bgr, *settings["detect"])
            # smooth over the last 5 frames so a one-frame misdetection does not make the overlay jump
            hist = (st.session_state.get("cap_hist", []) + [circles[0] if circles else None])[-5:]
            st.session_state["cap_hist"] = hist
            smoothed = robust_median([c for c in hist if c], min_hits=2)
            circles = [smoothed[0]] + circles[1:] if smoothed else circles
            p = VISION.measure(circles[0], depth, settings["cap_d"]) if circles else None
            grip_px = VISION.gripper_pixel(p) if p else None
        except Exception as e:
            st.warning(f"Could not read {CAM}: {e}")
            return
        img_col, depth_col = st.columns(2) if show_depth and depth is not None else (st.container(), None)
        img_col.image(VISION.annotate(bgr, circles, rejected if show_rejected else (), grip_px), use_container_width=True)
        if depth_col is not None:
            depth_col.image(colorize_depth(depth), use_container_width=True, caption="Depth (black = no reading)")
        if circles:
            u, v, r, _ = circles[0]
            st.caption(f"Circle at ({u:.0f}, {v:.0f}) px, r {r:.0f} px · camera frame x {p['x']:.1f}, "
                       f"y {p['y']:.1f}, z {p['z']:.1f} mm ({'depth' if p['method'] == 'depth' else 'from cap size'}"
                       f", depth valid {p['valid']:.0%}) · {len(rejected)} other circles rejected")
        else:
            st.caption(f"No cap found ({len(rejected)} circles rejected).")

    with feed_col:
        cam_feed(settings)


# ---------- unscrew (cap_unscrew.py) ----------
@st.dialog("Unscrew cap", width="large")
def confirm_unscrew(params):
    problem = None if (params.start_joints or params.start_target) else UNSCREWER.check()  # start pose: checked on arrival
    if problem:
        st.error(problem)
        return
    st.write(f"Run this on **{CAM_ARM}** / **{CAM_GRIPPER}**?")
    for line in describe(params):
        st.markdown(f"- {line}")
    st.caption("The other arm must already be holding the bottle. Keep the e-stop in reach.")
    if st.button("Yes, unscrew", type="primary"):
        UNSCREWER.start(params)
        st.rerun()


if UNSCREWER is not None:
    st.divider()
    st.subheader(f"Unscrew cap — {CAM_ARM}")
    busy = UNSCREWER.running or FULL_RUNNING or (TRACKER is not None and TRACKER.running)
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
        starts = [q["name"] for q in load_poses() if q.get("arm", "arm") == CAM_ARM]
        start_pose = st.selectbox("Start pose", starts + ["(current pose)"],
                                  index=starts.index(UNSCREW_START_POSE) if UNSCREW_START_POSE in starts else len(starts),
                                  help=f"{CAM_ARM} keyframe to move to first (gripper open), fingers around the cap.")
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


# ---------- full run (bottle_task.py): keyframes -> unscrew -> trash drop ----------
@st.dialog("Full run", width="large")
def confirm_full_run(keyframes, params, drop, retreat, via):
    st.write("Run all three phases, one after another?")
    st.markdown("**1. Keyframes** (table order)")
    for kf in keyframes:
        st.markdown(f"- {kf['arm']} → **{kf['name']}**" + (f", then gripper {kf['gripper']}" if kf.get("gripper") else "")
                    + (f", wait {kf['delay']:g} s" if kf.get("delay") else ""))
    st.markdown(f"**2. Unscrew** ({CAM_ARM}, settings from the Unscrew panel)")
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
    st.caption("The other arm must be holding the bottle by phase 2. Keep the e-stop in reach; STOP ALL or Stop "
               "halts every phase.")
    if st.button("Yes, run all", type="primary"):
        BOTTLE_TASK.start(keyframes, params, drop, retreat, via)
        st.rerun()


if BOTTLE_TASK is not None:
    st.divider()
    st.subheader("Full run — keyframes → unscrew → retreat → trash drop")
    all_kf = load_poses()
    drop_kf = next((k for k in all_kf if k["name"] == TRASH_DROP_POSE), None)
    pre_kf = [k for k in all_kf if k["name"] != TRASH_DROP_POSE]
    retreat_kf = next((k for k in all_kf if k["name"] == RETREAT_POSE), None)
    via_kf = [k for name in VIA_POSES for k in all_kf if k["name"] == name]
    with st.container(border=True):
        if drop_kf is None:
            st.warning(f"Save a keyframe named `{TRASH_DROP_POSE}` (where the cap is dropped) to enable the full run.")
        else:
            st.caption(f"1. Keyframes: {' → '.join(k['name'] for k in pre_kf) or '(none)'} · "
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

    st.caption(f"Total programmed delay: {sum(q.get('delay', 0.0) for q in poses[:-1]):g} s "
               "(the last keyframe's delay is ignored)")
    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True, disabled=FULL_RUNNING):
        confirm_run()

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            target = p["pose"] if "pose" in p else [round(v, 3) for v in p["values"]]
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {target}, {p.get('delay', 0.0):g}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")
