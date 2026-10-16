"""Streamlit keyframe recorder for Viam arms (arm + arm2).

Config comes from environment variables (see env.sh / run.sh); nothing secret is stored in code.
"""
import asyncio
import json
import os
import threading
from pathlib import Path

import streamlit as st
from viam.components.arm import Arm, JointPositions
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
ARM_GRIPPERS = {"arm": "gripper", "arm2": "gripper2"}
POSES_FILE = Path(os.environ.get("POSES_FILE", "poses.json"))
MAX_JOG_DEG = 15.0
JOINTS = [f"Joint {i}" for i in range(6)]  # 0-based, matching the Viam app

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
        arms, grippers = {}, {}
        for arm_name, grip_name in ARM_GRIPPERS.items():
            try:
                arms[arm_name] = Arm.from_robot(robot, arm_name)
                grippers[arm_name] = Gripper.from_robot(robot, grip_name)
            except Exception:  # arm2 may not be wired up yet
                pass
        return robot, arms, grippers

    _, arms, grippers = run(connect())
    return run, arms, grippers


run, ARMS, GRIPPERS = get_robot()


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
        if st.button(f"💾 Save current pose of {arm}", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Give the keyframe a name.")
            else:
                def do_save():
                    pose = {"name": name.strip(), "arm": arm, "values": joints(arm),
                            "gripper": "" if g_action == "none" else g_action}
                    save_poses([p for p in load_poses() if p["name"] != pose["name"]] + [pose])
                if safely(do_save):
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

# ---------- keyframe table ----------
st.divider()
st.subheader("Keyframes (play order, top → bottom)")
poses = load_poses()
if not poses:
    st.info("No keyframes yet. Jog the arm to a pose and save it above.")
else:
    header = st.columns([0.4, 2, 1, 4, 1, 3])
    for col, text in zip(header, ["#", "Name", "Arm", "Joints (deg)", "Gripper", ""]):
        col.markdown(f"**{text}**")
    for i, p in enumerate(poses):
        row = st.columns([0.4, 2, 1, 4, 1, 3])
        row[0].write(i + 1)
        row[1].write(p["name"])
        row[2].write(p.get("arm", "arm"))
        row[3].code("  ".join(f"{v:7.1f}" for v in p["values"]), language=None)
        row[4].write(p.get("gripper") or "–")
        go_c, up_c, down_c, del_c = row[5].columns(4)
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
            save_poses(poses[:i] + poses[i + 1:])
            st.rerun()

    run_c, _ = st.columns([1, 3])
    if run_c.button("▶ Run sequence", type="primary", use_container_width=True):
        confirm_run()

    with st.expander("Export as Python"):
        lines = ["POSES = {"]
        for p in poses:
            lines.append(f"    {p['name']!r}: ({p.get('arm', 'arm')!r}, {[round(v, 3) for v in p['values']]}),")
        lines.append("}")
        st.code("\n".join(lines), language="python")
