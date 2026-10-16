# MotionPlanner

A Streamlit dashboard for teaching and replaying hardcoded joint-space keyframes on Viam-controlled
xArm6 robot arms. It was built for a two-arm bottle-opening project: one arm holds the bottle while the
other unscrews the cap.

It drives the `arm` / `gripper` and `arm2` / `gripper2` components on a Viam machine (`armfarm16`).

## Features

- Live joint angles (Joint 0–5, matching the Viam app) and end-effector pose, refreshed every second
- Jog any joint by a chosen step (0.5°–10°, capped at 15° per click)
- Type exact joint angles and move there, with a confirmation popup showing the change per joint
- Open / close the gripper
- Save the current pose as a named keyframe, optionally with a gripper action after arriving
- Reorder, delete, replay (Go) and run all keyframes in order across both arms
- Export keyframes as a Python `POSES = {...}` dict for use in your own scripts
- Keyframes can be joints (`"values": [j0..j5]`, joint-space move) or a pose (`"pose": {"x", "y", "z", "o_x", "o_y", "o_z", "theta"}` in mm / orientation vector, in the arm's base frame as `get_end_position` reports; moved to with the motion planner). Choose with **Save as** when saving
- STOP ALL button for both arms
- Live `cam2` feed (wrist camera on `arm2`) with circle detection (e.g. a bottle cap), optional depth view,
  and an **Align** button that moves `arm2` so `gripper2` sits over the circle

## Camera alignment

`cam2` is a RealSense on `arm2`'s wrist, looking down. Each frame the app finds circles with a Hough
transform and keeps only the bottle cap: a circle that is mostly silver (bright, unsaturated), has almost no
dark pixels (the dark bottle body and neck fail this) and has a crisp outline most of the way round (partial
circles inside the cap fail this). **Align** uses the median cap over 5 frames, so one bad frame cannot steer
the arm. Turn on *Show rejected circles* to see what was filtered out (thin red). Its distance comes
from the depth image when at least 60% of the circle has a depth reading, otherwise from the circle's pixel size
and the cap diameter you enter (`Z = fx · D / 2r`). The center pixel is deprojected with `cam2`'s intrinsics and
transformed through the Viam frame system into `arm2`'s base frame.

**Align** keeps the arm's orientation and moves only position: XY only by default, or XY plus a hover height
above the circle. Each move is capped at 100 mm (`MAX_ALIGN_MM`); press Align again to close the rest, which
also corrects for any error in the first estimate. A confirmation shows the planned move before anything moves.

## Code layout

| File | What it does |
|---|---|
| `streamlit_app.py` | Dashboard UI; keyframes, jog, camera panel |
| `cap_vision.py` | Cap detection, depth/size distance, frame-system geometry. Never moves the arm |
| `cap_tracker.py` | Optional slow XY tracking of the cap (toggle, off by default; on hold) |
| `cap_unscrew.py` | Regrasp unscrew sequence and tool levelling |
| `bottle_task.py` | Full run: keyframes → unscrew → trash drop |

## Full run

**Full run** plays, one after another: (1) every keyframe in table order except `trash_drop_pose`, (2) the unscrew
sequence with the Unscrew panel's settings, (3) straight up to `open_bottle_cap_up_pose`'s height while still holding
the cap, keeping the orientation the unscrew left (so it clears the bottle; `RETREAT_POSE` in `streamlit_app.py`), through `safe_bottle_start_pose` (move only, `VIA_POSES`), then `trash_drop_pose` and its
gripper action (`open`, so the cap drops). Any failure or Stop ends the run where it is and stops the arms; it never continues to the drop after a
failed unscrew. The drop keyframe name is `TRASH_DROP_POSE` in `streamlit_app.py`. Pose keyframes are in the arm's
base frame; the Viam app's arm card shows world coordinates (for `arm2` the world Y is 914 mm lower).

## Unscrew

With the other arm holding the bottle, press **Unscrew**. The gripper opens to the release width and `arm2`
moves to the **Start pose** keyframe (default `open_bottle_cap_pose`, `UNSCREW_START_POSE` in
`streamlit_app.py`), which should put the open fingers around the cap. It then **levels the tool to point exactly straight down**,
pivoting about the gripper (105 mm from the flange) so the fingers stay over the cap, and refuses if the pose
is more than 15° off vertical. **Grip force** (G2 gripper, 1–100 %, default 60) applies to every grab and is
reset to 50 % afterwards. Then each twist: **grab → unscrew** (rotate Joint 5 by the twist angle) **→ open →
twist back**, all at the same height. The last twist keeps hold of the cap and **lifts it straight up** by the
final lift, and the sequence ends there (it does not return to the start pose). The arm runs at
the **Speed** setting (default 60°/s) and its speed is restored to 60°/s afterwards, even on error. Stop / STOP ALL halt it.

It refuses to start unless the tool points straight down. With the tool down, +Joint 5 turns clockwise seen
from above, so loosening is −Joint 5. **Do a dry run first** (on by default), without a cap: the full cycle
runs, including open/close and the final lift, but an empty grab is not an error. Watch that the
wrist turns counter-clockwise seen from above while closed; if not, switch the direction.

Levelling builds the smallest tilt that points the tool down in the tool's own frame, lets the frame system
convert it, and checks it before moving (axis down, gripper point fixed, roll unchanged).

Don't command `arm2` from anywhere else (Viam app, another script, a config save) while a sequence runs: the
xArm driver cancels the move in progress, which shows up as `context canceled`.

Requirements on the Viam side:

- `cam2` needs `"align_color_depth": true` so depth pixels line up with color pixels.
- The D435 cannot measure closer than about 28 cm at 1280×720. Closer than that (e.g. the cap right under the
  gripper) the app falls back to the cap-size estimate, so set the cap diameter accurately.

Moves are joint-space (`move_to_joint_positions`), so they are repeatable and the motion planner
cannot pick a surprising path.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp env.sh.example env.sh
```

Edit `env.sh` and fill in your Viam API key ID and API key. Get them from the Viam app:
your machine → **Connect** → Code sample. Set `VIAM_MACHINE_ADDRESS` to your machine's address.

`env.sh` is gitignored. Never commit it.

## Run

```bash
./run.sh
```

Then open http://127.0.0.1:8501.

## Configuration

| Variable | Purpose |
|---|---|
| `VIAM_API_KEY_ID` | Viam API key ID (required) |
| `VIAM_API_KEY` | Viam API key (required) |
| `VIAM_MACHINE_ADDRESS` | Machine address, e.g. `armfarm16-main.<location>.viam.cloud` |
| `POSES_FILE` | Where keyframes are saved (default `poses.json`) |

Arm and gripper names are set in `ARM_GRIPPERS` at the top of `streamlit_app.py`.

## Safety

- Keep the e-stop within reach. Jog buttons move the arm immediately.
- Go, Run sequence and typed moves ask for confirmation first.
- Run sequence moves one arm at a time, not both at once.
- `collision_sensitivity` on the arm config is 0 by default; consider a small non-zero value for
  first tests with the cap.
