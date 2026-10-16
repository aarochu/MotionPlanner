# MotionPlanner

A Streamlit dashboard for teaching and replaying hardcoded joint-space keyframes on Viam-controlled
xArm6 robot arms. It was built for a two-arm bottle-opening project: one arm holds the bottle while the
other unscrews the cap.

It drives the `arm` / `gripper` and `arm2` / `gripper2` components on a Viam machine (`armfarm16`).

New here (engineer or AI agent)? Read [`AGENTS.md`](AGENTS.md) for the full project context.
Picking up the bottle-stabilizer (`arm`) work? Start with [`HANDOFF.md`](HANDOFF.md).
Running Aaron's `arm` sequence and then Rishi's `arm2` sequence on one computer? See [`RUN_BOTH.md`](RUN_BOTH.md).

## Features

- Live joint angles (Joint 0–5, matching the Viam app) and end-effector pose, refreshed every second
- Jog any joint by a chosen step (0.5°–10°, capped at 15° per click)
- Type exact joint angles and move there, with a confirmation popup showing the change per joint
- Open / close the gripper
- Save the current pose as a named keyframe, optionally with a gripper action after arriving
- Reorder, delete, replay (Go) and run all keyframes in order across both arms
- Export keyframes as a Python `POSES = {...}` dict for use in your own scripts
- STOP ALL button for both arms (also stops the unscrew and the full run)
- **Unscrew** (`arm2`): regrasp unscrew of the cap (grab, twist, open, twist back; the last twist lifts the cap)
- **Full run**: `arm`'s sequence, then `arm2`'s keyframes, the unscrew, a retreat and the trash drop, in one go
- Keyframes as joints or as XYZ + orientation
- A camera section that follows the **active arm**, with a live feed and optional depth view. Each arm has its
  own detector:
  - **`arm2` (`cam2`): bottle cap** (his logic). Finds the silver cap and has an **Align** button that moves
    `arm2` so `gripper2` sits over it.
  - **`arm` (`cam`): pink marker on the bottle** (Aaron's logic). Finds the pink marker and reports its world
    position with **Locate bottle**. It is read-only and never moves the arm.

## Bottle cap detection and alignment (`arm2`)

`cam2` is a RealSense on `arm2`'s wrist, looking down. Each frame the app finds circles with a Hough
transform and keeps only the bottle cap: a circle that is mostly silver (unsaturated and at least half as bright
as the white table in the same frame, so shade or lighting changes don't matter), has almost no
dark pixels (the dark bottle body and neck fail this) and has a crisp outline most of the way round (partial
circles inside the cap fail this). **Align** uses the median cap over 5 frames, so one bad frame cannot steer
the arm. Turn on *Show rejected circles* to see what was filtered out (thin red). Its distance comes
from the depth image when at least 60% of the circle has a depth reading, otherwise from the circle's pixel size
and the cap diameter you enter (`Z = fx · D / 2r`). The center pixel is deprojected with `cam2`'s intrinsics and
transformed through the Viam frame system into `arm2`'s base frame.

**Align** keeps the arm's orientation and moves only position: XY only by default, or XY plus a hover height
above the circle. Each move is capped at 100 mm (`MAX_ALIGN_MM`); press Align again to close the rest, which
also corrects for any error in the first estimate. A confirmation shows the planned move before anything moves.

Requirements on the Viam side:

- `cam2` needs `"align_color_depth": true` so depth pixels line up with color pixels.
- The D435 cannot measure closer than about 28 cm at 1280×720. Closer than that (e.g. the cap right under the
  gripper) the app falls back to the cap-size estimate, so set the cap diameter accurately.

## Pink marker detection (`arm`)

`cam` is a RealSense on `arm`'s wrist. A pink note or band on the bottle is found by color (HSV range), and the
largest pink blob wins. An optional shape check (rectangularity and aspect ratio) can reject blobs that don't
look like a rectangle; it is off by default. The logic lives in `vision.py` and is shown in the same camera
section as the cap detector.

**Locate bottle** captures one frame and converts the marker to world x, y, z in one of two ways:

- **Depth** (camera looking level): median depth over the marker, then the bottle axis is one bottle radius
  behind that surface along the camera ray.
- **Plane** (camera looking down): casts the camera ray through the marker's center and intersects it with a
  horizontal plane at the bottle's grasp height.

Pink settings are saved with **Save as default** in `vision_config.json` and loaded on start. Nothing in the
pink path moves an arm: there is no Align for `arm` until the grasp sequence is designed.

Notes: depth only lines up with the color image if the camera has `"align_color_depth": true` in the Viam
config (`cam2` does; `cam` did not when last checked). Depth cameras of this type can't read closer than about
30 to 40 cm, so scan from 40 to 50 cm away.

## Corrected grab (`arm`)

If the bottle is slightly misplaced, the taught grab can push it or grip it badly. The corrected grab measures
the bottle from the `NewMid` keyframe and shifts the `grab_pos` grab by how far the bottle moved.

**Teach once, with the bottle placed correctly** (camera section on `arm`, *Grab correction*):

1. Go to `parking`, then press **Set parking reference**. The bottle is measured from the parking view.
2. Go to `NewMid`, then press **Set reference**. The bottle is measured over 5 frames and saved.
3. Go to `grab_pos`, then press **Record grab pose**. The gripper's pose there is saved.

Both are stored in `grab_reference.json`.

**Each run:** at `parking`, press **Scan at parking** (read-only) for a first look: it confirms the bottle is there and no more than 50 mm off before the arm goes anywhere. Then go to `NewMid` and press **Corrected grab**. The `NewMid` scan must agree with the parking scan within 15 mm, or nothing moves (the parking scan is optional but recommended). It measures the bottle (median of 5 frames; 3 must
agree within 15 mm), computes the offset from the reference, and shows the planned correction. After you
confirm, the arm moves to a pre-grab point 60 mm back along the gripper's approach, goes straight in to the
corrected grab (height and orientation as taught), and grabs.

**Black dot:** a dark dot drawn in the middle of the pink marker is used as the exact aim point (the pink center is the fallback when no dot is seen). Distance still comes from the marker's height, since a small dot is only a few pixels across.

The offset is relative to the reference, taken from the same pose with the same method, so most camera error
cancels out, even with the marker-size distance fallback. Nothing moves if the marker isn't found
consistently, if the arm isn't at `NewMid`, or if the correction is over 30 mm (`MAX_GRAB_FIX_MM`).

Moves are joint-space (`move_to_joint_positions`), so they are repeatable and the motion planner
cannot pick a surprising path.

## Full run (both arms)

**Full run** plays, one after another, with the two arms never moving at the same time:

1. **Keyframes:** every `arm` keyframe (Aaron's sequence: `parking` → `NewMid` → `grab_pos` → `grabframe1` → `end`,
   which ends holding the bottle), then every `arm2` keyframe except `trash_drop_pose`
   (`safe_bottle_start_pose` → `open_bottle_cap_up_pose` → `open_bottle_cap_pose`). Same order as Run sequence's
   "arm, then arm2".
2. **Unscrew** with the Unscrew panel's settings (below).
3. **Retreat and drop:** straight up to `open_bottle_cap_up_pose`'s height while still holding the cap, keeping the
   orientation the unscrew left (`RETREAT_POSE`); through `safe_bottle_start_pose` (move only, `VIA_POSES`); then
   `trash_drop_pose` and its gripper action (`open`, so the cap drops; `TRASH_DROP_POSE`).

Any failure, **Stop full run** or STOP ALL ends the run where it is and stops the arms; it never continues to the
drop after a failed unscrew. A confirmation lists every step of all three phases first.

Keyframes can be joints (`"values": [j0..j5]`, joint-space move) or a pose (`"pose": {"x", "y", "z", "o_x", "o_y",
"o_z", "theta"}` in mm / orientation vector, in the arm's base frame as `get_end_position` reports; moved to with the
motion planner). Choose with **Save as** when saving. The `arm2` keyframes are poses. Note the Viam app's arm card
shows world coordinates; for `arm2` (base at world y = −914 mm) the world Y is 914 mm lower than the base-frame Y.

Known issue: the planner has rejected the move to `trash_drop_pose` (its collision model reports `arm2`'s wrist
against `cam2` and the forearm against `table` at that pose, which was taught by jogging joints). If it fails there,
the run stops holding the cap at `safe_bottle_start_pose`; storing `trash_drop_pose` as joints is the simplest fix.

## Unscrew (`arm2`)

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

## Code layout

| File | What it does |
|---|---|
| `streamlit_app.py` | The dashboard: joints, jogging, gripper, keyframes, and the camera section that follows the active arm (`arm2`: cap detection and Align; `arm`: pink marker, Locate and corrected grab) |
| `vision.py` | Pink-marker detection and positioning for `arm`, with no UI code: color mask, optional shape check, center dot, depth with marker-size fallback, multi-frame measuring |
| `vision_config.json` | Saved detector settings (colors, shape gates, depth range). Loaded on start; edit it by hand or use **Save as default** |
| `poses.json` | Saved keyframes, grouped by arm: the `arm` group (arm 1, picks up and holds the bottle) and the `arm2` group (unscrews the cap). Each group says which arm it drives, and `play_order` keeps the order across both |
| `grab_reference.json` | Corrected-grab references for `arm` (created when you teach them) |
| `poses_io.py` | Reading and writing `poses.json`, and which keyframes each Run sequence choice plays |
| `import_poses.py` | Adds keyframes from another poses file (e.g. Rishi's `arm2` ones) after the existing ones |
| `cap_vision.py` | Bottle-cap detection for `cam2` (used by the camera section) and its geometry helpers |
| `cap_unscrew.py` | The `arm2` unscrew sequence, grip force/speed handling and tool levelling |
| `bottle_task.py` | The full run (both arms' keyframes → unscrew → retreat → trash drop) and the shared keyframe move |
| `cap_tracker.py` | Slow XY tracking of the cap with `arm2` (not wired into this GUI yet) |

Detection is color first, then shape. Color finds blobs; each blob is scored on rectangularity, aspect
ratio and solidity, and only blobs that pass the gates are considered. The best-scoring one is used.

## Configuration

| Variable | Purpose |
|---|---|
| `VIAM_API_KEY_ID` | Viam API key ID (required) |
| `VIAM_API_KEY` | Viam API key (required) |
| `VIAM_MACHINE_ADDRESS` | Machine address, e.g. `armfarm16-main.<location>.viam.cloud` |
| `POSES_FILE` | Where keyframes are saved (default `poses.json`) |
| `VISION_CONFIG_FILE` | Saved pink-detection settings (default `vision_config.json`) |

Arm and gripper names are set in `ARM_GRIPPERS` at the top of `streamlit_app.py`.

## Safety

- Keep the e-stop within reach. Jog buttons move the arm immediately.
- Go, Run sequence and typed moves ask for confirmation first.
- Run sequence moves one arm at a time, not both at once.
- `collision_sensitivity` on the arm config is 0 by default; consider a small non-zero value for
  first tests with the cap.
