# Project context for engineers and AI agents

This file explains what this repo is for, how the hardware is set up, what has been built, why it was built
that way, what is verified, and what is still open. Read it before changing anything. It reflects Aaron's work
up to the creation of the `aaron-main` branch.

Related docs: `README.md` (features, setup), `HANDOFF.md` (short handoff for the `arm` work), `RUN_BOTH.md`
(running both arms' sequences on one computer).

## 1. Goal

A two-arm bottle opener built on Viam. The bottle is a Saratoga 28 fl oz cobalt-blue glass bottle.

- **`arm`** (arm 1, "bottle stabilizer", Aaron): picks the bottle up from a box, moves it to an end position on
  the table, and holds it steady.
- **`arm2`** (Rishi): unscrews the cap while `arm` holds the bottle.

Everything is **hardcoded**: taught joint-space keyframes replayed in order. There is no learned policy (no
VLA). Cameras are used only to measure where the bottle or cap is, so the taught motion can be shifted slightly.

## 2. Hardware and Viam setup

One Viam machine, **armfarm16** (org `hackathons`, location `Arm Farm`, part `armfarm16-main`), runs both arms.
Address: `armfarm16-main.310sld03v2.viam.cloud`. Machine id `bef83ec4-99e5-49cd-8a73-0b2324c6673c`.

| Resource | Model | Notes |
|---|---|---|
| `arm` | `viam:ufactory:xArm6` | From shared fragment `xarm-realsense-gripper2` (`859f2598-60f1-43a2-8df2-13ffbee949cc`). Frame = `world` origin, so arm's base frame equals world |
| `cam` | `viam:camera:realsense` (likely D435) | On `arm`'s wrist. 1280×720 color + depth. `align_color_depth: true` added as a fragment mod (see §7) |
| `gripper` | `viam:ufactory:gripper` | On `arm`, 105 mm out from the flange |
| `arm2` | `viam:ufactory:xArm6` | Defined directly on the machine. Base at world (0, −723.9, 0) |
| `cam2` | `viam:camera:realsense` | On `arm2`'s wrist, looking down. `align_color_depth: true` |
| `gripper2` | `viam:ufactory:gripper` | On `arm2` |
| Obstacles | `table`, `wall-front`, `wall-side`, `ceiling`, `box-shelf` | Modeled boxes the motion planner avoids. `box-shelf` was moved by a teammate during the work |
| Others | `switch_1` (arm-position-saver), `bottle-handoff` (fake switch: unknown / station_clear / bottle_in_position) | Added by teammates; not used by this code yet |

Arm settings (both): `speed_degs_per_sec: 60` (module default), acceleration at the module default (~382°/s²),
`collision_sensitivity: 0` (collision stop off). Joints are numbered **0–5**, matching the Viam app.

| Joint | Plain meaning |
|---|---|
| 0 | Base spin: turns the whole arm left/right |
| 1 | Shoulder: tilts the arm forward/back |
| 2 | Elbow: bends the arm to reach in/out |
| 3 | Forearm roll: twists the forearm along its length |
| 4 | Wrist bend: tilts the gripper up/down |
| 5 | Wrist twist: spins the gripper (the unscrew motion; also roll when the gripper points forward) |

## 3. Repo layout

| File | What it is |
|---|---|
| `streamlit_app.py` | The dashboard (Rishi's GUI, extended). Joints, jog, gripper, keyframes, Run sequence, and a camera section that follows the active arm |
| `vision.py` | Pink-marker detection and positioning for `arm` (no UI code) |
| `poses_io.py` | `poses.json` reading/writing and which keyframes each Run choice plays (no UI code) |
| `import_poses.py` | CLI to add keyframes from another poses file after the existing ones |
| `poses.json` | Saved keyframes, grouped by arm (§5) |
| `vision_config.json` | Saved pink-detection settings (loaded on start; **Save as default** writes it) |
| `grab_reference.json` | Corrected-grab references; created when taught (not in the repo yet) |
| `run.sh` / `env.sh.example` | Launcher and credentials template. `env.sh` is gitignored |

## 4. Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp env.sh.example env.sh   # fill in VIAM_API_KEY_ID and VIAM_API_KEY (Viam app -> Connect -> Code sample)
./run.sh                   # http://127.0.0.1:8501
```

The sidebar's **Active arm** chooses which arm the jog/gripper/save controls and the camera section act on.

## 5. Keyframes and sequencing

`poses.json` groups keyframes by arm so it is unambiguous which arm each one drives:

```json
{"arm":  {"description": "...NOT for arm2.", "keyframes": [...]},
 "arm2": {"description": "...NOT for arm.",  "keyframes": [...]},
 "play_order": [{"arm": "arm", "name": "parking"}, ...]}
```

Current `arm` keyframes (`arm2` group is empty in this repo; Rishi's are on his computer):

| Keyframe | Joints 0–5 (°) | Gripper | Role |
|---|---|---|---|
| `parking` | −34.8, −75.8, −2.9, 164.9, 9.4, 194.1 | – | Rest; camera sees the whole bottle (parking scan) |
| `NewMid` | −64.7, 11.8, −54.1, 107.2, 72.4, 218.7 | – | Scan pose for the corrected grab |
| `grab_pos` | −49.3, 22.0, −70.6, 117.0, 58.7, 222.2 | grab | Grabs the bottle |
| `grabframe1` | −50.7, 6.0, −68.7, 110.5, 58.6, 236.9 | – | After the grab |
| `end` | −117.6, 27.5, −43.9, 94.1, 88.6, 199.4 | – | Holds the bottle on the table (no gripper action, so it keeps holding) |

All keyframe moves are joint-space (`move_to_joint_positions`): repeatable, and the planner cannot choose a
surprising path. Each keyframe can have a gripper action (grab/open) and a delay after it.

**Run sequence** choices (`poses_io.RUN_OPTIONS`): **arm, then arm2** (default: all arm keyframes, then all arm2),
**Both arms (table order)**, **arm only**, **arm2 only**. Arms never move at the same time. Run sequence plays
keyframes exactly as taught; camera corrections are not part of it.

## 6. Vision

### `arm2` (Rishi's logic, unchanged)

Hough circles filtered for a silver cap (silver share, few dark pixels, crisp outline); median over 5 frames;
distance from aligned depth if ≥ 60% of a center disk is valid, else from cap size (`Z = fx·D/2r`, D = cap
diameter). **Align** moves `arm2` so `gripper2` is over the cap with `move_to_position`, max 100 mm per press,
after a confirmation.

### `arm` (Aaron's logic, `vision.py`)

- **Marker:** a pink sticky note on the bottle with a small black dot in its middle. Found by HSV color
  (hue 165–179, sat ≥ 100, val ≥ 80); the largest blob wins. An optional shape check (rectangularity / aspect)
  exists but is **off**, because rotating the bottle turns the note away and makes it look tall and thin.
- **Aim point:** the black dot's center when found (it's ~5 mm, ~10 px from `parking`), else the pink centroid.
- **Distance:** depth is used only if ≥ 60% of the marker's interior reads **and** readings agree within 40 mm;
  otherwise `Z = fy × marker_h / height_px` using the marker's **height** (it runs along the bottle axis, so the
  bottle's curve and rotation don't shrink it). The bottle axis is one bottle radius behind the marker surface.
- **Pixel → world:** each capture records the camera pose via the Viam frame system (`transform_pose`), so the
  conversion stays correct even if the arm moves afterwards.
- **Measuring:** `measure_stable` takes the median of 5 frames; 3 must agree within 15 mm.

### Corrected grab (built; motion not yet run on hardware)

Differential: compare the bottle position measured now with a reference measured the same way when the bottle
was placed correctly, and shift the taught grab by the difference. Camera bias mostly cancels, which is why the
size fallback is acceptable.

- Teach once, bottle placed correctly: `parking` → **Set parking reference**; `NewMid` → **Set reference**;
  `grab_pos` → **Record grab pose** (stores the end-effector pose). Saved to `grab_reference.json`.
- Run: **Scan at parking** (read-only; bottle present and ≤ 50 mm off) → Go `NewMid` → **Corrected grab** →
  confirm. Moves to a pre-grab point 60 mm back along the approach, then straight in to the corrected grab
  (`move_to_position`, orientation and height kept), then grabs.
- Gates (nothing moves): arm not within 1° of `NewMid`; marker not found consistently; correction > 30 mm;
  parking and `NewMid` offsets disagree by > 15 mm.
- Keyframe names are coupled to code constants `SCAN_KEYFRAME`, `GRAB_KEYFRAME`, `PARK_KEYFRAME`.

## 7. Decisions and history (why things are the way they are)

- **Joint-space keyframes, not Cartesian/planner moves,** for repeatability. Cartesian `move_to_position` is used
  only for small, vision-derived corrections (Rishi's Align, Aaron's corrected grab).
- **Marker evolution:** blue mesh tape → rejected (same blue as the table tape and the bottle glass); pink note →
  clean single blob; black dot added for a sharp aim point. A single note on a cylinder shifts its centroid up to
  one radius when the bottle rotates; a full pink band around the bottle would avoid that (not done).
- **Depth problems found and fixed/handled:** `cam` originally lacked `align_color_depth`, so depth didn't line up
  with color. Fixed by the fragment mod `"components.cam.attributes.align_color_depth": true` (revert: delete
  that line in `fragment_mods`). Close up (marker > ~250 px tall, ~22–30 cm) the D435 can't read at all; at
  `parking` (~45 cm) ~70% of pixels read but they spread ~600 mm on the paper-on-glass note, so the size fallback
  is used there too.
- **Merge with Rishi (`cam_branch`):** his GUI became the base; the camera section follows the active arm; `arm2`
  uses only his cap logic and `arm` only the pink logic. His commit was squashed in with him as author and
  without its AI co-author trailer.
- **Keyframes as config:** `poses.json` was gitignored on `cam_branch`; here it is tracked and grouped by arm.
- **Run "arm, then arm2"** was added so one computer runs Aaron's sequence, then Rishi's.

## 8. Verified vs not verified

Verified: live joint reads and jogging; keyframe save/Go; detection on live frames; depth decoding matches the
SDK; frame transforms for both cameras; size fallback math; corrected-grab math and every gate; 5-frame
measuring repeatable within 2–3 mm; parking scan; poses grouping round trip; import script; Run choice and
dialog (opened, not executed) — all via headless `streamlit.testing` runs and read-only probes.

Not verified: the corrected-grab motion on the arm; a combined arm→arm2 run; Rishi's real keyframe file;
absolute accuracy of positions against a tape measure; the alignment fix's effect (only indirectly).

## 9. Open items / next steps

1. Measure the pink note's height and the bottle radius; set `marker_h` and `radius`, **Save as default**
   (defaults 76 mm and 40 mm are assumptions; size distance scales with `marker_h`).
2. Teach the three corrected-grab references and try the corrected grab slowly; check the pre-grab point stops in
   front of the bottle. Commit `grab_reference.json`.
3. Add the corrected grab to Run sequence once reliable (correct `grab_pos` automatically, same gates).
4. Rishi: pull, import his `arm2` keyframes (`RUN_BOTH.md`), commit `poses.json`.
5. `arm` appears faster than `arm2` with identical config; measure both on the same jog. To slow `arm`, add
   `"components.arm.attributes.speed_degs_per_sec": 30` to the same fragment mod.
6. Possibly try Rishi's center-disk depth sampling around the dot at `parking`; possibly a ~40 mm circle for
   circle-based distance; possibly use the `bottle-handoff` switch to signal arm → arm2.

## 10. Working rules (from the project owner)

- **Never move an arm without the user's go-ahead.** Read-only checks (joint reads, camera frames, transforms)
  are fine. Moves in the app always go through a confirmation; keep it that way.
- **Never commit secrets.** API keys live only in `env.sh` (gitignored). Scan staged changes for keys before
  committing. The machine address is not a secret.
- **No AI co-author or attribution lines** in commit messages or PRs.
- Commit and push only when asked; prefer a separate branch over rewriting someone else's.
- Machine config changes are live and immediate (no undo): ask first, and record exactly what changed and how to
  revert it. Fragment-provided components (`arm`, `cam`, `gripper`) can only be changed through `fragment_mods`
  in the Viam app; editing the shared fragment itself would affect other machines.

## 11. Gotchas

- `get_robot()` is `st.cache_resource`: one robot connection per server process. Restart `./run.sh` after
  changing what it returns.
- Streamlit fragments (live feeds) keep the globals of the last full run; the camera globals (`CAM`, `CAMERA`,
  `INTRINSICS`, `CAM_ARM`) are set right after the sidebar picks the arm.
- Headless test pattern: `streamlit.testing.v1.AppTest.from_file("streamlit_app.py").run()` with `env.sh`
  sourced; point `POSES_FILE` / `GRAB_REF_FILE` / `VISION_CONFIG_FILE` at temp copies so real files aren't
  touched, and never click the confirm buttons ("Yes, move", "Yes, run").
- Rishi's checkout has an untracked `poses.json`; git refuses to check out a branch that tracks it until it is
  moved aside (`RUN_BOTH.md` covers this).
- Streamlit prints `use_container_width` deprecation warnings; harmless.

## 12. Branches

| Branch | Contents |
|---|---|
| `main` | Everything through the `arm` keyframes |
| `cam_branch` | Rishi's original cam2/cap work (its commit message has an AI co-author trailer; the squashed copy in this history does not) |
| `bottle-stabilizer` | `main` + `HANDOFF.md` |
| `combined-sequence` | + Run "arm, then arm2", `poses_io.py`, `import_poses.py`, `RUN_BOTH.md` |
| `aaron-main` | Everything above + this file. **Start here.** |
