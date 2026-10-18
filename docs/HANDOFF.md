# Handoff: bottle stabilizer (`arm`)

This branch holds the work for **`arm`** (arm 1), the bottle stabilizer: it picks the bottle up from the box,
moves it to the end position on the table, and holds it steady while **`arm2`** unscrews the cap. `arm2` and its
cap detection are Rishi's work (merged from `cam_branch`); this document covers `arm`.

Read `README.md` for setup and the full feature list. This file is what you need to pick the work up.

## Start here

1. Follow **Setup** in `README.md`: create the venv, install `requirements.txt`, copy `env.sh.example` to
   `env.sh` and fill in your Viam API key ID and key (Viam app → armfarm16 → Connect → Code sample).
   `env.sh` is gitignored and must never be committed.
2. Run `./run.sh` and open http://127.0.0.1:8501.
3. In the sidebar, set **Active arm** to `arm`. The camera section then shows `cam` with the pink-marker
   detection; with `arm2` selected it shows Rishi's cap detection instead.

Keep the e-stop in reach. Jog buttons move the arm immediately; Go, Move, Run sequence and Corrected grab ask
for confirmation first.

## Keyframes for `arm` (in `poses.json`)

`poses.json` groups keyframes by arm. These are all in the `arm` group; the `arm2` group is empty.

| Keyframe | Joints 0–5 (°) | Gripper | Role |
|---|---|---|---|
| `parking` | −34.8, −75.8, −2.9, 164.9, 9.4, 194.1 | – | Rest pose; the camera sees the whole bottle from here (parking scan) |
| `NewMid` | −64.7, 11.8, −54.1, 107.2, 72.4, 218.7 | – | Scan pose for the corrected grab |
| `grab_pos` | −49.3, 22.0, −70.6, 117.0, 58.7, 222.2 | grab | Grabs the bottle |
| `grabframe1` | −50.7, 6.0, −68.7, 110.5, 58.6, 236.9 | – | After the grab |
| `end` | −117.6, 27.5, −43.9, 94.1, 88.6, 199.4 | – | Holding position on the table |

**Run sequence** plays these exactly as taught. It does **not** apply the camera correction. To run this sequence and then Rishi's `arm2` sequence on one computer, see `RUN_BOTH.md` (Run sequence → **arm, then arm2**).

## How the camera finds the bottle

- A **pink sticky note** on the bottle is found by color; a **black dot** drawn in its middle is the aim point
  (the pink center is used if no dot is seen). Settings live in `vision_config.json` (hue 165–179, saturation
  ≥ 100, brightness ≥ 80; shape check off).
- **Distance:** depth is used only when ≥ 60% of the marker reads and the readings agree within 40 mm. Otherwise
  it comes from the marker's height: `Z = fy × marker_h / height_px`. Near the bottle the depth camera is inside
  its minimum range (about 30–40 cm), so expect the size fallback at `NewMid`.
- **Measuring** uses the median of 5 frames; 3 must agree within 15 mm or nothing is reported.

## Corrected grab (built, not yet run on the arm)

Purpose: if the bottle is slightly misplaced, shift the grab so the gripper doesn't push it or grip it badly.

**Teach once, with the bottle placed where `grab_pos` grabs it well** (camera section → Grab correction):

1. Go `parking` → **Set parking reference**
2. Go `NewMid` → **Set reference**
3. Go `grab_pos` → **Record grab pose**

These are saved to `grab_reference.json` (not in the repo yet because nothing has been taught since the last
reset; commit it once taught). Don't re-teach with the bottle in a moved spot: that makes the correction zero.

**Each run:** at `parking` press **Scan at parking** (read-only check: bottle present and ≤ 50 mm off), Go
`NewMid`, press **Corrected grab**, check the popup, confirm. The arm goes to a pre-grab point 60 mm back along
the approach, straight in to the corrected grab, then grabs. Nothing moves if the marker isn't found
consistently, the arm isn't at `NewMid`, the correction is over 30 mm, or the parking and `NewMid` scans
disagree by more than 15 mm.

What has been tested: the correction math (synthetic), every safety gate, live 5-frame measuring (repeatable
within 2–3 mm). What has **not**: the corrected motion itself on the arm. First run: teach, nudge the bottle
~10 mm, run slowly, and check the pre-grab point stops in front of the bottle.

## Machine config changes made on armfarm16

- `cam` (arm 1's camera) now has `align_color_depth: true`, added as a fragment mod for fragment
  `859f2598-60f1-43a2-8df2-13ffbee949cc`:
  `"components.cam.attributes.align_color_depth": true`. To revert, delete that line in the machine's
  `fragment_mods` and save. `cam2` already had it.

## Open items and known issues

- **Measure the marker and the bottle.** `marker_h` (76 mm, a standard 3-inch note) and the bottle radius
  (40 mm) are assumptions. The size-based distance scales directly with `marker_h`. Set both in the pink
  detection settings and press **Save as default**.
- **Corrected grab untested on hardware**, see above.
- **Corrected grab is not in Run sequence yet.** It runs from its own button with a confirmation. Adding it to
  Run sequence (so `grab_pos` is corrected automatically, with the same gates) is the natural next step once it
  works reliably.
- **`arm` moves faster than `arm2`** even though both are configured at `speed_degs_per_sec: 60` with default
  acceleration. Likely causes: joint moves with large base/shoulder swings versus `arm2`'s capped Cartesian
  Align moves, a runtime `set_speed` DoCommand, or a controller-side limit. Not yet measured. To slow `arm`, add
  `"components.arm.attributes.speed_degs_per_sec": 30` to the same fragment mod.
- **Depth on the paper note is noisy** even at `parking` (≈ 70% valid but readings spread ≈ 600 mm), so the
  size fallback is used. Sampling only a small disk around the black dot (Rishi's approach for the cap) might
  make depth usable at `parking`; untested.
- **The black dot is small** (≈ 5 mm, ≈ 10 px from `parking`): fine as an aim point, too small for distance. A
  ~40 mm circle would be large enough to measure distance from.

## Where things are in the code

| What | Where |
|---|---|
| Pink detection, center dot, depth/size distance, 5-frame measuring | `vision.py` |
| Corrected grab, parking scan, references | `streamlit_app.py`: constants near the top (`SCAN_KEYFRAME`, `GRAB_KEYFRAME`, `PARK_KEYFRAME`, limits), and the camera section for `arm` |
| Keyframe file format (grouped by arm) | `load_poses` / `save_poses` in `streamlit_app.py` |
| `arm2` cap detection and Align (Rishi) | `streamlit_app.py`: `detect_circles`, `measure`, `plan_align`, `confirm_align` |

If you rename the `NewMid`, `grab_pos` or `parking` keyframes, update `SCAN_KEYFRAME`, `GRAB_KEYFRAME` and
`PARK_KEYFRAME` to match.
