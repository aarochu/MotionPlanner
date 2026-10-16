# Running both sequences on one computer

This branch (`combined-sequence`) has Aaron's sequence for **`arm`** (picks up the bottle and holds it) saved in
`poses.json`. The goal: one computer runs Aaron's `arm` sequence first, then Rishi's `arm2` sequence (unscrews
the cap).

Both arms are on the same Viam machine (armfarm16), so one dashboard on one computer controls both.

## For Rishi: pull this branch and add your arm2 keyframes

Your `arm2` keyframes only exist in your local `poses.json`; `cam_branch` git-ignores that file, so they were
never pushed. Here `poses.json` is tracked, so move yours aside before switching branches (git refuses to
overwrite an untracked file otherwise), then import it.

From your MotionPlanner folder:

```bash
mv poses.json ~/rishi_poses_backup.json
```

```bash
git fetch origin
```

```bash
git checkout combined-sequence
```

```bash
.venv/bin/pip install -r requirements.txt
```

Preview the import first. It should list your arm2 keyframes as "added":

```bash
.venv/bin/python import_poses.py ~/rishi_poses_backup.json --dry-run
```

Then import for real (poses.json is backed up to poses.json.bak first):

```bash
.venv/bin/python import_poses.py ~/rishi_poses_backup.json
```

If it says "no arm2 keyframes", your saved keyframes aren't tagged `"arm": "arm2"`; check the file. Untagged
keyframes count as `arm`.

Keep your own `env.sh` (it's gitignored), then start the dashboard:

```bash
./run.sh
```

## Run it

1. Open http://127.0.0.1:8501 and check the **Keyframes** table: Aaron's `arm` keyframes (`parking`, `NewMid`,
   `grab_pos`, `grabframe1`, `end`) and your `arm2` ones.
2. Under **Run which keyframes**, choose **arm, then arm2** (the default).
3. Press **▶ Run sequence** and confirm. It plays every `arm` keyframe, then every `arm2` keyframe, one move at a
   time. The two arms never move at the same time.

Keep the e-stop in reach, especially on the first run.

## Full run (keyframes, unscrew, drop in one go)

On `full-sequence` the `arm2` keyframes are already in `poses.json`. The **Full run** panel plays all of `arm`'s
keyframes, then `arm2`'s (except `trash_drop_pose`), then the unscrew (Unscrew panel settings), then lifts the cap
straight up, passes `safe_bottle_start_pose` and drops the cap at `trash_drop_pose`. Turn off the Unscrew panel's
**Dry run** for a real cap. See README → Full run.

## Good to know

- **Aaron's sequence ends at `end` holding the bottle.** `end` has no gripper action, so the gripper stays closed
  while `arm2` works.
- **Run sequence plays keyframes as taught.** Camera steps are not part of it: Rishi's **Align** (cap) and
  Aaron's **Corrected grab** (bottle) still run from their own buttons in the camera section.
- **Other choices:** "Both arms (table order)" plays the table top to bottom; "arm only" / "arm2 only" play one arm.
- **Commit the result.** After importing, commit `poses.json` on this branch so both sequences live in the repo
  and the other person can pull them.
- **Delays** in the keyframe table still apply between steps, including the step where `arm` hands over to `arm2`.
