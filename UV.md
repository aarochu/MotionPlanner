# uv: pulling the repo and validating it (for Rishi, Alex and their agents)

The repo now uses [uv](https://docs.astral.sh/uv/) instead of `pip` + a hand-made venv. Only tooling changed:
no Python logic, no keyframes and no settings. The four dependencies and their pins are the same as before.

| Before | Now |
|---|---|
| `python3 -m venv .venv` + `.venv/bin/pip install -r requirements.txt` | `uv sync` |
| `.venv/bin/python script.py` | `uv run python script.py` |
| `run.sh` started `.venv/bin/streamlit` | `run.sh` starts `uv run streamlit` |
| Versions loosely pinned in `requirements.txt` | Exact versions for every package in `uv.lock` |
| Any Python | Python 3.13 via `.python-version` (3.11+ is the minimum; `streamlit` 1.66 needs it) |

`requirements.txt` is still there, exported from `uv.lock`, for anyone not using uv. Don't edit it by hand.

## 1. Install uv (once per computer)

macOS: `brew install uv`. Other systems: https://docs.astral.sh/uv/getting-started/installation/. Check with
`uv --version`.

## 2. Pull and validate

Once the pull request is merged, pull `main`. To review it before merging, check out `uv-migration` instead.

```bash
git fetch origin
```

```bash
git checkout main && git pull --ff-only
```

Then run the offline checks (installs exactly what `uv.lock` pins, compiles everything, runs the unit tests):

```bash
scripts/validate.sh
```

With `env.sh` filled in and armfarm16 online, also run the live check. It renders the dashboard for `arm` and
`arm2` against the real machine and fails on any error. It is read-only: it never presses a button, so no arm
moves and no file is written.

```bash
scripts/validate.sh --live
```

Expected result: `Ran 16 tests … OK`, then for each of `arm` and `arm2` a list of sections (Live joints, Jog,
Gripper, Save keyframe, Move to joint angles, Camera, YOLO bottle preview, Unscrew cap, Full run, Sequence steps),
`smoke test passed`, and finally `all checks passed`.

Start the dashboard as before:

```bash
./run.sh
```

## 3. If you have an old pip-made `.venv`

`uv sync` reuses the `.venv` folder and brings it to the locked versions. If anything behaves oddly, rebuild it:

```bash
rm -rf .venv && uv sync
```

## 4. Bringing your own branch up to date (`rishi-sequence`, `full-sequence`, `feature/arm-1-control`, …)

Merge `main` into your branch after this pull request lands:

```bash
git merge origin/main
```

- Conflicts in `requirements.txt` or `run.sh`: take `main`'s version (the uv one).
- If your branch added a dependency to `requirements.txt`, add it the uv way instead, which also updates the lock:
  `uv add <package>`. Commit `pyproject.toml` and `uv.lock` together.
- Then run `scripts/validate.sh --live` again.

## 5. Rules for agents working in this repo

- Use uv for everything Python: `uv sync`, `uv run …`, `uv add` / `uv remove`. Never `pip install`.
- Commit `pyproject.toml` and `uv.lock` together. If `uv sync --locked` fails with "lock file needs to be
  updated", someone changed `pyproject.toml` without relocking: run `uv lock`, check the diff, commit it.
- After changing dependencies, refresh the export: `uv export --format requirements-txt --no-hashes --no-dev -o requirements.txt`.
- Validate before pushing: `scripts/validate.sh` (and `--live` when the machine is reachable).
- Safety rules from `AGENTS.md` still apply: never move an arm without the user's go-ahead, never commit `env.sh`
  or keys, no AI co-author or attribution lines in commits.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Install uv (step 1) and open a new terminal |
| `lock file needs to be updated` | `uv lock`, review, commit `uv.lock` |
| uv downloads Python 3.13 on first run | Expected: it follows `.python-version` |
| Live check can't connect | Check `env.sh` (key ID, key, address) and that armfarm16 is online in the Viam app |
| `context canceled` during a run | Something else commanded the arm mid-move (see README); not a uv issue |
