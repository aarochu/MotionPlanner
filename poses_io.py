"""Keyframe file (poses.json) reading, writing and sequencing. No Streamlit or robot code.

The file groups keyframes by the arm they drive, so it's unambiguous which arm each one belongs to:

    {"arm":  {"description": "...", "keyframes": [...]},
     "arm2": {"description": "...", "keyframes": [...]},
     "play_order": [{"arm": "arm", "name": "parking"}, ...]}

An older flat list of keyframes (each with an "arm" field, or none meaning "arm") is still read.
"""
import json
from pathlib import Path

ARMS = ("arm", "arm2")
GROUP_DESC = {
    "arm": "Keyframes for arm (arm 1, with cam and gripper): picks up the bottle and holds it. NOT for arm2.",
    "arm2": "Keyframes for arm2 (with cam2 and gripper2): unscrews the cap. NOT for arm.",
}

# Run sequence choices: key -> label
RUN_OPTIONS = {
    "arm_then_arm2": "arm, then arm2",
    "both": "Both arms (table order)",
    "arm": "arm only",
    "arm2": "arm2 only",
}


def parse(data):
    """Flat list of keyframes (each with its "arm") in play order, from either file format."""
    if isinstance(data, list):
        return [dict(kf, arm=kf.get("arm", "arm")) for kf in data]
    order = data.get("play_order")
    flat = [dict(kf, arm=arm_name) for arm_name, group in data.items() if isinstance(group, dict)
            for kf in group.get("keyframes", [])]
    if order:  # keep the cross-arm play order the table and Run sequence use
        rank = {(o["arm"], o["name"]): n for n, o in enumerate(order)}
        flat.sort(key=lambda kf: rank.get((kf["arm"], kf["name"]), len(rank)))
    return flat


def load(path):
    path = Path(path)
    return parse(json.loads(path.read_text())) if path.exists() else []


def save(path, poses):
    """Write keyframes grouped by arm, with a description per group, plus the cross-arm play order."""
    data = {}
    for arm_name in list(ARMS) + sorted({p.get("arm", "arm") for p in poses} - set(ARMS)):
        group = [p for p in poses if p.get("arm", "arm") == arm_name]
        data[arm_name] = {"description": GROUP_DESC.get(arm_name, f"Keyframes for {arm_name}."), "keyframes": group}
    data["play_order"] = [{"arm": p.get("arm", "arm"), "name": p["name"]} for p in poses]
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def sequence(poses, selection):
    """Keyframes to play for a Run sequence choice (see RUN_OPTIONS).

    "arm_then_arm2" plays every arm keyframe (in its own order) and then every arm2 keyframe, whatever the table
    order; "both" plays the table order; "arm" / "arm2" play one arm only."""
    if selection == "arm_then_arm2":
        return [p for p in poses if p["arm"] == "arm"] + [p for p in poses if p["arm"] == "arm2"]
    if selection in ARMS:
        return [p for p in poses if p["arm"] == selection]
    return list(poses)
