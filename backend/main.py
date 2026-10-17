"""FastAPI backend for the two-arm bottle opener.

Serves the single-page frontend (frontend/index.html) and exposes REST endpoints
for robot control. The frontend talks to this over HTTP; all Viam calls live here.

Run with:
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import importlib.util
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from robot import (
    connect, disconnect, joints, end_position, move_joint_positions,
    move_position, gripper_action, camera_frame, list_arm_names,
    list_all_resource_names, arm_stop, gripper_stop,
)
from yolo_grab import DEFAULTS as YOLO_DEFAULTS, pick_and_place as yg_pick_and_place, reset_arm as yg_reset_arm
from cap_unscrew import UnscrewParams, Unscrewer, describe as unscrew_describe
from bottle_task import BottleTask, go_to_keyframe as bt_go_to_keyframe, apply_gripper as bt_apply_gripper

# Load poses_io from the repo root without adding it to sys.path (avoids shadowing backend modules).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_poses_spec = importlib.util.spec_from_file_location("poses_io", _REPO_ROOT / "poses_io.py")
_poses_io = importlib.util.module_from_spec(_poses_spec)
_poses_spec.loader.exec_module(_poses_io)
sequence = _poses_io.sequence
load_poses = _poses_io.load

app = FastAPI(title="Viam Bottle Opener Backend")

# Resolve paths relative to this file so cwd doesn't matter.
BACKEND_DIR = Path(__file__).resolve().parent
POSES_FILE = Path(os.environ.get("POSES_FILE", str(_REPO_ROOT / "poses.json")))

# Singletons (created at startup after robot connects)
_UNSCREWER: Unscrewer | None = None
_BOTTLE_TASK: BottleTask | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class JointMove(BaseModel):
    values: list[float]


class PoseMove(BaseModel):
    x: float
    y: float
    z: float
    o_x: float
    o_y: float
    o_z: float
    theta: float


class GripperAction(BaseModel):
    action: str  # "grab" or "open"


class Keyframe(BaseModel):
    arm: str
    name: str
    values: list[float] | None = None
    pose: dict[str, float] | None = None
    gripper: str | None = None
    delay: float | None = None


class RunSequence(BaseModel):
    keyframes: list[Keyframe]


class UnscrewRequest(BaseModel):
    twists: int = 4
    twist_deg: float = 90.0
    loosen_sign: int = -1
    release_pos: int = 500
    final_lift_mm: float = 30.0
    speed_deg_s: float = 20.0
    normal_speed_deg_s: float = 60.0
    grip_force_pct: int = 50
    normal_grip_force_pct: int = 50
    max_level_tilt_deg: float = 15.0
    start_pose: str = ""
    start_joints: list[float] | None = None
    start_target: list[float] | None = None
    dry_run: bool = False


class PickPlaceRequest(BaseModel):
    # Uses defaults from yolo_grab.DEFAULTS; only override what's needed
    pass


class FullRunRequest(BaseModel):
    # Runs all keyframes (arm then arm2), then unscrew, then retreat/drop
    unscrew: UnscrewRequest | None = None


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    connect()
    global _UNSCREWER, _BOTTLE_TASK
    _UNSCREWER = Unscrewer(arm_name="arm2", gripper_name="gripper2")
    _BOTTLE_TASK = BottleTask(
        arms={"arm": "arm", "arm2": "arm2"},
        grippers={"arm": "gripper", "arm2": "gripper2"},
        unscrewer=_UNSCREWER,
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    disconnect()


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(BACKEND_DIR.parent / "frontend" / "index.html")


# ---------------------------------------------------------------------------
# Robot resources
# ---------------------------------------------------------------------------
@app.get("/api/arms")
def api_arms() -> dict[str, Any]:
    try:
        names = list_arm_names()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"arms": names}


@app.get("/api/resources")
def api_resources() -> dict[str, Any]:
    try:
        return list_all_resource_names()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/arm/{name}/joints")
def api_joints(name: str) -> dict[str, Any]:
    try:
        return {"arm": name, "values": joints(name)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/arm/{name}/position")
def api_position(name: str) -> dict[str, Any]:
    try:
        return {"arm": name, **end_position(name)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/move/joints")
def api_move_joints(name: str, move: JointMove) -> dict[str, str]:
    if len(move.values) != 6:
        raise HTTPException(status_code=400, detail="Expected 6 joint values")
    try:
        move_joint_positions(name, move.values)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/move/position")
def api_move_position(name: str, move: PoseMove) -> dict[str, str]:
    try:
        move_position(name, move.x, move.y, move.z, move.o_x, move.o_y, move.o_z, move.theta)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/gripper/{action}")
def api_gripper(name: str, action: str) -> dict[str, str]:
    if action not in ("grab", "open"):
        raise HTTPException(status_code=400, detail="action must be 'grab' or 'open'")
    try:
        gripper_action(name, action)
        return {"status": "ok", "arm": name, "action": action}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/stop")
def api_arm_stop(name: str) -> dict[str, str]:
    try:
        arm_stop(name)
        gripper_stop(name)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Keyframes / poses
# ---------------------------------------------------------------------------
@app.get("/api/poses")
def api_poses() -> dict[str, Any]:
    if not POSES_FILE.exists():
        return {"poses": {}}
    data = json.loads(POSES_FILE.read_text())
    return {"poses": data}


@app.post("/api/poses/run")
def api_run_poses(seq: RunSequence) -> dict[str, str]:
    """Play a list of keyframes in order, one arm at a time (blocking)."""
    try:
        for kf in seq.keyframes:
            arm_name = kf.arm
            d = kf.model_dump()
            if d.get("pose") is not None:
                p = d["pose"]
                move_position(arm_name,
                              float(p["x"]), float(p["y"]), float(p["z"]),
                              float(p["o_x"]), float(p["o_y"]), float(p["o_z"]), float(p["theta"]))
            else:
                move_joint_positions(arm_name, kf.values or [])
            if kf.gripper:
                gripper_action(arm_name, kf.gripper)
            if kf.delay and kf.delay > 0:
                time.sleep(float(kf.delay))
        return {"status": "ok", "played": len(seq.keyframes)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# YOLO pick & place (arm 1)
# ---------------------------------------------------------------------------
@app.post("/api/pick_and_place")
def api_pick_and_place() -> dict[str, str]:
    """Run the full YOLO pick & place on arm 1 (blocking)."""
    try:
        cfg = dict(YOLO_DEFAULTS)
        result = yg_pick_and_place(
            arm_name="arm", gripper_name="gripper",
            cam_name="cam", cam_frame="cam", grip_frame="gripper",
            newmid=[-64.7, 11.8, -54.1, 107.2, 72.4, 218.7],
            parking=[-34.8, -75.8, -2.9, 164.9, 9.4, 194.1],
            cfg=cfg,
        )
        return {"status": "ok", "frames": result[0]["frames"], "width_mm": result[0]["width_mm"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/reset_arm1")
def api_reset_arm1() -> dict[str, str]:
    """Reset arm 1: open gripper, move to parking keyframe."""
    try:
        yg_reset_arm(
            arm_name="arm", gripper_name="gripper",
            parking=[-34.8, -75.8, -2.9, 164.9, 9.4, 194.1],
        )
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Unscrew (arm 2)
# ---------------------------------------------------------------------------
@app.post("/api/unscrew")
def api_unscrew(req: UnscrewRequest | None = None) -> dict[str, str]:
    """Run the cap unscrew sequence on arm 2 (blocking)."""
    if _UNSCREWER is None:
        raise HTTPException(status_code=500, detail="Unscrewer not initialized")
    if _UNSCREWER.running:
        raise HTTPException(status_code=409, detail="Unscrew already running")
    try:
        p = UnscrewParams(**(req.model_dump() if req else {}))
        _UNSCREWER.start(p)
        # Wait for completion (blocking endpoint)
        while _UNSCREWER.running:
            time.sleep(0.1)
        u = _UNSCREWER.status()
        if u["state"] != "done":
            raise HTTPException(status_code=500, detail=f"unscrew {u['state']} at '{u['step']}': {u.get('error')}")
        return {"status": "ok", "step": u["step"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/unscrew/describe")
def api_unscrew_describe() -> dict[str, Any]:
    """Human-readable step list for the current unscrew params."""
    p = UnscrewParams()
    return {"steps": unscrew_describe(p)}


@app.post("/api/unscrew/stop")
def api_unscrew_stop() -> dict[str, str]:
    if _UNSCREWER:
        _UNSCREWER.stop()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Full run (arm's keyframes -> arm2 keyframes -> unscrew -> retreat -> drop)
# ---------------------------------------------------------------------------
@app.post("/api/full_run")
def api_full_run(req: FullRunRequest | None = None) -> dict[str, str]:
    """Run the complete bottle-opening sequence (blocking)."""
    if _BOTTLE_TASK is None:
        raise HTTPException(status_code=500, detail="BottleTask not initialized")
    if _BOTTLE_TASK.running:
        raise HTTPException(status_code=409, detail="Full run already running")
    try:
        all_kf = load_poses()
        keyframes = sequence(all_kf, "arm_then_arm2")
        # Remove trash_drop from keyframe list (it's handled separately as drop)
        drop_kf = next((k for k in keyframes if k["arm"] == "arm2" and k["name"] == "trash_drop_pose"), None)
        pre_kf = [k for k in keyframes if k != drop_kf]
        # alongside: arm2 starts safe_bottle_start_pose together with arm's first keyframe
        alongside = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == "safe_bottle_start_pose"), None)
        if alongside is not None and pre_kf and pre_kf[0]["arm"] != "arm2":
            pre_kf.remove(alongside)
        # via: safe_bottle_start_pose (move only, after unscrew)
        via = [k for k in pre_kf if k["arm"] == "arm2" and k["name"] == "safe_bottle_start_pose"]
        # retreat: open_bottle_cap_up_pose (rise up holding cap)
        retreat = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == "open_bottle_cap_up_pose"), None)
        p = UnscrewParams(**(req.unscrew.model_dump() if req and req.unscrew else {}))
        _BOTTLE_TASK.start(pre_kf, p, drop_kf, retreat, via, alongside)
        while _BOTTLE_TASK.running:
            time.sleep(0.1)
        s = _BOTTLE_TASK.status()
        if s["state"] != "done":
            raise HTTPException(status_code=500, detail=f"full run {s['state']} at '{s['step']}': {s.get('error')}")
        return {"status": "ok", "step": s["step"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/full_run/stop")
def api_full_run_stop() -> dict[str, str]:
    if _BOTTLE_TASK:
        _BOTTLE_TASK.stop()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status() -> dict[str, Any]:
    result = {"arms": {}}
    for name in ("arm", "arm2"):
        try:
            result["arms"][name] = {
                "joints": joints(name),
                "position": end_position(name),
            }
        except Exception:
            result["arms"][name] = {"error": "unavailable"}
    if _UNSCREWER:
        result["unscrew"] = _UNSCREWER.status()
    if _BOTTLE_TASK:
        result["full_run"] = _BOTTLE_TASK.status()
    return result


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
