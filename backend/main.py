"""FastAPI backend for the two-arm bottle opener.

Serves the single-page frontend (frontend/index.html) and exposes REST endpoints
for robot control. The frontend talks to this over HTTP; all Viam calls live here.

Run with:
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from robot import (
    connect, disconnect, joints, end_position, move_joint_positions,
    move_position, gripper_action, camera_frame, list_arm_names,
    list_all_resource_names,
)

app = FastAPI(title="Viam Bottle Opener Backend")

# Resolve paths relative to this file so cwd doesn't matter.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
POSES_FILE = Path(os.environ.get("POSES_FILE", str(REPO_ROOT / "poses.json")))


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


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    connect()


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
    """List configured arm resource names."""
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
    """Play a list of keyframes in order, one arm at a time."""
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
                import time
                time.sleep(float(kf.delay))

        return {"status": "ok", "played": len(seq.keyframes)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@app.get("/api/camera/{name}/frame")
def api_camera_frame(name: str) -> dict[str, Any]:
    import base64
    try:
        jpg = camera_frame(name)
        return {"jpeg_base64": base64.b64encode(jpg).decode("ascii")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
