"""Viam robot client wrapper for the bottle-opener backend.

Holds a single RobotClient connection (cached) and exposes small async helpers
used by the FastAPI routes. All Viam calls go through `run()` so the async SDK
can be called from FastAPI's sync route functions via asyncio.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient
from viam.proto.common import Pose

# Load .env from the backend directory (and fall back to repo root)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
API_KEY_ID = os.environ.get("VIAM_API_KEY_ID", "")
API_KEY = os.environ.get("VIAM_API_KEY", "")

RIGS = {"arm": ("cam", "gripper"), "arm2": ("cam2", "gripper2")}


class Robot:
    """Cached robot connection + convenience accessors."""

    def __init__(self) -> None:
        self._robot: RobotClient | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> RobotClient:
        if self._robot is None:
            async with self._lock:
                if self._robot is None:
                    opts = RobotClient.Options()
                    if API_KEY_ID and API_KEY:
                        opts = RobotClient.Options(
                            refresh_interval=0,
                            auth_entity=ADDRESS.split(".")[0],
                        )
                    self._robot = await RobotClient.at_address(ADDRESS, opts)
        return self._robot

    async def close(self) -> None:
        if self._robot is not None:
            await self._robot.close()
            self._robot = None

    def arm(self, name: str) -> Arm:
        return Arm.from_robot(self._sync_robot(), name)

    def gripper(self, name: str) -> Gripper:
        return Gripper.from_robot(self._sync_robot(), name)

    def camera(self, name: str) -> Camera:
        return Camera.from_robot(self._sync_robot(), name)

    def _sync_robot(self) -> RobotClient:
        """Blocking accessor: use inside run() only."""
        if self._robot is None:
            raise RuntimeError("Robot not connected; call connect() first")
        return self._robot


# Module-level singleton used by route handlers.
_robot = Robot()
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


def run(coro) -> Any:
    """Run an async coroutine on the module event loop (blocking)."""
    loop = _get_loop()
    return loop.run_until_complete(coro)


def connect() -> None:
    """Connect the robot synchronously (called at import / startup)."""
    run(_robot.connect())


def disconnect() -> None:
    run(_robot.close())


def joints(name: str) -> list[float]:
    """Read current joint positions (values list) for an arm."""
    return list(run(_robot.arm(name).get_joint_positions()).values)


def end_position(name: str) -> dict[str, float]:
    """Read the current end position pose for an arm."""
    e = run(_robot.arm(name).get_end_position())
    return {
        "x": float(e.x), "y": float(e.y), "z": float(e.z),
        "o_x": float(e.o_x), "o_y": float(e.o_y), "o_z": float(e.o_z), "theta": float(e.theta),
    }


def move_joint_positions(name: str, values: list[float]) -> None:
    """Move an arm to joint positions (joint-space move)."""
    run(_robot.arm(name).move_to_joint_positions(JointPositions(values=list(values))))


def move_position(name: str, x: float, y: float, z: float,
                  o_x: float, o_y: float, o_z: float, theta: float) -> None:
    """Move an arm to a Cartesian pose via the motion planner."""
    run(_robot.arm(name).move_to_position(
        Pose(x=x, y=y, z=z, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)
    ))


def gripper_action(arm_name: str, action: str) -> None:
    """Grab or open the gripper on an arm."""
    grip_name = RIGS[arm_name][1]
    if action == "grab":
        run(_robot.gripper(grip_name).grab())
    elif action == "open":
        run(_robot.gripper(grip_name).open())
    else:
        raise ValueError(f"Unknown gripper action: {action}")


def camera_frame(cam_name: str) -> bytes:
    """Capture a JPEG color frame from a camera. Returns raw JPEG bytes."""
    images, _ = run(_robot.camera(cam_name).get_images())
    by_name = {i.name: i for i in images}
    image = next((im for im in images if "color" in (im.name or "").lower()
                  and ("jpeg" in (im.mime_type or "") or "png" in (im.mime_type or ""))), None)
    if image is None:
        raise RuntimeError(f"No color image from camera {cam_name}")
    return image.data


def list_arm_names() -> list[str]:
    """Return the arm resource names configured on the machine."""
    robot = run(_robot.connect())
    return [r for r in robot.resource_names(subtype="arm")]


def list_all_resource_names() -> dict[str, list[str]]:
    robot = run(_robot.connect())
    return {
        "arm": list(robot.resource_names(subtype="arm")),
        "camera": list(robot.resource_names(subtype="camera")),
        "gripper": list(robot.resource_names(subtype="gripper")),
    }
