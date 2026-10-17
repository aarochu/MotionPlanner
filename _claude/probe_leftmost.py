"""READ-ONLY CLI: source env.sh && .venv/bin/python _claude/probe_leftmost.py  (never moves anything)."""
import asyncio, os, sys, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "_claude")]; os.chdir(ROOT)
from viam.robot.client import RobotClient
from probe_lib import probe

loop = asyncio.new_event_loop(); threading.Thread(target=loop.run_forever, daemon=True).start()
def run(c): return asyncio.run_coroutine_threadsafe(c, loop).result(timeout=120)
robot = run(RobotClient.at_address(os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud"),
            RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])))
try:
    probe(run, robot); print("OK wrote _claude/probe.json")
finally:
    run(robot.close())
