"""Claude's READ-ONLY bottle probe page: YOLO + depth -> world positions. This page never moves an arm."""
import asyncio, json, os, sys, threading
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "_claude")):
    if p not in sys.path:
        sys.path.insert(0, p)
from viam.robot.client import RobotClient
import probe_lib


@st.cache_resource
def probe_robot():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    def run(c):
        return asyncio.run_coroutine_threadsafe(c, loop).result(timeout=120)
    robot = run(RobotClient.at_address(os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud"),
                RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])))
    return run, robot


st.title("Bottle probe (read-only)")
st.caption("YOLO + aligned depth from cam on arm, converted to world mm. Never moves anything.")
if st.button("Run probe", key="run_probe"):
    run, robot = probe_robot()
    with st.spinner("Probing…"):
        try:
            out = probe_lib.probe(run, robot)
            st.success(f"Probe done at {out['time']} — wrote _claude/probe.json")
            st.image(str(ROOT / "_claude" / "probe_annotated.jpg"))
        except Exception as e:
            st.error(f"Probe failed: {e!r}")
