"""Render the dashboard headlessly for arm and arm2 against the live machine and fail on any error.

Read-only: it connects, renders every section and switches the active arm. It never presses a button, so no arm
moves and no file is written. Run it through scripts/validate.sh --live (needs env.sh sourced).
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def check(at, arm):
    problems = [f"exception: {e.value}" for e in at.exception] + [f"error: {e.value}" for e in at.error]
    print(f"[{arm}] sections: {', '.join(h.value for h in at.subheader)}")
    print(f"[{arm}] {len(at.button)} buttons, {len(at.slider)} sliders, {len(at.number_input)} number inputs")
    for p in problems:
        print(f"[{arm}] {p[:300]}")
    return not problems


def main():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    ok = check(at, "arm")
    at.sidebar.selectbox[0].select("arm2").run()
    ok = check(at, "arm2") and ok
    print("smoke test passed" if ok else "smoke test FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
