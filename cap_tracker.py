"""Slow closed-loop tracking: keep the gripper over the bottle cap as the bottle moves.

Runs in a background thread so it survives Streamlit reruns. Each cycle grabs a cam2 frame, finds the cap
(cap_vision), and nudges the arm a small step toward it in XY at the current height and orientation.

Safety:
- Steps are small and slow: a fraction (`gain`) of the error, capped at `max_step_mm`, at most every `period_s`.
- Within `deadband_mm` the arm holds still.
- If the cap is missing for `lost_after` frames in a row (bottle removed) the arm is stopped and the tracker
  holds; it resumes only after the cap is seen in `reacquire_after` frames in a row.
- A detection that jumps far from the last one is ignored rather than chased.
- The arm never goes more than `max_travel_mm` (XY) from where tracking started; at the limit it stops and holds.
- Any error stops the arm and turns tracking off.
"""
import threading
import time

import numpy as np

from cap_vision import detect_circles


class CapTracker:
    def __init__(self, vision, arm, *, period_s=0.5, gain=0.5, max_step_mm=10.0, deadband_mm=3.0,
                 lost_after=3, reacquire_after=3, max_travel_mm=150.0):
        self.vision, self.arm = vision, arm
        self.period_s, self.gain, self.max_step_mm, self.deadband_mm = period_s, gain, max_step_mm, deadband_mm
        self.lost_after, self.reacquire_after, self.max_travel_mm = lost_after, reacquire_after, max_travel_mm
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._settings = None
        self._status = {"state": "off"}

    # ---------- control ----------
    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, detect, cap_diameter_mm):
        """Begin tracking with the given detection settings (same tuple the dashboard uses)."""
        if self.running:
            self.update(detect, cap_diameter_mm)
            return
        self.update(detect, cap_diameter_mm)
        self._stop.clear()
        self._set(state="starting", error=None, moves=0, error_mm=None, cap_px=None)
        self._thread = threading.Thread(target=self._loop, name="cap-tracker", daemon=True)
        self._thread.start()

    def stop(self):
        """Stop tracking and stop the arm. Safe to call when not running."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._halt()
        if self.status()["state"] != "error":
            self._set(state="off")

    def update(self, detect, cap_diameter_mm):
        with self._lock:
            self._settings = (detect, cap_diameter_mm)

    def status(self):
        with self._lock:
            return dict(self._status)

    # ---------- internals ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _halt(self):
        try:
            self.vision.run(self.arm.stop())
        except Exception:
            pass

    def _loop(self):
        home = self.vision.run(self.arm.get_end_position())  # travel limit is measured from here
        misses, streak, jumps, lost, last = 0, 0, 0, False, None
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                with self._lock:
                    detect, cap_d = self._settings
                bgr, depth = self.vision.capture()
                caps, _ = detect_circles(bgr, *detect)

                if not caps:
                    misses, streak = misses + 1, 0
                    if misses >= self.lost_after and not lost:
                        self._halt()
                        lost, last = True, None
                    self._set(state="lost — holding" if lost else "searching", cap_px=None, error_mm=None)
                else:
                    cap = caps[0]
                    misses, streak = 0, streak + 1
                    jumped = last is not None and np.hypot(cap[0] - last[0], cap[1] - last[1]) > 0.5 * last[2]
                    jumps = jumps + 1 if jumped else 0
                    if lost and streak < self.reacquire_after:
                        self._set(state=f"reacquiring {streak}/{self.reacquire_after}", cap_px=cap[:3])
                    elif jumped and jumps < 2:
                        self._set(state="ignoring jump", cap_px=cap[:3])  # chase a jump only if the next frame agrees
                    else:
                        lost, last, jumps = False, cap, 0
                        self._step(home, self.vision.measure(cap, depth, cap_d), cap)
            except Exception as e:
                self._halt()
                self._set(state="error", error=str(e))
                return
            self._stop.wait(max(0.0, self.period_s - (time.monotonic() - t0)))

    def _step(self, home, point, cap):
        cap_b, grip_b = self.vision.cap_error(point)
        err = np.array([cap_b.x - grip_b.x, cap_b.y - grip_b.y])
        dist = float(np.hypot(*err))
        if dist <= self.deadband_mm:
            self._set(state="locked", cap_px=cap[:3], error_mm=dist)
            return
        step = self.gain * err
        n = float(np.hypot(*step))
        if n > self.max_step_mm:
            step *= self.max_step_mm / n
        target = self.vision.offset_target((step[0], step[1], 0.0))
        if np.hypot(target.x - home.x, target.y - home.y) > self.max_travel_mm:
            if not self.status()["state"].startswith("travel limit"):
                self._halt()
            self._set(state=f"travel limit ({self.max_travel_mm:.0f} mm) — holding", cap_px=cap[:3], error_mm=dist)
            return
        self.vision.run(self.arm.move_to_position(target))
        with self._lock:
            self._status.update(state="tracking", cap_px=cap[:3], error_mm=dist,
                                moves=self._status.get("moves", 0) + 1)
