"""READ-ONLY probe helpers (no motion). probe(run, robot) -> dict; also writes _claude/probe.json + images."""
import json, time
from pathlib import Path
import cv2, numpy as np
from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.media.video import ViamImage, CameraMimeType
import vision as V

OUT = Path(__file__).resolve().parent


def probe(run, robot, n_frames=5):
    if True:
        arm, cam = Arm.from_robot(robot, "arm"), Camera.from_robot(robot, "cam")
        det = VisionClient.from_robot(robot, "bottle-yolo")
        out = {"time": time.strftime("%H:%M:%S"),
               "joints": list(run(arm.get_joint_positions()).values)}
        e = run(arm.get_end_position())
        out["end"] = {k: getattr(e, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
        fs = run(robot.get_frame_system_config())
        out["frames"] = [MessageToDict(f.frame) for f in fs if f.frame.reference_frame in ("arm", "gripper", "cam")
                         or f.frame.pose_in_observer_frame.reference_frame in ("arm", "gripper")]
        def frame_axes(name):
            o = V.world_point(run, robot, name)
            ax = [(V.world_point(run, robot, name, *a) - o) / 100 for a in ((100, 0, 0), (0, 100, 0), (0, 0, 100))]
            return {"origin": o.tolist(), "x": ax[0].tolist(), "y": ax[1].tolist(), "z": ax[2].tolist()}
        out["gripper_world"] = frame_axes("gripper")
        out["arm_end_world"] = frame_axes("arm")
        frames = []
        for i in range(n_frames):
            cap = V.capture_frame(run, robot, cam, "cam", "arm")
            ok, enc = cv2.imencode(".jpg", cap["bgr"])
            dets = run(det.get_detections(ViamImage(enc.tobytes(), CameraMimeType.JPEG)))
            H, W = cap["bgr"].shape[:2]
            bottles = sorted([d for d in dets if d.class_name.lower() == "bottle" and d.confidence >= .4],
                             key=lambda d: d.x_min + d.x_max)
            rec = {"cam_origin": cap["origin"].tolist(), "cam_R": cap["R"].tolist(), "fx": cap["fx"], "fy": cap["fy"],
                   "cx": cap["cx"], "cy": cap["cy"], "W": W, "H": H, "depth_matches_color": cap["depth_matches_color"],
                   "others": [(d.class_name, round(d.confidence, 2)) for d in dets if d.class_name.lower() != "bottle"],
                   "bottles": []}
            D = cap["depth"]
            for d in bottles:
                u = (d.x_min + d.x_max) / 2
                b = {"conf": d.confidence, "box": [d.x_min, d.y_min, d.x_max, d.y_max], "rows": []}
                # depth along the bottle's center column, in 10 bands from top to bottom of the box
                for f in np.linspace(.05, .95, 10):
                    v = d.y_min + f * (d.y_max - d.y_min)
                    half = max(3, int(.15 * (d.x_max - d.x_min)))
                    roi = D[int(v - 4):int(v + 5), int(u - half):int(u + half + 1)] if D is not None else np.zeros(0)
                    vals = roi[(roi >= 150) & (roi <= 3000)].astype(float)
                    row = {"f": round(float(f), 2), "v": round(v, 1), "valid": round(vals.size / max(roi.size, 1), 2)}
                    if vals.size:
                        z = float(np.median(vals))
                        row.update(z=z, spread=float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                                   surface_world=V._surface_to_world(cap, {"px": (u, v)}, z, 0).tolist(),
                                   left_world=V._surface_to_world(cap, {"px": (d.x_min, v)}, z, 0).tolist(),
                                   right_world=V._surface_to_world(cap, {"px": (d.x_max, v)}, z, 0).tolist())
                    b["rows"].append(row)
                rec["bottles"].append(b)
            frames.append(rec)
            if i == 0:
                img = cap["bgr"].copy()
                for n, d in enumerate(bottles):
                    cv2.rectangle(img, (d.x_min, d.y_min), (d.x_max, d.y_max), (0, 255, 0) if n == 0 else (0, 200, 255), 2)
                    cv2.putText(img, f"{n}:{d.confidence:.2f}", (d.x_min, d.y_min + 25), 0, .8, (0, 255, 0), 2)
                cv2.drawMarker(img, (int(cap["cx"]), int(cap["cy"])), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.imwrite(str(OUT / "probe_annotated.jpg"), img)
                if D is not None:
                    s = np.clip((D.astype(np.int32) - 150) * 255 // 850, 0, 255).astype(np.uint8)
                    rgb = cv2.applyColorMap(255 - s, cv2.COLORMAP_JET); rgb[D == 0] = 0
                    cv2.imwrite(str(OUT / "probe_depth.jpg"), rgb)
                    np.save(OUT / "probe_depth.npy", D)
            time.sleep(.2)
        out["captures"] = frames
        out["joints_after"] = list(run(arm.get_joint_positions()).values)
        (OUT / "probe.json").write_text(json.dumps(out, indent=1, default=float))
        return out
