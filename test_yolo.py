"""
test_yolo.py — YOLOv8 + vse_core integration test (subprocess isolation)
=========================================================================
Runs vse_core in a separate process to avoid OpenCV dual-runtime segfault
(PyTorch bundles its own OpenCV/libtorch; loading both in one process
with -march=native causes a SIMD clash on Apple Silicon).

Architecture:
  ┌─────────────────────────┐      multiprocessing.Queue      ┌──────────────────────┐
  │  Main process           │ ──── measurements (track_id,   ─▶  Worker process      │
  │  YOLOv8 detection       │      x, y, ts_ms) ────────────  │  vse_core C++ pipeline│
  │  OpenCV display         │ ◀─── alerts ────────────────── │  Kalman filter        │
  └─────────────────────────┘                                  └──────────────────────┘

Usage:
    python3 test_yolo.py --video trafik.mp4 [--scale 40] [--speed 80]

Arguments:
    --video   Path to input video
    --scale   pixels_per_metre (default: calibration window)
    --speed   Alert threshold km/h (default 80)
    --model   YOLO model (default yolov8n.pt)
    --imgsz   YOLO inference resolution (default 640)
    --save    Save annotated output to file
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time
import collections
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

# ── ultralytics ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ERROR: pip3 install ultralytics")

# ---------------------------------------------------------------------------
# VSE Worker — runs in a completely separate process (no PyTorch)
# ---------------------------------------------------------------------------
def _vse_worker(build_dir: str,
                meas_q: "mp.Queue[tuple | None]",
                alert_q: "mp.Queue[int]",
                pending,
                scale: float,
                speed_thr: float,
                accel_thr: float):
    """
    Child process: owns vse_core exclusively.
    Reads (track_id, x, y, ts_ms) tuples from meas_q, pushes alerts to alert_q.
    Sentinel None in meas_q signals shutdown.
    """
    sys.path.insert(0, build_dir)
    import vse_core  # imported only in the child

    pipe = vse_core.VSEPipeline(
        pixels_per_metre     = scale,
        speed_threshold_kmh  = speed_thr,
        accel_threshold_mpss = accel_thr,
    )
    pipe.start()

    total_alerts = 0
    while True:
        item = meas_q.get()
        with pending.get_lock():
            pending.value = max(0, pending.value - 1)
        if item is None:                # shutdown sentinel
            break
        track_id, x, y, ts_ms = item
        pipe.push_measurement(track_id, x, y, ts_ms)

        new_alerts = pipe.alert_count()
        if new_alerts > total_alerts:
            alert_q.put(new_alerts)
            total_alerts = new_alerts

    pipe.stop()
    alert_q.put(-1)   # signal worker done


# ---------------------------------------------------------------------------
# Robust speed tracker: EMA smoothed, outlier-rejected, min-age gated
# ---------------------------------------------------------------------------
class SpeedTracker:
    """
    Per-track speed estimator:
      - EMA smoothing (alpha=0.25) to kill bbox jitter
      - Hard cap: raw single-frame speed > 200 km/h → keep previous smoothed value
      - Min 5 good samples before showing anything
      - Auto-reset if track absent > 1 s
    """
    MAX_RAW  = 200.0   # km/h hard cap for single-frame readings
    ALPHA    = 0.25    # EMA weight (lower = smoother)
    MIN_AGE  = 5       # frames needed before display
    RESET_MS = 1000    # ms gap that triggers history reset

    def __init__(self, scale_ppm: float):
        self._ppm    = scale_ppm
        self._prev:   dict[int, tuple] = {}   # tid → (cx, cy, ts_ms)
        self._smooth: dict[int, float] = {}   # EMA speed
        self._age:    dict[int, int]   = {}   # good-sample counter

    def update(self, tid: int, cx: float, cy: float, ts_ms: int) -> float | None:
        prev = self._prev.get(tid)
        self._prev[tid] = (cx, cy, ts_ms)

        if prev is None:
            self._age[tid] = 0
            return None

        px, py, pt = prev
        dt = (ts_ms - pt) / 1000.0
        if dt <= 0 or dt > self.RESET_MS / 1000.0:
            self._age[tid] = 0
            self._smooth.pop(tid, None)
            return None

        raw = (np.hypot(cx-px, cy-py) / self._ppm) / dt * 3.6

        if raw > self.MAX_RAW:
            # Spike — teleporting detection; keep old smooth value, don't count
            return self._smooth.get(tid)

        prev_s = self._smooth.get(tid)
        smoothed = raw if prev_s is None else \
                   self.ALPHA * raw + (1 - self.ALPHA) * prev_s
        self._smooth[tid] = smoothed
        self._age[tid] = self._age.get(tid, 0) + 1

        return smoothed if self._age[tid] >= self.MIN_AGE else None

    def clear(self, tid: int):
        self._prev.pop(tid, None)
        self._smooth.pop(tid, None)
        self._age.pop(tid, None)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck


def draw_label(frame, text, x, y, alert=False):
    color = (0, 50, 255) if alert else (60, 220, 60)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    tw, th = cv2.getTextSize(text, font, fs, ft)[0]
    p = 4
    cv2.rectangle(frame, (x-p, y-th-p), (x+tw+p, y+p), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), font, fs, color, ft, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def calibrate_scale(frame):
    pts = []
    clone = frame.copy()
    cv2.putText(clone, "Click 2 points on a known object, then press ENTER",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
    cv2.imshow("Calibrate", clone)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 7, (0, 120, 255), -1)
            if len(pts) == 2:
                cv2.line(clone, pts[0], pts[1], (0, 120, 255), 2)
            cv2.imshow("Calibrate", clone)

    cv2.setMouseCallback("Calibrate", on_mouse)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 13 and len(pts) == 2:
            break
        if k == 27:
            cv2.destroyWindow("Calibrate")
            return float(input("pixels_per_metre > "))
    cv2.destroyWindow("Calibrate")
    px = np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
    m  = float(input(f"Line = {px:.1f} px. Real length in metres? > "))
    print(f"  scale = {px:.1f} / {m} = {px/m:.2f} px/m")
    return px / m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",     required=True)
    ap.add_argument("--model",     default="yolov8n.pt")
    ap.add_argument("--scale",     type=float, default=None)
    ap.add_argument("--vanish-y",  type=float, default=0.42,
                    help="Vanishing-point Y as fraction of frame height (default 0.42). "
                         "Look for where lanes converge in the video.")
    ap.add_argument("--speed",     type=float, default=80.0)
    ap.add_argument("--accel",     type=float, default=6.0)
    ap.add_argument("--conf",      type=float, default=0.35)
    ap.add_argument("--imgsz",     type=int,   default=640)
    ap.add_argument("--tracker",   default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--save",      default=None)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    # ── Find build dir ───────────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    build_dir  = str(script_dir / "build")
    if not list(Path(build_dir).glob("vse_core*.so")):
        sys.exit("vse_core.so not found in build/. Run: cmake --build build")

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VSE] {W}x{H} @ {fps:.1f}fps  ({total} frames)")

    # ── Calibration ──────────────────────────────────────────────────────────
    if args.scale is None:
        ret, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        scale = calibrate_scale(cv2.resize(first, (min(W,1920), min(H,1080))))
    else:
        scale = args.scale
    print(f"[VSE] scale={scale:.2f} px/m  speed_thr={args.speed} km/h")



    # ── Load YOLO ────────────────────────────────────────────────────────────
    print(f"[VSE] Loading {args.model}…")
    model = YOLO(args.model)
    # Warmup — initialise PyTorch runtime fully before spawning C++ thread
    print("[VSE] Warming up YOLO…")
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[VSE] Warmup done.")

    # ── Start VSE worker process ──────────────────────────────────────────────
    meas_q  = mp.Queue(maxsize=512)
    alert_q = mp.Queue()
    pending = mp.Value('i', 0)   # shared counter (macOS qsize workaround)
    worker  = mp.Process(target=_vse_worker, daemon=True,
                         args=(build_dir, meas_q, alert_q, pending,
                               scale, args.speed, args.accel))
    worker.start()
    print("[VSE] C++ pipeline worker started (PID", worker.pid, ")")

    # ── Video writer ─────────────────────────────────────────────────────────
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # ── State ────────────────────────────────────────────────────────────────
    speed_tracker = SpeedTracker(scale)
    track_speeds: dict[int, float] = {}
    alert_ids:    set[int]         = set()
    total_alerts = 0
    frame_idx    = 0
    t0           = time.perf_counter()

    # ── Infer scale factor within display window ──────────────────────────────
    infer_ratio = min(1.0, args.imgsz / max(W, H))
    iW, iH      = int(W * infer_ratio), int(H * infer_ratio)
    sx, sy      = W / iW, H / iH
    disp_ratio  = min(1.0, 1920 / max(W, H))
    dW, dH      = int(W * disp_ratio), int(H * disp_ratio)

    # ── Frame loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts_ms = int(frame_idx / fps * 1000)

        # Drain alert queue (non-blocking)
        while not alert_q.empty():
            v = alert_q.get_nowait()
            if v > 0:
                if v > total_alerts:
                    print(f"[ALERT] t={ts_ms/1000:.2f}s  total={v}")
                total_alerts = max(total_alerts, v)

        # YOLO inference on downscaled frame
        infer = cv2.resize(frame, (iW, iH)) if infer_ratio < 1.0 else frame
        results = model.track(infer, persist=True, conf=args.conf,
                              iou=0.45, classes=VEHICLE_CLASSES,
                              tracker=args.tracker, verbose=False)

        vis        = frame.copy()
        active_ids: set[int] = set()

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id is None or boxes.id[i] is None:
                    continue
                tid = int(boxes.id[i])
                active_ids.add(tid)

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, x2 = x1*sx, x2*sx
                y1, y2 = y1*sy, y2*sy
                cx      = (x1+x2)/2.0
                cy_bot  = float(y2)

                # ── Send raw pixel coords to C++ worker (non-blocking) ─────
                # C++ Kalman tracks in pixel space; scale_ref handles metres.
                try:
                    meas_q.put_nowait((tid, cx, cy_bot, ts_ms))
                    with pending.get_lock():
                        pending.value += 1
                except Exception:
                    pass  # queue full — drop

                # Perspective-corrected display speed (mid-y ppm)
                spd = speed_tracker.update(tid, cx, cy_bot, ts_ms)
                if spd is not None:
                    track_speeds[tid] = spd

                d     = track_speeds.get(tid)
                alrt  = d is not None and d > args.speed
                if alrt:
                    alert_ids.add(tid)

                col = (0, 50, 255) if alrt else (0, 200, 100)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              col, 3 if alrt else 2)
                lbl = f"ID:{tid}  {d:.0f} km/h{'  ⚠' if alrt else ''}" \
                      if d else f"ID:{tid}  …"
                draw_label(vis, lbl, int(x1), max(int(y1)-6, 20), alrt)
                cv2.circle(vis, (int(cx), int(cy_bot)), 4, (255, 200, 0), -1)

        # Cleanup lost tracks
        for tid in list(track_speeds):
            if tid not in active_ids:
                speed_tracker.clear(tid)
                track_speeds.pop(tid, None)



        # Console speed report every 30 frames (≈1 s at 30fps)
        if frame_idx % 30 == 0 and track_speeds:
            spd_str = "  ".join(
                f"ID{tid}:{spd:.0f}km/h"
                for tid, spd in sorted(track_speeds.items())
            )
            print(f"[t={ts_ms/1000:.1f}s] {spd_str}")

        # HUD
        proc_fps = frame_idx / (time.perf_counter() - t0)
        for i, txt in enumerate([
            f"Frame {frame_idx}/{total}  {proc_fps:.1f}fps",
            f"Tracks: {len(active_ids)}   Alerts: {total_alerts}",
            f"Queue: {pending.value}",
            f"Scale: {scale:.1f} px/m",
        ]):
            cv2.putText(vis, txt, (8, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1, cv2.LINE_AA)

        if writer:
            writer.write(vis)
        if not args.no_window:
            disp = cv2.resize(vis, (dW, dH)) if disp_ratio < 1.0 else vis
            cv2.imshow("VSE • YOLOv8 Pipeline", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("[VSE] Q — stopping.")
                break
            elif k == ord("s"):
                cv2.imwrite(f"snap_{frame_idx:06d}.jpg", vis)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    meas_q.put(None)     # shutdown sentinel
    worker.join(timeout=5)
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*48}")
    print(f"  Frames     : {frame_idx}")
    print(f"  Alerts     : {total_alerts}")
    print(f"{'='*48}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # macOS: spawn avoids fork+OpenCV crash
    main()
