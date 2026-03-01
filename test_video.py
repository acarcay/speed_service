"""
test_video.py — VSE pipeline integration test (no YOLO required)
----------------------------------------------------------------
Uses OpenCV background subtraction + centroid tracking to feed
real bounding-box detections into the vse_core C++ pipeline.

Usage:
    python3 test_video.py --video traffic.mp4 [--scale 0.04] [--speed 80]

Arguments:
    --video   Path to any traffic/highway video file
    --scale   pixels_per_metre calibration (default 0.04 = ~25 m/px)
              Tune this: measure a known object length (lane ~3 m) in pixels
              then scale = pixels / metres
    --speed   Alert threshold in km/h (default 80)

Output:
    - Annotated video window showing tracked boxes + estimated speed/alerts
    - Console prints every alert fired by the C++ pipeline
"""

import argparse
import time
import threading
import cv2
import sys

try:
    import vse_core
except ImportError:
    sys.exit(
        "ERROR: vse_core not found.\n"
        "Run:  export PYTHONPATH=build:$PYTHONPATH\n"
        "then: python3 test_video.py --video <file>"
    )

# ---------------------------------------------------------------------------
# Minimal centroid tracker (no external deps)
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects: dict[int, tuple[float, float]] = {}  # id -> (cx, cy)
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def update(self, rects):
        """rects: list of (x, y, w, h)"""
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return self.objects

        input_centroids = [(x + w // 2, y + h // 2) for x, y, w, h in rects]

        if not self.objects:
            for cx, cy in input_centroids:
                self.objects[self.next_id] = (cx, cy)
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.objects

        # Match existing objects to closest new centroids
        import numpy as np
        obj_ids = list(self.objects.keys())
        obj_cents = list(self.objects.values())

        D = np.linalg.norm(
            np.array(obj_cents)[:, None] - np.array(input_centroids)[None, :],
            axis=2
        )
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > 80:  # max distance threshold
                continue
            oid = obj_ids[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]
                del self.disappeared[oid]

        for c in set(range(len(input_centroids))) - used_cols:
            cx, cy = input_centroids[c]
            self.objects[self.next_id] = (cx, cy)
            self.disappeared[self.next_id] = 0
            self.next_id += 1

        return self.objects


# ---------------------------------------------------------------------------
# Alert listener (runs in background thread — set by C++ pipeline)
# ---------------------------------------------------------------------------
alerts: list[str] = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True, help="Path to video file")
    ap.add_argument("--scale",  type=float, default=0.04,
                    help="pixels_per_metre (tune for your camera)")
    ap.add_argument("--speed",  type=float, default=80.0,
                    help="Speed alert threshold km/h")
    ap.add_argument("--accel",  type=float, default=6.0,
                    help="Accel alert threshold m/s²")
    args = ap.parse_args()

    pipe = vse_core.VSEPipeline(
        pixels_per_metre=args.scale,
        speed_threshold_kmh=args.speed,
        accel_threshold_mpss=args.accel,
    )
    pipe.start()
    print(f"[VSE] Pipeline started  scale={args.scale} px/m  "
          f"speed_thr={args.speed} km/h")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = CentroidTracker(max_disappeared=15)

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=40, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame_idx = 0
    last_alert_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        ts_ms = int((frame_idx / fps) * 1000)  # synthetic timestamp

        # --- Detection via background subtraction ---
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 800:   # skip tiny noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > frame.shape[1] * 0.8:     # skip near-full-width blobs
                continue
            rects.append((x, y, w, h))

        objects = tracker.update(rects)

        # --- Feed pipeline ---
        for track_id, (cx, cy) in objects.items():
            # Use bottom-centre of the bounding box approximation
            pipe.push_measurement(
                track_id=track_id,
                x=float(cx),
                y=float(cy),
                timestamp_ms=ts_ms,
            )

        # --- Draw ---
        vis = frame.copy()
        for track_id, (cx, cy) in objects.items():
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(vis, f"ID:{track_id}", (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show alert count
        ac = pipe.alert_count()
        mc = pipe.measurement_count()
        qs = pipe.queue_size()
        if ac > last_alert_count:
            print(f"[ALERT] t={ts_ms/1000:.2f}s  total_alerts={ac}")
            last_alert_count = ac

        cv2.putText(vis,
                    f"Alerts:{ac}  Measurements:{mc}  Queue:{qs}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("VSE Pipeline Test", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pipe.stop()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[VSE] Done. Alerts fired: {pipe.alert_count()}")


if __name__ == "__main__":
    main()
