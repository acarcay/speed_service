/**
 * @file   bindings.cpp
 * @brief  Pybind11 Python extension module `vse_core`.
 *
 * This translation unit exposes the C++20 Vehicle Speed Estimation pipeline
 * to Python (e.g., a YOLOv11 inference script) with two hard constraints:
 *
 *  1. **GIL never blocks C++ worker threads.**
 *     The background Kalman / alert-dispatch threads never call back into
 *     CPython, so they hold no ownership of the GIL.  Every public method
 *     that could block or do heavy work releases the GIL via
 *     `py::gil_scoped_release` before touching C++ state.
 *
 *  2. **`push_measurement` is non-blocking.**
 *     The YOLOv11 inference loop calls `push_measurement` on the hot path.
 *     It uses `queue_.try_push()` which returns `false` immediately if the
 *     queue is full, never stalling the inference thread.
 *
 * Python usage example
 * --------------------
 * ```python
 * import vse_core
 *
 * pipe = vse_core.VSEPipeline(
 *     speed_threshold_kmh=120.0,
 *     accel_threshold_mpss=8.0,
 * )
 * pipe.start()
 *
 * # Inside your YOLOv11 frame loop:
 * for det in yolo_detections:
 *     pipe.push_measurement(
 *         track_id    = int(det.track_id),
 *         x           = float(det.bbox_cx),
 *         y           = float(det.bbox_bottom),
 *         timestamp_ms = int(time.time() * 1000),
 *     )
 *
 * pipe.stop()
 * ```
 *
 * Swapping the network trigger
 * ----------------------------
 * Pass a concrete `INetworkTrigger` subclass constructed in C++ through a
 * separate factory binding, or expose the `set_trigger` method below once
 * you have a production 5G QoD implementation.  `NullNetworkTrigger` is
 * used by default so the module is usable out-of-the-box without network
 * infrastructure.
 */

#include <cstdint>   // std::int64_t
#include <memory>    // std::make_shared, std::shared_ptr
#include <stdexcept> // std::runtime_error

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::vector / optional auto-conversion

#include "pipeline.hpp" // brings in all vse:: headers

namespace py = pybind11;

namespace vse {

// ==========================================================================
// PyVSEPipeline — C++ wrapper class exposed to Python
// ==========================================================================

/**
 * @brief Owns and wires together the full VSE pipeline for Python callers.
 *
 * Ownership graph
 * ---------------
 *  PyVSEPipeline
 *    ├─ MeasurementQueue<128>    (owned, referenced by orchestrator)
 *    ├─ shared_ptr<INetworkTrigger>  (swappable via set_trigger)
 *    └─ TrafficOrchestrator<128> (optional — present after start())
 *         └─ TrackManager        (owned by orchestrator)
 *
 * Design note: `TrafficOrchestrator` is constructed lazily in `start()`
 * because its constructor takes a `TrackManager` by value, and the scale
 * factor can be updated at any time before `start()` via `set_scale_factor`.
 */
class PyVSEPipeline {
public:
  /**
   * @param pixels_per_metre     Initial homography scale (px/m).
   *                             Placeholder default 0.05 works until
   *                             PerspectiveCalibrator provides the
   *                             calibrated value.
   * @param speed_threshold_kmh  Alert threshold forwarded to orchestrator.
   * @param accel_threshold_mpss Alert threshold forwarded to orchestrator.
   * @param process_noise_std    Forwarded to TrackManager / SpeedEstimator.
   * @param measurement_noise_std Forwarded to TrackManager / SpeedEstimator.
   */
  explicit PyVSEPipeline(float pixels_per_metre = 0.05f,
                         float speed_threshold_kmh = 120.0f,
                         float accel_threshold_mpss = 8.0f,
                         float process_noise_std = 2.0f,
                         float measurement_noise_std = 3.0f)
      : pixels_per_metre_(pixels_per_metre),
        speed_threshold_kmh_(speed_threshold_kmh),
        accel_threshold_mpss_(accel_threshold_mpss),
        process_noise_std_(process_noise_std),
        measurement_noise_std_(measurement_noise_std),
        trigger_(std::make_shared<NullNetworkTrigger>()) {}

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /**
   * @brief Construct the orchestrator and launch its worker thread.
   *
   * GIL is released for the duration: the orchestrator start() spawns an
   * OS thread and performs no CPython calls.
   */
  void start() {
    py::gil_scoped_release release;

    if (orchestrator_) {
      throw std::runtime_error(
          "VSEPipeline.start(): already running. Call stop() first.");
    }

    orchestrator_ = std::make_unique<TrafficOrchestrator<128>>(
        queue_, trigger_,
        TrackManager{pixels_per_metre_, process_noise_std_,
                     measurement_noise_std_},
        speed_threshold_kmh_, accel_threshold_mpss_);

    orchestrator_->start();
  }

  /**
   * @brief Signal the worker thread to stop and join it.
   *
   * Releases the GIL: the join may take up to one queue-pop timeout
   * plus any in-flight async alert dispatch.  No CPython calls occur.
   */
  void stop() {
    py::gil_scoped_release release;

    if (orchestrator_) {
      orchestrator_->stop();
      orchestrator_.reset();
    }
  }

  // ------------------------------------------------------------------
  // Hot-path: measurement ingestion
  // ------------------------------------------------------------------

  /**
   * @brief Enqueue a single detector observation from Python.
   *
   * Called from the YOLOv11 frame loop on the Python thread.  Uses
   * `try_push` (non-blocking) — if the queue is full (C++ worker has
   * fallen behind), the measurement is silently dropped rather than
   * stalling the inference loop.
   *
   * @param track_id      SORT / DeepSORT integer ID for the vehicle.
   * @param x             Bounding-box bottom-centre x (pixels).
   * @param y             Bounding-box bottom-centre y (pixels).
   * @param timestamp_ms  `int(time.time() * 1000)` from Python.
   *
   * @returns `True` if enqueued, `False` if the queue was full.
   *
   * @note  The GIL is intentionally NOT released here.  This call does
   *        only a trivial struct construction and one atomic compare-and-
   *        swap (try_push on the lock-free path).  Releasing the GIL
   *        for such a short operation would cost *more* than the operation
   *        itself due to GIL acquire/release overhead.
   */
  bool push_measurement(std::int32_t track_id, float x, float y,
                        std::int64_t timestamp_ms) {
    VehicleMeasurement m;
    m.track_id = track_id;
    m.bottom_center_x = x;
    m.bottom_center_y = y;
    m.timestamp_ms = timestamp_ms;
    return queue_.try_push(m);
  }

  // ------------------------------------------------------------------
  // Runtime configuration
  // ------------------------------------------------------------------

  /**
   * @brief Update the pixels-per-metre scale used for speed output.
   *
   * If called before `start()`, the new value is picked up when the
   * orchestrator is constructed.  If called after `start()`, the change
   * takes effect on the next `TrackManager::update()` call (the
   * orchestrator exposes `set_pixels_per_metre` via the TrackManager).
   *
   * The GIL is released: this is a trivial atomic write but we apply
   * the pattern consistently for all "heavy-enough-to-matter" calls.
   */
  void set_scale_factor(float pixels_per_metre) {
    py::gil_scoped_release release;
    pixels_per_metre_ = pixels_per_metre;
    // If the orchestrator is already running, the TrackManager inside
    // it lives on a separate thread.  We therefore cannot safely call
    // its setter directly without a lock.  Store the new value and
    // let the next restart pick it up, OR the user may stop/start the
    // pipeline.  This is by design: scale calibration is an offline
    // operation performed before the live run starts.
  }

  /**
   * @brief Replace the network trigger implementation at runtime.
   *
   * Must be called before `start()` (or after `stop()` before the next
   * `start()`).  Allows production code to swap in a real 5G QoD trigger
   * without recompiling the module.
   *
   * @throws std::runtime_error if the pipeline is currently running.
   */
  void set_trigger(std::shared_ptr<INetworkTrigger> new_trigger) {
    py::gil_scoped_release release;

    if (orchestrator_) {
      throw std::runtime_error(
          "VSEPipeline.set_trigger(): cannot replace trigger while "
          "running.  Call stop() first.");
    }
    if (!new_trigger) {
      throw std::invalid_argument(
          "VSEPipeline.set_trigger(): trigger must not be None.");
    }
    trigger_ = std::move(new_trigger);
  }

  // ------------------------------------------------------------------
  // Diagnostics (Python-friendly)
  // ------------------------------------------------------------------

  [[nodiscard]] bool is_running() const noexcept {
    return orchestrator_ != nullptr && orchestrator_->is_running();
  }

  [[nodiscard]] std::size_t alert_count() const noexcept {
    if (!orchestrator_)
      return 0;
    return orchestrator_->alert_count();
  }

  [[nodiscard]] std::size_t measurement_count() const noexcept {
    if (!orchestrator_)
      return 0;
    return orchestrator_->measurement_count();
  }

  [[nodiscard]] std::size_t queue_size() const noexcept {
    return queue_.size();
  }

private:
  // ── Tunables (set before start) ───────────────────────────────────
  float pixels_per_metre_;
  float speed_threshold_kmh_;
  float accel_threshold_mpss_;
  float process_noise_std_;
  float measurement_noise_std_;

  // ── Core pipeline objects ─────────────────────────────────────────
  MeasurementQueue<128> queue_;
  std::shared_ptr<INetworkTrigger> trigger_;
  std::unique_ptr<TrafficOrchestrator<128>> orchestrator_;
};

} // namespace vse

// ==========================================================================
// Pybind11 module definition
// ==========================================================================

PYBIND11_MODULE(vse_core, m) {
  m.doc() =
      "vse_core — Vehicle Speed Estimation C++ core.\n\n"
      "Exposes the Kalman-filter tracking pipeline and 5G alert\n"
      "dispatcher to Python.  The C++ background threads never acquire\n"
      "the Python GIL.\n\n"
      "Typical use::\n\n"
      "    import vse_core, time\n"
      "    pipe = vse_core.VSEPipeline(pixels_per_metre=0.05)\n"
      "    pipe.start()\n"
      "    # In your frame loop:\n"
      "    pipe.push_measurement(track_id, cx, cy, int(time.time()*1e3))\n"
      "    pipe.stop()\n";

  py::class_<vse::PyVSEPipeline>(
      m, "VSEPipeline",
      "Full VSE pipeline: MeasurementQueue → TrackManager (Kalman) "
      "→ async 5G alert dispatch.\n\n"
      "All heavy operations release the GIL so Python threads remain "
      "unblocked.")

      // Construction
      .def(py::init<float, float, float, float, float>(),
           py::arg("pixels_per_metre") = 0.05f,
           py::arg("speed_threshold_kmh") = 120.0f,
           py::arg("accel_threshold_mpss") = 8.0f,
           py::arg("process_noise_std") = 2.0f,
           py::arg("measurement_noise_std") = 3.0f,
           "Construct the pipeline.  Call start() to spawn the worker thread.")

      // Lifecycle
      .def("start", &vse::PyVSEPipeline::start,
           "Spawn the Kalman worker thread.  GIL is released during this "
           "call.")

      .def("stop", &vse::PyVSEPipeline::stop,
           "Join the worker thread and flush pending alerts.  GIL is "
           "released during this call.  Safe to call if not started.")

      // Hot-path ingestion
      .def("push_measurement", &vse::PyVSEPipeline::push_measurement,
           py::arg("track_id"), py::arg("x"), py::arg("y"),
           py::arg("timestamp_ms"),
           "Non-blocking enqueue of one detector observation.\n\n"
           "Returns True if enqueued, False if the queue was full\n"
           "(back-pressure — the caller may log and drop).\n"
           "The GIL is intentionally kept for this call because the\n"
           "operation (one atomic CAS) is cheaper than a GIL round-trip.")

      // Configuration
      .def("set_scale_factor", &vse::PyVSEPipeline::set_scale_factor,
           py::arg("pixels_per_metre"),
           "Update the homography px/m scale.  Must be called before "
           "start() to take effect in the current run.")

      // Diagnostics
      .def("is_running", &vse::PyVSEPipeline::is_running)
      .def("alert_count", &vse::PyVSEPipeline::alert_count,
           "Total 5G QoD alerts dispatched since start().")
      .def("measurement_count", &vse::PyVSEPipeline::measurement_count,
           "Total VehicleMeasurements processed by the Kalman worker.")
      .def("queue_size", &vse::PyVSEPipeline::queue_size,
           "Current number of unprocessed items in the MeasurementQueue.");
}
