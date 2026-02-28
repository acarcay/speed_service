/**
 * @file   traffic_orchestrator.hpp
 * @brief  Top-level integration class: consumes the MeasurementQueue,
 *         drives the Kalman TrackManager, and dispatches 5G QoD alerts
 *         without ever blocking the tracking thread.
 *
 * Architecture
 * ------------
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  GPU / YOLO thread                                              │
 *   │      VehicleMeasurement → MeasurementQueue::push()             │
 *   └────────────────────────────┬────────────────────────────────────┘
 *                                │ (blocking MPSC queue)
 *   ┌────────────────────────────▼────────────────────────────────────┐
 *   │  TrafficOrchestrator — worker_thread_                           │
 *   │                                                                 │
 *   │    MeasurementQueue::pop()  ←  blocks until item arrives        │
 *   │           │                                                     │
 *   │    TrackManager::update()   ←  predict + correct (Kalman)      │
 *   │           │                                                     │
 *   │    per-track hysteresis check:                                  │
 *   │      • speed_kmh  > SPEED_ALERT_THRESHOLD         ─────────┐   │
 *   │      • |dv/dt|    > ACCEL_ALERT_THRESHOLD (erratic braking) ┘   │
 *   │           │                                                     │
 *   │    std::async(launch::async, trigger->onHighRiskDetected(...))  │
 *   │      fires on a *separate OS thread* — tracking loop           │
 *   │      continues immediately, never waits for the REST call      │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Hysteresis / alert policy
 * -------------------------
 * Firing an alert on every frame for a speeding vehicle would flood the
 * 5G QoD API.  Two mechanisms prevent this:
 *
 *  1. Per-track cooldown (`kAlertCooldownMs` = 2 000 ms): an alert for
 *     a given track_id is not repeated more frequently than once per
 *     cooldown window, regardless of the trigger condition.
 *
 *  2. Dual trigger conditions (either suffices):
 *     a. Absolute speed:    speed_kmh > speed_threshold_kmh
 *     b. Erratic dynamics:  |Δspeed_mps / Δt_s| > accel_threshold_mpss
 *        (catches sudden braking even if instantaneous speed is low)
 *
 * Async dispatch & future management
 * -----------------------------------
 * `std::async(std::launch::async, ...)` returns a `std::future<void>`.
 * Discarding a future immediately would block on its destructor (the
 * standard mandates this for `std::launch::async` futures).  To avoid
 * this, futures are stored in `pending_futures_` and reaped lazily every
 * tracking cycle — consuming only futures that are already ready.  The
 * final batch is awaited in the destructor.
 *
 * Dependencies
 * ------------
 * C++17 / C++20.  No OpenCV.  No GUI.  Thread-safe externally: the
 * public API (`start`, `stop`, destructor) must not be called concurrently.
 *
 * @tparam QueueCapacity  Must match the `MeasurementQueue` instantiation
 *                        used by the producer (default 128).
 */

#pragma once

#include <atomic>        // std::atomic<bool>
#include <chrono>        // (not used directly; timestamps from types.hpp)
#include <cmath>         // std::abs, std::hypot
#include <cstdint>       // std::int32_t, std::int64_t
#include <future>        // std::async, std::future, std::launch
#include <memory>        // std::shared_ptr
#include <mutex>         // std::mutex, std::lock_guard
#include <stdexcept>     // std::runtime_error
#include <thread>        // std::thread
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include "measurement_queue.hpp" // vse::MeasurementQueue
#include "network_trigger.hpp"   // vse::INetworkTrigger
#include "speed_estimator.hpp"   // vse::TrackManager, vse::SpeedEstimate

namespace vse {

// ===========================================================================
// TrafficOrchestrator
// ===========================================================================

/**
 * @brief Ties the queue, Kalman tracker, and 5G alert dispatcher together.
 *
 * Lifetime contract
 * -----------------
 * - Construct → `start()` → (pipeline runs) → `stop()` / destructor.
 * - `stop()` is idempotent and safe to call before `start()`.
 * - The destructor calls `stop()` automatically, so RAII usage is safe.
 *
 * @tparam QueueCapacity  Capacity of the `MeasurementQueue` to consume.
 */
template <std::size_t QueueCapacity = 128> class TrafficOrchestrator final {
public:
  // ------------------------------------------------------------------
  // Tunable thresholds
  // ------------------------------------------------------------------

  /// Default speed above which an alert fires (km/h).
  static constexpr float kDefaultSpeedThresholdKmh = 120.0f;

  /**
   * @brief Default acceleration magnitude above which an alert fires
   *        (m/s²).  8 m/s² ≈ 0.8 g — covers hard braking events but
   *        ignores normal urban deceleration.
   */
  static constexpr float kDefaultAccelThresholdMpss = 8.0f;

  /// Minimum interval between two alerts for the same track_id (ms).
  static constexpr std::int64_t kAlertCooldownMs = 2000LL;

  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  /**
   * @param queue              Reference to the shared measurement queue.
   *                           Must outlive this object.
   * @param trigger            5G QoD alert sink.  May be `NullNetworkTrigger`
   *                           for dry-run deployments or unit tests.
   * @param track_manager      Pre-configured `TrackManager` (with
   *                           `pixels_per_metre` already set if available).
   * @param speed_threshold_kmh  Alert if instantaneous speed exceeds this.
   * @param accel_threshold_mpss Alert if |dv/dt| exceeds this (m/s²).
   */
  explicit TrafficOrchestrator(
      MeasurementQueue<QueueCapacity> &queue,
      std::shared_ptr<INetworkTrigger> trigger, TrackManager track_manager,
      float speed_threshold_kmh = kDefaultSpeedThresholdKmh,
      float accel_threshold_mpss = kDefaultAccelThresholdMpss)
      : queue_(queue), trigger_(std::move(trigger)),
        track_manager_(std::move(track_manager)),
        speed_threshold_kmh_(speed_threshold_kmh),
        accel_threshold_mpss_(accel_threshold_mpss) {
    if (!trigger_) {
      throw std::invalid_argument(
          "TrafficOrchestrator: trigger must not be null.  "
          "Pass NullNetworkTrigger for no-op behaviour.");
    }
  }

  // Non-copyable, non-movable — owns a std::thread.
  TrafficOrchestrator(const TrafficOrchestrator &) = delete;
  TrafficOrchestrator &operator=(const TrafficOrchestrator &) = delete;
  TrafficOrchestrator(TrafficOrchestrator &&) = delete;
  TrafficOrchestrator &operator=(TrafficOrchestrator &&) = delete;

  /**
   * @brief Stop the worker thread (if running) and await all pending
   *        async alert dispatches before destroying state.
   */
  ~TrafficOrchestrator() { stop(); }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /**
   * @brief Spawn the background worker thread and begin processing.
   *
   * @throws std::runtime_error if already running.
   */
  void start() {
    if (running_.exchange(true)) {
      throw std::runtime_error("TrafficOrchestrator::start: already running.");
    }
    worker_thread_ = std::thread(&TrafficOrchestrator::worker_loop, this);
  }

  /**
   * @brief Signal the worker to exit and join it.
   *
   * Internally calls `queue_.shutdown()` so that any blocked `pop()`
   * returns `std::nullopt` and the loop terminates cleanly.
   *
   * Idempotent — safe to call multiple times or before `start()`.
   */
  void stop() noexcept {
    if (!running_.exchange(false))
      return;

    queue_.shutdown();

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }

    // Drain all pending async dispatch futures (may briefly block for
    // the last network call if it hasn't finished yet, but this only
    // happens at pipeline shutdown — never on the hot path).
    drain_pending_futures(/*wait_all=*/true);
  }

  // ------------------------------------------------------------------
  // Diagnostics
  // ------------------------------------------------------------------

  /// Number of alerts dispatched since construction.
  [[nodiscard]] std::size_t alert_count() const noexcept {
    return alert_count_.load(std::memory_order_relaxed);
  }

  /// Number of measurements processed since construction.
  [[nodiscard]] std::size_t measurement_count() const noexcept {
    return measurement_count_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool is_running() const noexcept {
    return running_.load(std::memory_order_relaxed);
  }

private:
  // ==================================================================
  // Per-track hysteresis state
  // ==================================================================

  /**
   * @brief Bookkeeping held for every active track across update cycles.
   *
   * Stores enough information to compute:
   *  - Instantaneous acceleration (Δspeed / Δt), and
   *  - Whether the cooldown window has elapsed since the last alert.
   */
  struct TrackHysteresis {
    float last_speed_kmh{0.0f};

    /// Timestamp of the `SpeedEstimate` that populated `last_speed_kmh`.
    std::int64_t last_estimate_ms{0};

    /// Timestamp of the most recent alert dispatched for this track.
    /// Initialised well in the past so the first alert fires immediately.
    std::int64_t last_alert_ms{-kAlertCooldownMs * 4};
  };

  // ==================================================================
  // Worker thread
  // ==================================================================

  /**
   * @brief Main loop of the background Kalman consumer thread.
   *
   * Runs until `queue_` is shut down (returns `std::nullopt`) or
   * `running_` is explicitly cleared by `stop()`.
   */
  void worker_loop() {
    while (running_.load(std::memory_order_relaxed)) {

      // ── 1. Pull one measurement (blocks until available) ───────
      auto opt = queue_.pop();
      if (!opt.has_value())
        break; // queue shut down

      const VehicleMeasurement &m = *opt;
      measurement_count_.fetch_add(1, std::memory_order_relaxed);

      // ── 2. Feed to TrackManager (predict + correct) ────────────
      //
      // TrackManager::update expects a vector; wrap the single item.
      // The `now_ms` timestamp for coasting comes from the measurement
      // itself — this keeps the clock source consistent with the GPU
      // inference thread that stamped the measurement.
      //
      const std::vector<VehicleMeasurement> batch{m};
      const std::vector<SpeedEstimate> estimates =
          track_manager_.update(batch, m.timestamp_ms);

      // ── 3. Hysteresis evaluation & alert dispatch ──────────────
      process_estimates(estimates);

      // ── 4. Lazily reap finished async futures (non-blocking) ───
      drain_pending_futures(/*wait_all=*/false);
    }
  }

  // ==================================================================
  // Alert evaluation
  // ==================================================================

  /**
   * @brief Evaluate hysteresis conditions for every estimate and
   *        dispatch alerts where warranted.
   */
  void process_estimates(const std::vector<SpeedEstimate> &estimates) {
    for (const auto &est : estimates) {
      if (est.track_id < 0)
        continue;
      maybe_fire_alert(est);
    }
  }

  /**
   * @brief Check whether `est` crosses either alert threshold and, if so,
   *        dispatch `trigger_->onHighRiskDetected` on a fresh async thread.
   *
   * Two conditions (either is sufficient):
   *   a. speed_kmh > speed_threshold_kmh_
   *   b. |Δspeed_mps / Δt_s| > accel_threshold_mpss_
   *
   * The cooldown guard prevents re-alerting within `kAlertCooldownMs`.
   */
  void maybe_fire_alert(const SpeedEstimate &est) {
    auto &hyst = hysteresis_[est.track_id]; // default-constructs if new

    bool should_alert = false;
    std::string reason; // diagnostic only (not sent over the wire)

    // ── Condition A: absolute speed ────────────────────────────────
    if (est.speed_kmh > speed_threshold_kmh_) {
      should_alert = true;
    }

    // ── Condition B: erratic acceleration ─────────────────────────
    if (!should_alert && hyst.last_estimate_ms > 0) {
      const float dt_s =
          static_cast<float>(est.timestamp_ms - hyst.last_estimate_ms) *
          1.0e-3f;

      if (dt_s > 0.0f) {
        const float delta_speed_mps =
            (est.speed_kmh - hyst.last_speed_kmh) / 3.6f;
        const float accel_mpss = std::abs(delta_speed_mps) / dt_s;

        if (accel_mpss > accel_threshold_mpss_) {
          should_alert = true;
        }
      }
    }

    // ── Cooldown guard ─────────────────────────────────────────────
    if (should_alert) {
      const std::int64_t since_last_alert =
          est.timestamp_ms - hyst.last_alert_ms;

      if (since_last_alert < kAlertCooldownMs) {
        should_alert = false; // within cooldown window — suppress
      }
    }

    // ── Async dispatch ─────────────────────────────────────────────
    if (should_alert) {
      hyst.last_alert_ms = est.timestamp_ms;

      // Capture by value so the lambda is self-contained.  The trigger
      // shared_ptr keeps the implementation alive for the duration of
      // the async call even if the orchestrator is destroyed.
      const std::int32_t tid = est.track_id;
      const float speed = est.speed_kmh;
      auto trigger = trigger_; // shared_ptr copy

      {
        std::lock_guard lock{futures_mutex_};
        pending_futures_.push_back(
            std::async(std::launch::async, [trigger, tid, speed]() noexcept {
              trigger->onHighRiskDetected(tid, speed);
            }));
      }
      alert_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Update hysteresis state ────────────────────────────────────
    hyst.last_speed_kmh = est.speed_kmh;
    hyst.last_estimate_ms = est.timestamp_ms;
  }

  // ==================================================================
  // Future management
  // ==================================================================

  /**
   * @brief Reap completed async futures from `pending_futures_`.
   *
   * @param wait_all  If true, wait for *all* pending futures to complete
   *                  (used at shutdown).  If false, only discard futures
   *                  that are already ready (non-blocking hot-path).
   */
  void drain_pending_futures(bool wait_all) noexcept {
    std::lock_guard lock{futures_mutex_};

    if (wait_all) {
      // At shutdown: block until every dispatched call has returned.
      for (auto &f : pending_futures_) {
        if (f.valid())
          f.wait();
      }
      pending_futures_.clear();
    } else {
      // Hot path: erase futures that finished without blocking.
      pending_futures_.erase(
          std::remove_if(pending_futures_.begin(), pending_futures_.end(),
                         [](std::future<void> &f) {
                           return !f.valid() ||
                                  f.wait_for(std::chrono::seconds(0)) ==
                                      std::future_status::ready;
                         }),
          pending_futures_.end());
    }
  }

  // ==================================================================
  // Members
  // ==================================================================

  // ── Core pipeline components ───────────────────────────────────────
  MeasurementQueue<QueueCapacity> &queue_;
  std::shared_ptr<INetworkTrigger> trigger_;
  TrackManager track_manager_;

  // ── Alert thresholds ───────────────────────────────────────────────
  float speed_threshold_kmh_;
  float accel_threshold_mpss_;

  // ── Per-track hysteresis bookkeeping ──────────────────────────────
  std::unordered_map<std::int32_t, TrackHysteresis> hysteresis_;

  // ── Background worker ─────────────────────────────────────────────
  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  // ── Async alert futures ───────────────────────────────────────────
  std::mutex futures_mutex_;
  std::vector<std::future<void>> pending_futures_;

  // ── Metrics ───────────────────────────────────────────────────────
  std::atomic<std::size_t> alert_count_{0};
  std::atomic<std::size_t> measurement_count_{0};
};

} // namespace vse
