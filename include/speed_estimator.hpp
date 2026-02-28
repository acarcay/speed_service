/**
 * @file   speed_estimator.hpp
 * @brief  Kalman-filter-based vehicle speed estimator.
 *
 * Design
 * ------
 * Two classes are provided:
 *
 *  · **`SpeedEstimator`** — owns a single `cv::KalmanFilter` instance and
 *    tracks one vehicle.  The state vector is 4-dimensional:
 *
 *        X = [x,  y,  v_x,  v_y]^T
 *
 *    and the measurement vector is 2-dimensional:
 *
 *        Z = [x,  y]^T
 *
 *    The transition matrix F is a standard constant-velocity model; it is
 *    updated *dynamically* before every prediction step to incorporate the
 *    real inter-frame time delta (dt).
 *
 *  · **`TrackManager`** — owns a map of `SpeedEstimator` instances keyed by
 *    `track_id`.  It implements the full predict / correct cycle, plus the
 *    identity-switch guard:
 *
 *      - Track missing < 1 500 ms  →  predict() only (coasted velocity).
 *      - Track missing ≥ 1 500 ms  →  destroy the `SpeedEstimator` to free
 *        memory.
 *
 * Units
 * -----
 * All pixel-space coordinates are in pixels.  Velocities produced by the
 * filter are in pixels/second (the caller supplies dt in *milliseconds*; the
 * class converts internally).  `SpeedEstimate::speed_mps` uses a caller-
 * supplied pixels-per-metre scale factor; if that factor is 0 the field is
 * left as 0.0f (the caller must supply homography-based scaling separately).
 *
 * Dependencies
 * ------------
 * Requires OpenCV (>= 4.x) `opencv_video` module for `cv::KalmanFilter`.
 * No GUI modules are needed.
 *
 * C++17 / C++20 compatible.  No GUI dependencies.  Thread-safety: a single
 * `TrackManager` instance must NOT be accessed concurrently from multiple
 * threads without external synchronisation.  Use one `TrackManager` per
 * consumer thread.
 */

#pragma once

#include <cmath>         // std::hypot
#include <cstdint>       // std::int32_t, std::int64_t
#include <unordered_map> // std::unordered_map  (TrackManager::tracks_)
#include <unordered_set> // std::unordered_set  (TrackManager::seen_this_tick_)
#include <vector>        // std::vector         (update return type)

#include <opencv2/video/tracking.hpp> // cv::KalmanFilter

#include "types.hpp" // vse::VehicleMeasurement

namespace vse {

// ===========================================================================
// SpeedEstimate  —  result type returned by the estimator
// ===========================================================================

/**
 * @brief Snapshot of one vehicle's speed estimate at a given instant.
 */
struct SpeedEstimate {
  /// Owning tracker ID (mirrors `VehicleMeasurement::track_id`).
  std::int32_t track_id{-1};

  /// Filtered x-position (pixels).
  float x{0.0f};
  /// Filtered y-position (pixels).
  float y{0.0f};

  /// Filtered x-velocity (pixels / second).
  float vx{0.0f};
  /// Filtered y-velocity (pixels / second).
  float vy{0.0f};

  /**
   * @brief Scalar speed in metres per second.
   *
   * Computed as `hypot(vx, vy) / pixels_per_metre`.
   * Zero if the caller has not provided a valid `pixels_per_metre` scale.
   */
  float speed_mps{0.0f};

  /// Convenience: `speed_mps * 3.6f`.
  float speed_kmh{0.0f};

  /**
   * @brief Source timestamp in milliseconds since UNIX epoch.
   *
   * Equal to the `timestamp_ms` of the last measurement that triggered
   * this estimate (predict-only coasted estimates carry the timestamp
   * of the current wall-clock tick supplied to `TrackManager::update`).
   */
  std::int64_t timestamp_ms{0};

  /**
   * @brief True when this estimate was produced by a predict-only step
   *        (i.e., no measurement was received for this track in the
   *        current update cycle).
   */
  bool is_coasted{false};
};

// ===========================================================================
// SpeedEstimator  —  wraps cv::KalmanFilter for one track
// ===========================================================================

/**
 * @brief Single-track Kalman filter for pixel-space speed estimation.
 *
 * State:       X = [x, y, v_x, v_y]^T   (4 × 1)
 * Measurement: Z = [x, y]^T              (2 × 1)
 *
 * The constant-velocity kinematic model assumes:
 *   x(k+1)   = x(k)   + v_x(k) * dt
 *   y(k+1)   = y(k)   + v_y(k) * dt
 *   v_x(k+1) = v_x(k)
 *   v_y(k+1) = v_y(k)
 *
 * Noise parameters (Q, R) are tunable via constructor arguments.
 */
class SpeedEstimator final {
public:
  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  /**
   * @brief Construct and initialise the filter from the first measurement.
   *
   * @param initial               First `VehicleMeasurement` for this track.
   *                              Used to seed the state estimate (velocity =
   * 0).
   * @param process_noise_std     Standard deviation for the process noise
   *                              model.  Smaller values → smoother velocity
   *                              at the cost of responsiveness.  A value of
   *                              ~2.0 works well for typical surveillance
   *                              frame rates (25–30 fps).
   * @param measurement_noise_std Standard deviation for detector
   *                              localisation noise (pixels).  ~3–5 px is
   *                              a reasonable starting point for a modern
   *                              YOLO detector.
   */
  explicit SpeedEstimator(const VehicleMeasurement &initial,
                          float process_noise_std = 2.0f,
                          float measurement_noise_std = 3.0f)
      : track_id_(initial.track_id), last_timestamp_ms_(initial.timestamp_ms),
        kf_(4 /*dynamParams*/, 2 /*measureParams*/, 0 /*controlParams*/,
            CV_32F) {
    //
    // ── Measurement Matrix H  (2×4, static) ───────────────────────────
    //
    //   H = | 1  0  0  0 |
    //       | 0  1  0  0 |
    //
    // We observe only position; velocity is a hidden (latent) state.
    //
    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);

    //
    // ── Process Noise Covariance Q  (4×4) ─────────────────────────────
    //
    // Diagonal "discrete white-noise" model: Q = q * I  where
    // q = process_noise_std^2.  Treats position and velocity noise as
    // independent — appropriate for the constant-velocity model.
    //
    const float q = process_noise_std * process_noise_std;
    kf_.processNoiseCov =
        (cv::Mat_<float>(4, 4) << q, 0.0f, 0.0f, 0.0f, 0.0f, q, 0.0f, 0.0f,
         0.0f, 0.0f, q, 0.0f, 0.0f, 0.0f, 0.0f, q);

    //
    // ── Measurement Noise Covariance R  (2×2) ─────────────────────────
    //
    //  R = r * I  where r = measurement_noise_std^2.
    //
    const float r = measurement_noise_std * measurement_noise_std;
    kf_.measurementNoiseCov = (cv::Mat_<float>(2, 2) << r, 0.0f, 0.0f, r);

    //
    // ── Error Covariance P (initial condition)  (4×4) ─────────────────
    //
    // High uncertainty on the velocity states because velocity cannot be
    // observed from a single position measurement.
    //
    kf_.errorCovPost =
        (cv::Mat_<float>(4, 4) << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f);

    //
    // ── Transition Matrix F  (4×4, placeholder) ───────────────────────
    //
    // Written as the identity here.  `update_transition_matrix(dt)` is
    // called immediately before every `kf_.predict()` so the dt columns
    // are always fresh.
    //
    kf_.transitionMatrix = cv::Mat::eye(4, 4, CV_32F);

    //
    // ── Seed the posterior state with the first observation ────────────
    //
    kf_.statePost.at<float>(0) = initial.bottom_center_x;
    kf_.statePost.at<float>(1) = initial.bottom_center_y;
    kf_.statePost.at<float>(2) = 0.0f; // v_x unknown at t = 0
    kf_.statePost.at<float>(3) = 0.0f; // v_y unknown at t = 0
  }

  // Non-copyable — cv::KalmanFilter owns large cv::Mat objects;
  // accidental copies would be expensive and semantically incorrect.
  SpeedEstimator(const SpeedEstimator &) = delete;
  SpeedEstimator &operator=(const SpeedEstimator &) = delete;

  // Movable so we can store instances in STL containers.
  SpeedEstimator(SpeedEstimator &&) = default;
  SpeedEstimator &operator=(SpeedEstimator &&) = default;

  ~SpeedEstimator() = default;

  // ------------------------------------------------------------------
  // Core Kalman operations
  // ------------------------------------------------------------------

  /**
   * @brief Advance the filter state forward in time by `dt_seconds`.
   *
   * The transition matrix F is updated *before* the prediction step so
   * that the off-diagonal dt cells always reflect the actual inter-frame
   * interval — not a fixed nominal rate.
   *
   * @param dt_seconds  Time elapsed since the previous measurement /
   *                    prediction step, in **seconds**.  Caller must
   *                    guarantee dt_seconds > 0.
   */
  void predict(float dt_seconds) noexcept {
    update_transition_matrix(dt_seconds);
    kf_.predict(); // kf_.statePre ← F · statePost
  }

  /**
   * @brief Incorporate a new position measurement into the filter.
   *
   * Should be called immediately after `predict()` in the same tick
   * when a detection is available.
   *
   * @param measurement  The observation from the YOLO / SORT pipeline.
   */
  void correct(const VehicleMeasurement &measurement) noexcept {
    cv::Mat z(2, 1, CV_32F);
    z.at<float>(0) = measurement.bottom_center_x;
    z.at<float>(1) = measurement.bottom_center_y;
    kf_.correct(z); // kf_.statePost ← updated posterior
    last_timestamp_ms_ = measurement.timestamp_ms;
  }

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  /// Returns the track ID this estimator was created for.
  [[nodiscard]] std::int32_t track_id() const noexcept { return track_id_; }

  /// Timestamp of the last measurement incorporated via `correct()`.
  [[nodiscard]] std::int64_t last_timestamp_ms() const noexcept {
    return last_timestamp_ms_;
  }

  /**
   * @brief Build a `SpeedEstimate` from the current posterior state.
   *
   * @param pixels_per_metre  Scale factor from homography calibration.
   *                          Pass 0.0f to skip metric conversion.
   * @param timestamp_ms      Wall-clock time to stamp the estimate.
   * @param is_coasted        True when no measurement was available for
   *                          this track in the current cycle.
   */
  [[nodiscard]] SpeedEstimate build_estimate(float pixels_per_metre,
                                             std::int64_t timestamp_ms,
                                             bool is_coasted) const noexcept {
    SpeedEstimate est;
    est.track_id = track_id_;
    // Read from the posterior state (statePost is updated by correct();
    // between ticks it holds the last corrected or predicted state).
    est.x = kf_.statePost.at<float>(0);
    est.y = kf_.statePost.at<float>(1);
    est.vx = kf_.statePost.at<float>(2); // px/s
    est.vy = kf_.statePost.at<float>(3); // px/s
    est.timestamp_ms = timestamp_ms;
    est.is_coasted = is_coasted;

    if (pixels_per_metre > 0.0f) {
      const float speed_px_s = std::hypot(est.vx, est.vy);
      est.speed_mps = speed_px_s / pixels_per_metre;
      est.speed_kmh = est.speed_mps * 3.6f;
    }
    return est;
  }

private:
  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  /**
   * @brief Write dt into the velocity-coupling cells of the transition
   *        matrix F.
   *
   *   F(dt) = | 1   0   dt   0  |
   *           | 0   1    0  dt  |
   *           | 0   0    1   0  |
   *           | 0   0    0   1  |
   *
   * Only two cells change between frames.  Direct element access is
   * cheaper than rebuilding the full 4×4 matrix each tick.
   */
  void update_transition_matrix(float dt_seconds) noexcept {
    kf_.transitionMatrix.at<float>(0, 2) = dt_seconds; // x  += vx * dt
    kf_.transitionMatrix.at<float>(1, 3) = dt_seconds; // y  += vy * dt
  }

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------

  std::int32_t track_id_;
  std::int64_t last_timestamp_ms_;
  cv::KalmanFilter kf_;
};

// ===========================================================================
// TrackManager  —  lifecycle manager for a fleet of SpeedEstimators
// ===========================================================================

/**
 * @brief Manages a dynamic collection of per-track `SpeedEstimator` objects.
 *
 * Call `update()` once per decoded video frame (pipeline tick):
 *
 *  1. For every track_id present in `measurements`:
 *        a. Creates a new `SpeedEstimator` if this is the first observation.
 *        b. Calls `predict(dt)` — dt derived from the measurement timestamp
 *           minus the track's `last_timestamp_ms`.
 *        c. Calls `correct(measurement)` to fold in the new position.
 *
 *  2. For every *active* track that did **not** receive a measurement:
 *        a. gap < `kMaxCoastMs` (1 500 ms) → `predict(dt)` only; the filter
 *           returns a mathematically projected (coasted) velocity.
 *        b. gap ≥ `kMaxCoastMs`            → the `SpeedEstimator` is erased
 *           from the map, freeing all internal `cv::Mat` memory.  This is
 *           the ID-loss / identity-switch guard.
 *
 * @par Thread Safety
 *   Not thread-safe.  Use one `TrackManager` per consumer thread.
 */
class TrackManager final {
public:
  /// Maximum gap before a track is considered permanently lost.
  static constexpr std::int64_t kMaxCoastMs = 1500LL;

  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  /**
   * @param pixels_per_metre      Homography scale; 0 = metric conversion
   *                              disabled (speed_mps / speed_kmh = 0).
   * @param process_noise_std     Forwarded to every new `SpeedEstimator`.
   * @param measurement_noise_std Forwarded to every new `SpeedEstimator`.
   */
  explicit TrackManager(float pixels_per_metre = 0.0f,
                        float process_noise_std = 2.0f,
                        float measurement_noise_std = 3.0f) noexcept
      : pixels_per_metre_(pixels_per_metre),
        process_noise_std_(process_noise_std),
        measurement_noise_std_(measurement_noise_std) {}

  // Non-copyable; movable.
  TrackManager(const TrackManager &) = delete;
  TrackManager &operator=(const TrackManager &) = delete;
  TrackManager(TrackManager &&) = default;
  TrackManager &operator=(TrackManager &&) = default;
  ~TrackManager() = default;

  // ------------------------------------------------------------------
  // Main update loop
  // ------------------------------------------------------------------

  /**
   * @brief Process one batch of measurements from a single detector frame.
   *
   * @param measurements  All `VehicleMeasurement` objects from this tick.
   *                      Multiple tracks may be present; invalid entries
   *                      (`is_valid() == false`) are silently ignored.
   * @param now_ms        Current wall-clock time (ms since UNIX epoch).
   *                      Used to age-out coasted tracks.
   *
   * @returns A vector of `SpeedEstimate` — one per active track (both
   *          corrected and coasted).  Destroyed tracks produce no entry.
   */
  [[nodiscard]] std::vector<SpeedEstimate>
  update(const std::vector<VehicleMeasurement> &measurements,
         std::int64_t now_ms) {
    std::vector<SpeedEstimate> results;
    results.reserve(tracks_.size() + measurements.size());

    // ── Step 1: process incoming measurements ─────────────────────────
    for (const auto &m : measurements) {
      if (!m.is_valid())
        continue;

      auto it = tracks_.find(m.track_id);
      if (it == tracks_.end()) {
        // ── New track: create, seed, return initial estimate ──────
        auto [ins_it, ok] = tracks_.emplace(
            std::piecewise_construct, std::forward_as_tuple(m.track_id),
            std::forward_as_tuple(m, process_noise_std_,
                                  measurement_noise_std_));
        static_cast<void>(ok); // always true for a new key
        // First estimate: velocity = 0, mark as non-coasted.
        results.push_back(ins_it->second.build_estimate(
            pixels_per_metre_, m.timestamp_ms, /*is_coasted=*/false));
      } else {
        // ── Known track: predict → correct ────────────────────────
        SpeedEstimator &est = it->second;
        const float dt_s =
            compute_dt_seconds(est.last_timestamp_ms(), m.timestamp_ms);

        est.predict(dt_s);
        est.correct(m);
        results.push_back(est.build_estimate(pixels_per_metre_, m.timestamp_ms,
                                             /*is_coasted=*/false));
      }
      seen_this_tick_.insert(m.track_id);
    }

    // ── Step 2: coast or reap tracks with no measurement this tick ─────
    std::vector<std::int32_t> to_erase;

    for (auto &[id, est] : tracks_) {
      if (seen_this_tick_.count(id))
        continue; // already handled above

      const std::int64_t gap_ms = now_ms - est.last_timestamp_ms();

      if (gap_ms >= kMaxCoastMs) {
        // Lost too long — schedule destruction.
        to_erase.push_back(id);
      } else {
        // Still within coast window — project state forward.
        const float dt_s =
            static_cast<float>(gap_ms > 0LL ? gap_ms : 1LL) * 1.0e-3f;
        est.predict(dt_s);
        results.push_back(
            est.build_estimate(pixels_per_metre_, now_ms, /*is_coasted=*/true));
      }
    }

    // Erase stale tracks (deferred to avoid iterator invalidation).
    for (std::int32_t id : to_erase) {
      tracks_.erase(id);
    }

    // Reset scratch set for the next call.
    seen_this_tick_.clear();

    return results;
  }

  // ------------------------------------------------------------------
  // Accessors / diagnostics
  // ------------------------------------------------------------------

  /// Number of currently active tracks.
  [[nodiscard]] std::size_t active_track_count() const noexcept {
    return tracks_.size();
  }

  /// Returns true iff a track with the given id is currently managed.
  [[nodiscard]] bool has_track(std::int32_t id) const noexcept {
    return tracks_.count(id) != 0;
  }

  /**
   * @brief Destroy all active tracks immediately (e.g., on pipeline reset
   *        or camera cut).  Memory held by `cv::KalmanFilter` instances is
   *        freed at once.
   */
  void clear() noexcept {
    tracks_.clear();
    seen_this_tick_.clear();
  }

  // ------------------------------------------------------------------
  // Runtime configuration
  // ------------------------------------------------------------------

  /**
   * @brief Update the pixel-to-metre scale used for speed conversion.
   *
   * Allows late-binding of homography calibration data without
   * reconstructing the manager or re-seeding any active filters.
   * Takes effect on the next `update()` call.
   */
  void set_pixels_per_metre(float ppm) noexcept { pixels_per_metre_ = ppm; }

private:
  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  /**
   * @brief Compute dt in seconds from two millisecond timestamps.
   *
   * Clamps to a minimum of 1 ms to avoid a zero-dt degenerate prediction
   * (which would leave the state unchanged and misrepresent the covariance
   * propagation).
   */
  [[nodiscard]] static float compute_dt_seconds(std::int64_t prev_ms,
                                                std::int64_t curr_ms) noexcept {
    const std::int64_t diff_ms = curr_ms - prev_ms;
    // Guard: if timestamps are non-monotone (clock jitter / wrap), use 1 ms.
    const std::int64_t clamped = diff_ms > 0LL ? diff_ms : 1LL;
    return static_cast<float>(clamped) * 1.0e-3f;
  }

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------

  float pixels_per_metre_;
  float process_noise_std_;
  float measurement_noise_std_;

  /// Per-track filter instances, keyed by tracker ID.
  std::unordered_map<std::int32_t, SpeedEstimator> tracks_;

  /**
   * @brief Scratch book-keeping: IDs that received a measurement in the
   *        current `update()` call.  Declared as a member to avoid
   *        a heap allocation on every call; cleared at the end of each tick.
   */
  std::unordered_set<std::int32_t> seen_this_tick_;
};

} // namespace vse
