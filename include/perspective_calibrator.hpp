/**
 * @file   perspective_calibrator.hpp
 * @brief  Inverse Perspective Mapping (IPM) and pixel-to-metre scale
 *         calibration for the vehicle speed estimation pipeline.
 *
 * Overview
 * --------
 * We do **not** have camera intrinsics, height, or tilt — only a dataset of
 * trajectories from the tracker paired with ground-truth speeds (km/h)
 * provided by the competition committee.
 *
 * This file provides `PerspectiveCalibrator` which solves two problems:
 *
 *  1. **IPM homography** (`cv::Mat H_`, 3×3) — maps image-plane pixel
 *     coordinates to a synthetic Bird's-Eye View (BEV) plane.  Two
 *     strategies are supported:
 *
 *     a. **Automatic** (`computeIPMFromFrame`) — runs a Canny+HoughLinesP
 *        pipeline on a representative frame, clusters lane-boundary lines by
 *        slope direction, fits left/right boundary rays, and derives the
 *        source trapezoid that `cv::getPerspectiveTransform` warps to a
 *        rectangular BEV canvas.
 *
 *     b. **Manual** (`setRoadPoints`) — the caller supplies the four corners
 *        of the road trapezoid directly (useful when automatic detection
 *        fails or when camera geometry is known).
 *
 *  2. **Scale factor K** (pixels per metre, `float scale_px_per_m_`) —
 *     learned offline via `calibrateScaleFactor`.  The function:
 *       - Projects each position in `track_history` through H_ into BEV,
 *       - Accumulates the total BEV Euclidean distance (pixels),
 *       - Divides by the total elapsed time (seconds) to get pixel speed,
 *       - Solves K = pixel_speed / ground_truth_speed_mps.
 *
 * Real-time usage
 * ---------------
 * Once both H_ and K are available `calculateSpeedKmh(pt1, pt2, dt)` gives
 * the metrical speed for any pair of consecutive tracker positions.
 *
 * Units
 * -----
 * - All raw points: image-plane pixels.
 * - BEV points:     BEV-canvas pixels  (after H_ transform).
 * - K:              px / m  (BEV pixels per real-world metre).
 * - Speed output:   km/h.
 *
 * Dependencies
 * ------------
 * OpenCV >= 4.x  (`core`, `imgproc`, `calib3d`).  No GUI modules required.
 *
 * C++17 / C++20 compatible.  Not thread-safe; construct one instance per
 * calibration context.  `calculateSpeedKmh` is const and may be called from
 * multiple threads if K and H_ are already fixed.
 */

#pragma once

#include <algorithm> // std::sort, std::remove_if
#include <array>     // std::array
#include <cmath>     // std::hypot, std::abs
#include <stdexcept> // std::runtime_error, std::invalid_argument
#include <vector>    // std::vector

#include <opencv2/calib3d.hpp> // cv::findHomography (for RANSAC variant)
#include <opencv2/core.hpp>    // cv::Mat, cv::Point2f, cv::Size, etc.
#include <opencv2/imgproc.hpp> // cv::Canny, cv::HoughLinesP, cv::cvtColor

#include "types.hpp" // vse::VehicleMeasurement

namespace vse {

// ===========================================================================
// PerspectiveCalibrator
// ===========================================================================

class PerspectiveCalibrator final {
public:
  // ------------------------------------------------------------------
  // Types
  // ------------------------------------------------------------------

  /**
   * @brief Four image-plane corners of the road trapezoid, in order:
   *        top-left, top-right, bottom-right, bottom-left.
   */
  using RoadQuad = std::array<cv::Point2f, 4>;

  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------

  /// Default BEV canvas width (pixels).
  static constexpr int kDefaultBevWidth = 600;
  /// Default BEV canvas height (pixels).
  static constexpr int kDefaultBevHeight = 800;

  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  /**
   * @param bev_size  Size of the synthetic Bird's-Eye View canvas.
   *                  The homography maps road pixels to this rectangle.
   */
  explicit PerspectiveCalibrator(
      cv::Size bev_size = {kDefaultBevWidth, kDefaultBevHeight}) noexcept
      : bev_size_(bev_size) {}

  PerspectiveCalibrator(const PerspectiveCalibrator &) = default;
  PerspectiveCalibrator &operator=(const PerspectiveCalibrator &) = default;
  ~PerspectiveCalibrator() = default;

  // ==================================================================
  // Strategy A — Automatic IPM from a representative frame
  // ==================================================================

  /**
   * @brief Derive the IPM homography by automatically detecting parallel
   *        road-boundary lines in `frame`.
   *
   * Algorithm
   * ---------
   * 1. Grayscale + Gaussian blur.
   * 2. Canny edge detection.
   * 3. HoughLinesP to extract line segments.
   * 4. Filter by inclination angle (reject near-horizontal lines that
   *    are road surface markings rather than boundaries).
   * 5. Cluster into "left" (negative slope) and "right" (positive slope).
   * 6. Weighted-least-squares fit of a line through each cluster's
   *    midpoints (weight = segment length).
   * 7. Intersect the two fitted rays to find the vanishing point.
   * 8. Extrapolate each ray to form a trapezoidal source region.
   * 9. Call `cv::getPerspectiveTransform` to get H_.
   *
   * @param frame         A representative BGR (or grayscale) video frame.
   *                      Need not be the first frame — pick a clear,
   *                      unoccluded image of the road.
   * @param roi           Optional sub-region of `frame` to search
   *                      (defaults to the lower 60 % of the image where
   *                      most road-boundary line energy is concentrated).
   * @param canny_low     Lower Canny threshold.
   * @param canny_high    Upper Canny threshold.
   * @param hough_thresh  HoughLinesP accumulator threshold.
   *
   * @returns `true` on success; `false` if insufficient lane lines were
   *          found (caller should fall back to `setRoadPoints`).
   *
   * @throws  `std::invalid_argument` if `frame` is empty.
   */
  bool computeIPMFromFrame(const cv::Mat &frame, cv::Rect roi = {},
                           double canny_low = 50.0, double canny_high = 150.0,
                           int hough_thresh = 40) {
    if (frame.empty()) {
      throw std::invalid_argument(
          "PerspectiveCalibrator::computeIPMFromFrame: frame is empty.");
    }

    const int W = frame.cols;
    const int H = frame.rows;

    // Default ROI: lower 60 % of the frame (rich in road markings).
    if (roi.empty() || !roi.area()) {
      roi = cv::Rect(0, static_cast<int>(H * 0.4f), W,
                     static_cast<int>(H * 0.6f));
    }
    roi &= cv::Rect(0, 0, W, H); // clamp to frame boundaries

    // 1. Grayscale
    cv::Mat gray;
    if (frame.channels() == 3) {
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = frame.clone();
    }
    cv::Mat roi_gray = gray(roi);

    // 2. Gaussian blur
    cv::GaussianBlur(roi_gray, roi_gray, {5, 5}, 0);

    // 3. Canny
    cv::Mat edges;
    cv::Canny(roi_gray, edges, canny_low, canny_high);

    // 4. HoughLinesP
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(edges, raw_lines,
                    /*rho=*/1, /*theta=*/CV_PI / 180.0, hough_thresh,
                    /*minLineLength=*/40, /*maxLineGap=*/20);

    if (raw_lines.empty())
      return false;

    // 5. Filter and cluster into left / right boundary segments.
    //    Translate ROI-relative coordinates back to full-frame space.
    std::vector<cv::Vec4i> left_segs, right_segs;

    for (const auto &seg : raw_lines) {
      const float x1 = static_cast<float>(seg[0]);
      const float y1 = static_cast<float>(seg[1]);
      const float x2 = static_cast<float>(seg[2]);
      const float y2 = static_cast<float>(seg[3]);

      const float dx = x2 - x1;
      if (std::abs(dx) < 1.0f)
        continue; // vertical — skip

      const float slope = (y2 - y1) / dx;

      // Reject near-horizontal lines (|slope| < 0.3 ~17°).
      if (std::abs(slope) < 0.3f)
        continue;

      // Translate to full-frame y.
      const int y_off = roi.y;
      const cv::Vec4i full{seg[0], seg[1] + y_off, seg[2], seg[3] + y_off};

      if (slope < 0.0f) {
        left_segs.push_back(full); // lane boundary on the left
      } else {
        right_segs.push_back(full); // lane boundary on the right
      }
    }

    if (left_segs.size() < 2 || right_segs.size() < 2)
      return false;

    // 6. Weighted least-squares line fit for each side.
    const auto fit_ray = [](const std::vector<cv::Vec4i> &segs)
        -> std::pair<float, float> // (slope m, intercept b)
    {
      // Each segment contributes its midpoint, weighted by length.
      double sw = 0, swx = 0, swy = 0, swx2 = 0, swxy = 0;
      for (const auto &s : segs) {
        const float mx = 0.5f * (s[0] + s[2]);
        const float my = 0.5f * (s[1] + s[3]);
        const float len = std::hypot(s[2] - s[0], s[3] - s[1]);
        sw += len;
        swx += len * mx;
        swy += len * my;
        swx2 += len * mx * mx;
        swxy += len * mx * my;
      }
      const double denom = sw * swx2 - swx * swx;
      if (std::abs(denom) < 1e-6)
        return {0.0f, 0.0f};
      const float m = static_cast<float>((sw * swxy - swx * swy) / denom);
      const float b = static_cast<float>((swy - m * swx) / sw);
      return {m, b};
    };

    const auto [ml, bl] = fit_ray(left_segs);
    const auto [mr, br] = fit_ray(right_segs);

    if (std::abs(mr - ml) < 1e-4f)
      return false; // parallel — no VP

    // 7. Vanishing point: ml*x+bl = mr*x+br → x_vp = (br-bl)/(ml-mr)
    const float x_vp = (br - bl) / (ml - mr);
    const float y_vp = ml * x_vp + bl;

    // 8. Source trapezoid.
    //
    //    bottom_y: scan line near the bottom of the frame (80 % height).
    //    top_y:    scan line slightly below the vanishing point.
    //
    const float bottom_y = static_cast<float>(H) * 0.80f;
    const float top_y = y_vp + static_cast<float>(H) * 0.08f;

    const auto x_at = [](float y, float m, float b) { return (y - b) / m; };

    const RoadQuad src = {{
        {x_at(top_y, ml, bl), top_y},       // top-left
        {x_at(top_y, mr, br), top_y},       // top-right
        {x_at(bottom_y, mr, br), bottom_y}, // bottom-right
        {x_at(bottom_y, ml, bl), bottom_y}  // bottom-left
    }};

    return compute_homography_from_quad(src);
  }

  // ==================================================================
  // Strategy B — Manual road-point specification
  // ==================================================================

  /**
   * @brief Set the IPM homography from manually identified road corners.
   *
   * @param src_quad  Four image-plane corners of the road trapezoid:
   *                  {top-left, top-right, bottom-right, bottom-left}.
   *                  These should correspond to a real-world rectangle
   *                  (e.g., lane markings with known spacing).
   */
  void setRoadPoints(const RoadQuad &src_quad) {
    compute_homography_from_quad(src_quad);
  }

  // ==================================================================
  // Offline calibration
  // ==================================================================

  /**
   * @brief Reverse-engineer the pixel-to-metre scale factor K from a
   *        ground-truth labelled track.
   *
   * Method
   * ------
   * For each consecutive pair (m_i, m_{i+1}) in the sorted track history:
   *   1. Project both positions through H_ into BEV space.
   *   2. Accumulate BEV Euclidean distance Σd_px.
   *   3. Accumulate elapsed time Σdt_s.
   *
   * Then:
   *   pixel_speed_px_per_s = Σd_px / Σdt_s
   *   K = pixel_speed_px_per_s / ground_truth_speed_mps
   *     where ground_truth_speed_mps = ground_truth_speed_kmh / 3.6
   *
   * The result is stored internally and used by `calculateSpeedKmh`.
   *
   * @param track_history       Ordered or unordered sequence of raw
   *                            image-plane measurements from one vehicle.
   *                            Must contain at least 2 valid entries.
   * @param ground_truth_speed_kmh  Mean speed of the vehicle over the
   *                            duration of `track_history`, as provided
   *                            by the competition committee (km/h).
   *
   * @returns The computed scale factor K (px/m).  Also stored internally.
   *
   * @throws std::runtime_error   if no IPM matrix has been set, if fewer
   *                              than 2 valid measurements exist, or if
   *                              the accumulated distance or time is zero.
   * @throws std::invalid_argument if ground_truth_speed_kmh <= 0.
   */
  float calibrateScaleFactor(std::vector<VehicleMeasurement> track_history,
                             float ground_truth_speed_kmh) {
    if (H_.empty()) {
      throw std::runtime_error(
          "calibrateScaleFactor: IPM matrix not set.  Call "
          "computeIPMFromFrame or setRoadPoints first.");
    }
    if (ground_truth_speed_kmh <= 0.0f) {
      throw std::invalid_argument(
          "calibrateScaleFactor: ground_truth_speed_kmh must be > 0.");
    }

    // Remove invalid entries, sort by timestamp.
    track_history.erase(std::remove_if(track_history.begin(),
                                       track_history.end(),
                                       [](const VehicleMeasurement &m) {
                                         return !m.is_valid();
                                       }),
                        track_history.end());

    if (track_history.size() < 2) {
      throw std::runtime_error(
          "calibrateScaleFactor: need at least 2 valid measurements.");
    }

    std::sort(track_history.begin(), track_history.end(),
              [](const VehicleMeasurement &a, const VehicleMeasurement &b) {
                return a.timestamp_ms < b.timestamp_ms;
              });

    // Project each position into BEV space and accumulate distance & time.
    double total_dist_px = 0.0;
    double total_time_s = 0.0;

    for (std::size_t i = 0; i + 1 < track_history.size(); ++i) {
      const auto &m0 = track_history[i];
      const auto &m1 = track_history[i + 1];

      const float dt_s =
          static_cast<float>(m1.timestamp_ms - m0.timestamp_ms) * 1.0e-3f;
      if (dt_s <= 0.0f)
        continue; // duplicate or non-monotone timestamp

      const cv::Point2f bev0 =
          transform_to_bev({m0.bottom_center_x, m0.bottom_center_y});
      const cv::Point2f bev1 =
          transform_to_bev({m1.bottom_center_x, m1.bottom_center_y});

      const double d_px = std::hypot(static_cast<double>(bev1.x - bev0.x),
                                     static_cast<double>(bev1.y - bev0.y));

      total_dist_px += d_px;
      total_time_s += static_cast<double>(dt_s);
    }

    if (total_time_s <= 0.0 || total_dist_px <= 0.0) {
      throw std::runtime_error(
          "calibrateScaleFactor: accumulated distance or time is zero.  "
          "Check that timestamps are monotonically increasing and that "
          "the vehicle actually moved.");
    }

    // pixel_speed (px/s) = total_dist_px / total_time_s
    // K (px/m)           = pixel_speed / ground_truth_speed_mps
    const double gt_speed_mps =
        static_cast<double>(ground_truth_speed_kmh) / 3.6;
    const double pixel_speed = total_dist_px / total_time_s;

    scale_px_per_m_ = static_cast<float>(pixel_speed / gt_speed_mps);
    calibrated_ = true;
    return scale_px_per_m_;
  }

  // ==================================================================
  // Real-time speed calculation
  // ==================================================================

  /**
   * @brief Compute vehicle speed (km/h) from two consecutive image-plane
   *        tracker positions and the elapsed time between them.
   *
   * @param pt1         Image-plane position at time t.
   * @param pt2         Image-plane position at time t + dt_seconds.
   * @param dt_seconds  Elapsed time between the two observations (seconds).
   *                    Must be > 0.
   *
   * @returns Speed in km/h.
   *
   * @throws std::runtime_error if the object is not fully calibrated
   *         (IPM matrix or scale factor missing).
   * @throws std::invalid_argument if dt_seconds <= 0.
   */
  [[nodiscard]] float calculateSpeedKmh(cv::Point2f pt1, cv::Point2f pt2,
                                        float dt_seconds) const {
    if (H_.empty()) {
      throw std::runtime_error("calculateSpeedKmh: IPM matrix not set.");
    }
    if (!calibrated_) {
      throw std::runtime_error(
          "calculateSpeedKmh: scale factor not calibrated.  Call "
          "calibrateScaleFactor first.");
    }
    if (dt_seconds <= 0.0f) {
      throw std::invalid_argument("calculateSpeedKmh: dt_seconds must be > 0.");
    }

    const cv::Point2f bev1 = transform_to_bev(pt1);
    const cv::Point2f bev2 = transform_to_bev(pt2);

    const float dist_px = std::hypot(bev2.x - bev1.x, bev2.y - bev1.y);

    // speed_mps = dist_px / (K [px/m] * dt_s)
    const float speed_mps = dist_px / (scale_px_per_m_ * dt_seconds);
    return speed_mps * 3.6f;
  }

  // ==================================================================
  // Accessors
  // ==================================================================

  /**
   * @brief Project a single image-plane point into BEV space.
   *
   * Useful for visualisation and unit tests.
   *
   * @throws std::runtime_error if no IPM matrix has been set.
   */
  [[nodiscard]] cv::Point2f transformToBEV(cv::Point2f pt) const {
    if (H_.empty()) {
      throw std::runtime_error("transformToBEV: IPM matrix not set.");
    }
    return transform_to_bev(pt);
  }

  /// The 3×3 IPM homography matrix (empty if not yet set).
  [[nodiscard]] const cv::Mat &ipmMatrix() const noexcept { return H_; }

  /// Learned pixel-to-metre scale factor (0 if not yet calibrated).
  [[nodiscard]] float scaleFactor() const noexcept { return scale_px_per_m_; }

  /// True once both H_ and scale_px_per_m_ are set.
  [[nodiscard]] bool isCalibrated() const noexcept {
    return !H_.empty() && calibrated_;
  }

  /// Size of the BEV canvas this calibrator was constructed for.
  [[nodiscard]] cv::Size bevSize() const noexcept { return bev_size_; }

private:
  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  /**
   * @brief Build H_ from a source road trapezoid, mapping it to the
   *        full BEV canvas rectangle.
   *
   * BEV destination corners (clock-wise from top-left):
   *   (0,0) → (W,0) → (W,H) → (0,H)
   *
   * @returns true on success.
   */
  bool compute_homography_from_quad(const RoadQuad &src) {
    const float W = static_cast<float>(bev_size_.width);
    const float H = static_cast<float>(bev_size_.height);

    const RoadQuad dst = {{
        {0.0f, 0.0f}, // top-left
        {W, 0.0f},    // top-right
        {W, H},       // bottom-right
        {0.0f, H}     // bottom-left
    }};

    // cv::getPerspectiveTransform requires std::vector<cv::Point2f>.
    const std::vector<cv::Point2f> src_v(src.begin(), src.end());
    const std::vector<cv::Point2f> dst_v(dst.begin(), dst.end());

    H_ = cv::getPerspectiveTransform(src_v, dst_v);
    return !H_.empty();
  }

  /**
   * @brief Apply the homography H_ to a single image-plane point.
   *
   * Performs the standard projective division:
   *   [wx', wy', w]^T = H * [x, y, 1]^T
   *   result = (x'/w, y'/w)
   *
   * Note: this is intentionally inlined (no virtual dispatch, no heap
   * allocation) for the real-time path.
   */
  [[nodiscard]] cv::Point2f transform_to_bev(cv::Point2f pt) const noexcept {
    // H_ is CV_64F (double) from getPerspectiveTransform.
    const double *h = H_.ptr<double>();

    const double w = h[6] * pt.x + h[7] * pt.y + h[8];
    const double xp = h[0] * pt.x + h[1] * pt.y + h[2];
    const double yp = h[3] * pt.x + h[4] * pt.y + h[5];

    return {static_cast<float>(xp / w), static_cast<float>(yp / w)};
  }

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------

  cv::Size bev_size_;          ///< BEV canvas dimensions.
  cv::Mat H_;                  ///< 3×3 IPM homography (CV_64F).
  float scale_px_per_m_{0.0f}; ///< K: BEV pixels per real-world metre.
  bool calibrated_{false};     ///< True once calibrateScaleFactor succeeds.
};

} // namespace vse
