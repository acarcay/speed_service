/**
 * @file   types.hpp
 * @brief  Core plain-old-data types shared across the vehicle speed
 *         estimation pipeline.
 *
 * C++17 / C++20 compatible.  No GUI dependencies.  No OpenCV types
 * are intentionally included here; callers that bridge from a
 * cv::Mat pipeline must convert before construction.
 */

#pragma once

#include <cstdint>   // std::int64_t, std::int32_t
#include <string>    // std::string (forward-compatible)

namespace vse {  // vehicle speed estimation

// ---------------------------------------------------------------------------
// VehicleMeasurement
// ---------------------------------------------------------------------------

/**
 * @brief A single, timestamped observation emitted by the YOLO detector for
 *        one tracked vehicle.
 *
 * Layout rationale
 * ----------------
 * - `track_id`        : Unique identifier assigned by the SORT / DeepSORT
 *                       tracker.  Negative values are reserved (invalid).
 * - `bottom_center_x` : Horizontal pixel coordinate of the bounding-box
 *                       bottom-centre contact point (sub-pixel precision
 *                       preserved as float).
 * - `bottom_center_y` : Vertical pixel coordinate of the same point.
 * - `timestamp_ms`    : Wall-clock milliseconds since UNIX epoch, captured
 *                       immediately after GPU inference completes so that
 *                       latency introduced by enqueuing is excluded from
 *                       speed calculations.
 *
 * Memory layout
 * -------------
 * The struct is trivially copyable and standard-layout; it can therefore
 * be safely used with lock-free atomic operations and ring-buffer
 * implementations that rely on memcpy semantics.
 */
struct VehicleMeasurement {
    /// Unique tracker ID.  Value −1 indicates an unassigned / invalid entry.
    std::int32_t track_id{-1};

    /// Horizontal pixel coordinate of the bounding-box bottom-centre point.
    float bottom_center_x{0.0f};

    /// Vertical pixel coordinate of the bounding-box bottom-centre point.
    float bottom_center_y{0.0f};

    /// Wall-clock capture time in milliseconds since UNIX epoch.
    std::int64_t timestamp_ms{0};

    // ------------------------------------------------------------------
    // Convenience helpers
    // ------------------------------------------------------------------

    /// Returns true when the measurement carries a valid track assignment.
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return track_id >= 0;
    }

    /// Equality — useful for unit tests and de-duplication guards.
    [[nodiscard]] constexpr bool operator==(const VehicleMeasurement& rhs) const noexcept {
        return track_id        == rhs.track_id
            && bottom_center_x == rhs.bottom_center_x
            && bottom_center_y == rhs.bottom_center_y
            && timestamp_ms    == rhs.timestamp_ms;
    }

    [[nodiscard]] constexpr bool operator!=(const VehicleMeasurement& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// Ensure the struct upholds the properties we rely on.
static_assert(std::is_trivially_copyable_v<VehicleMeasurement>,
              "VehicleMeasurement must remain trivially copyable.");
static_assert(std::is_standard_layout_v<VehicleMeasurement>,
              "VehicleMeasurement must remain standard-layout.");

}  // namespace vse
