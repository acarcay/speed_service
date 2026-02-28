/**
 * @file   network_trigger.hpp
 * @brief  Abstract interface for asynchronous 5G Quality-on-Demand (QoD)
 *         API dispatch upon high-risk speed detection events.
 *
 * C++17 / C++20 compatible.  No GUI dependencies.
 *
 * Design notes
 * ------------
 * - Implementations run in a *dedicated async dispatch thread* so that
 *   neither the GPU inference loop nor the Kalman tracker are stalled
 *   waiting for network I/O.
 * - Callers are responsible for ensuring that concrete implementations
 *   are thread-safe (i.e., `onHighRiskDetected` may be invoked from
 *   multiple worker threads simultaneously in a pool-based deployment).
 * - The class is non-copyable and non-movable by default to prevent
 *   accidental slicing of polymorphic objects.
 */

#pragma once

#include <cstdint> // std::int32_t
#include <string>  // std::string (for overloads in subclasses)

namespace vse {

// ---------------------------------------------------------------------------
// INetworkTrigger
// ---------------------------------------------------------------------------

/**
 * @brief Pure-virtual interface for 5G QoD event dispatchers.
 *
 * Implement this class to connect the speed-estimation pipeline to any
 * downstream network control plane (e.g., CAMARA / 3GPP QoD REST API,
 * gRPC endpoint, mock sink for unit tests).
 *
 * @par Thread Safety
 * `onHighRiskDetected` is expected to be called from the Kalman tracker
 * thread whenever a tracked vehicle's estimated speed exceeds a configured
 * threshold.  Concrete implementations MUST be thread-safe.
 *
 * @par Lifetime
 * Objects of this type should be managed through `std::shared_ptr` or
 * passed by reference with a lifetime that exceeds all producer threads.
 */
class INetworkTrigger {
public:
  // ------------------------------------------------------------------
  // Virtual interface
  // ------------------------------------------------------------------

  /**
   * @brief Called asynchronously when a vehicle's speed crosses the
   *        high-risk threshold configured by the operator.
   *
   * @param track_id   The unique tracker ID of the offending vehicle.
   *                   Matches `VehicleMeasurement::track_id`.
   * @param speed_kmh  Estimated instantaneous speed in km/h as computed
   *                   by the Kalman filter's velocity state.
   *
   * @note  Implementations should be non-blocking where possible.
   *        Use an internal async queue / thread pool if a network call
   *        must be made; do NOT perform synchronous blocking I/O on
   *        the calling thread.
   */
  virtual void onHighRiskDetected(std::int32_t track_id, float speed_kmh) = 0;

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Virtual destructor — mandatory for safe polymorphic deletion.
  virtual ~INetworkTrigger() = default;

  // Disallow copy and move at the interface level to prevent slicing.
  INetworkTrigger(const INetworkTrigger &) = delete;
  INetworkTrigger &operator=(const INetworkTrigger &) = delete;
  INetworkTrigger(INetworkTrigger &&) = delete;
  INetworkTrigger &operator=(INetworkTrigger &&) = delete;

protected:
  /// Only derived classes may be constructed.
  INetworkTrigger() = default;
};

// ---------------------------------------------------------------------------
// NullNetworkTrigger  (no-op implementation — useful for unit tests /
//                      dry-run deployments)
// ---------------------------------------------------------------------------

/**
 * @brief A no-op implementation of `INetworkTrigger`.
 *
 * Satisfies the interface contract while performing no I/O.  Useful as:
 *  - A safe default when no QoD integration is configured.
 *  - A drop-in replacement in unit tests that isolate tracking logic.
 */
class NullNetworkTrigger final : public INetworkTrigger {
public:
  /// Discards the event silently.
  void onHighRiskDetected([[maybe_unused]] std::int32_t track_id,
                          [[maybe_unused]] float speed_kmh) override {}
};

} // namespace vse
