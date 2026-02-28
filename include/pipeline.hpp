/**
 * @file   pipeline.hpp
 * @brief  Umbrella include for the Vehicle Speed Estimation (VSE) pipeline.
 *
 * Including this single header in a translation unit brings in:
 *   - vse::VehicleMeasurement       (types.hpp)
 *   - vse::MeasurementQueue         (measurement_queue.hpp)
 *   - vse::LockFreeMeasurementQueue (measurement_queue.hpp)
 *   - vse::INetworkTrigger          (network_trigger.hpp)
 *   - vse::NullNetworkTrigger       (network_trigger.hpp)
 *   - vse::SpeedEstimate            (speed_estimator.hpp)
 *   - vse::SpeedEstimator           (speed_estimator.hpp)
 *   - vse::TrackManager             (speed_estimator.hpp)
 *   - vse::PerspectiveCalibrator    (perspective_calibrator.hpp)
 *
 * Avoid including individual sub-headers; use this file everywhere so that
 * the dependency graph remains explicit and refactoring remains cheap.
 *
 * @note  `speed_estimator.hpp` depends on OpenCV (`opencv_video`).  Make
 *        sure the consuming CMake target links against `${OpenCV_LIBS}`.
 */

#pragma once

#include "measurement_queue.hpp"
#include "network_trigger.hpp"
#include "perspective_calibrator.hpp"
#include "speed_estimator.hpp"
#include "types.hpp"
