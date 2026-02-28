/**
 * @file   smoke_test.cpp
 * @brief  Compile-time and minimal run-time validation of the VSE pipeline
 *         core headers.
 *
 * Checks performed
 * ----------------
 *  1. VehicleMeasurement:  construction, validity predicate, equality.
 *  2. MeasurementQueue:    push / pop round-trip across two std::threads,
 *                          shutdown + drain, full-queue try_push rejection.
 *  3. LockFreeMeasurementQueue: SPSC push/pop in two std::threads.
 *  4. INetworkTrigger:     polymorphic dispatch via NullNetworkTrigger.
 *
 * No OpenCV, no GUI.  Returns EXIT_SUCCESS on pass, EXIT_FAILURE on error.
 */

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "pipeline.hpp" // pulls in all vse headers

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_passes = 0;
static int g_fails = 0;

#define VSE_EXPECT(cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "[FAIL] " << __FILE__ << ':' << __LINE__ << "  " << #cond   \
                << '\n';                                                       \
      ++g_fails;                                                               \
    } else {                                                                   \
      ++g_passes;                                                              \
    }                                                                          \
  } while (false)

// ---------------------------------------------------------------------------
// Test: VehicleMeasurement
// ---------------------------------------------------------------------------
static void test_vehicle_measurement() {
  using namespace vse;

  std::cout << "--- VehicleMeasurement ---\n";

  const VehicleMeasurement default_m{};
  VSE_EXPECT(default_m.track_id == -1);
  VSE_EXPECT(default_m.bottom_center_x == 0.0f);
  VSE_EXPECT(default_m.bottom_center_y == 0.0f);
  VSE_EXPECT(default_m.timestamp_ms == 0);
  VSE_EXPECT(!default_m.is_valid());

  const VehicleMeasurement valid_m{42, 320.5f, 480.0f, 1'700'000'000'000LL};
  VSE_EXPECT(valid_m.is_valid());
  VSE_EXPECT(valid_m == valid_m);
  VSE_EXPECT(valid_m != default_m);
}

// ---------------------------------------------------------------------------
// Test: MeasurementQueue (blocking)
// ---------------------------------------------------------------------------
static void test_measurement_queue() {
  using namespace vse;
  constexpr std::size_t kCap = 8;
  std::cout << "--- MeasurementQueue (blocking, cap=" << kCap << ") ---\n";

  MeasurementQueue<kCap> queue;

  // --- push / pop round-trip across threads ---
  constexpr int kItems = 100;
  std::thread producer([&] {
    for (int i = 0; i < kItems; ++i) {
      VehicleMeasurement m{i, static_cast<float>(i), 0.0f,
                           static_cast<std::int64_t>(i)};
      [[maybe_unused]] bool ok = queue.push(m);
    }
    queue.shutdown();
  });

  int received = 0;
  while (auto item = queue.pop()) {
    VSE_EXPECT(item->track_id == received);
    ++received;
  }
  producer.join();

  VSE_EXPECT(received == kItems);
  std::cout << "  Received " << received << " / " << kItems << " items.\n";

  // --- try_push rejects when full ---
  MeasurementQueue<2> tiny;
  VSE_EXPECT(tiny.try_push({0, 0, 0, 0}));
  VSE_EXPECT(tiny.try_push({1, 0, 0, 0}));
  VSE_EXPECT(!tiny.try_push({2, 0, 0, 0})); // full — must be rejected
}

// ---------------------------------------------------------------------------
// Test: LockFreeMeasurementQueue (SPSC)
// ---------------------------------------------------------------------------
static void test_lock_free_queue() {
  using namespace vse;
  constexpr std::size_t kCap = 64;
  std::cout << "--- LockFreeMeasurementQueue (SPSC, cap=" << kCap << ") ---\n";

  LockFreeMeasurementQueue<kCap> queue;
  constexpr int kItems = 500;

  std::atomic<int> consumed{0};

  std::thread producer([&] {
    for (int i = 0; i < kItems; ++i) {
      VehicleMeasurement m{i, static_cast<float>(i), 0.0f,
                           static_cast<std::int64_t>(i)};
      while (!queue.push(m)) {
        std::this_thread::yield(); // spin until space
      }
    }
  });

  std::thread consumer([&] {
    int expected = 0;
    while (expected < kItems) {
      if (auto item = queue.pop()) {
        VSE_EXPECT(item->track_id == expected);
        ++expected;
        consumed.store(expected, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  VSE_EXPECT(consumed.load() == kItems);
  std::cout << "  Consumed " << consumed.load() << " / " << kItems
            << " items.\n";
}

// ---------------------------------------------------------------------------
// Test: INetworkTrigger polymorphic dispatch
// ---------------------------------------------------------------------------
static void test_network_trigger() {
  using namespace vse;
  std::cout << "--- INetworkTrigger ---\n";

  // Inline concrete subclass that records the last call.
  struct RecordingTrigger final : public INetworkTrigger {
    std::int32_t last_track_id{-1};
    float last_speed{0.0f};
    int call_count{0};

    void onHighRiskDetected(std::int32_t track_id, float speed_kmh) override {
      last_track_id = track_id;
      last_speed = speed_kmh;
      ++call_count;
    }
  };

  RecordingTrigger trigger;
  INetworkTrigger &iface = trigger; // access through interface

  iface.onHighRiskDetected(7, 145.5f);
  VSE_EXPECT(trigger.last_track_id == 7);
  VSE_EXPECT(trigger.last_speed == 145.5f);
  VSE_EXPECT(trigger.call_count == 1);

  // NullNetworkTrigger should compile and run without crash.
  NullNetworkTrigger null_sink;
  null_sink.onHighRiskDetected(99, 200.0f);
  std::cout << "  NullNetworkTrigger::onHighRiskDetected OK (no-op).\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  std::cout << "=== VSE Pipeline Smoke Test ===\n\n";

  test_vehicle_measurement();
  test_measurement_queue();
  test_lock_free_queue();
  test_network_trigger();

  std::cout << "\n=== Results: " << g_passes << " passed, " << g_fails
            << " failed ===\n";

  return (g_fails == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
