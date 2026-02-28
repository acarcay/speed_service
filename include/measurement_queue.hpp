/**
 * @file   measurement_queue.hpp
 * @brief  Thread-safe, bounded MPSC queue for passing `VehicleMeasurement`
 *         objects from the YOLO GPU-inference thread (producer) to the
 *         CPU-based Kalman tracking thread (consumer).
 *
 * C++17 / C++20 compatible.  No GUI dependencies.  No OpenCV types.
 *
 * Design rationale
 * ----------------
 * Two complementary implementations are provided behind a single public
 * interface, selectable at compile time:
 *
 *  1.  **`MeasurementQueue` (default)** — mutex + `std::condition_variable`
 *      with a bounded `std::deque`.  This is the correct production choice:
 *      it puts the consumer to sleep when idle (saving CPU), applies back-
 *      pressure to the producer when the queue is full (preventing unbounded
 *      memory growth), and is straightforward to reason about.
 *
 *  2.  **`LockFreeMeasurementQueue`** — single-producer / single-consumer
 *      (SPSC) ring buffer built on `std::atomic` operations with
 *      `std::memory_order_release` / `std::memory_order_acquire` fences.
 *      Use this only when the consumer thread MUST NOT block (e.g., it
 *      drives another real-time subsystem).  It burns a CPU core polling.
 *
 * Both classes are final and non-copyable / non-movable.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef> // std::size_t
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "types.hpp"

namespace vse {

// ===========================================================================
// MeasurementQueue  —  mutex-guarded, bounded, blocking
// ===========================================================================

/**
 * @brief Bounded, blocking, thread-safe queue for `VehicleMeasurement`.
 *
 * Producer behaviour
 * ------------------
 * - `push(item)` blocks until capacity is available or the queue is
 *   shut down.  This intentionally throttles the GPU thread rather than
 *   allocating unbounded memory.
 * - `try_push(item)` returns `false` immediately if full (non-blocking
 *   alternative for latency-critical paths).
 *
 * Consumer behaviour
 * ------------------
 * - `pop()` blocks until an item is available or the queue is shut down.
 *   Returns `std::nullopt` on shutdown to signal the consumer to exit.
 * - `try_pop()` is the non-blocking counterpart.
 *
 * Shutdown
 * --------
 * Call `shutdown()` once (from any thread) to unblock all waiting
 * producers/consumers and allow graceful pipeline drain.
 *
 * @tparam Capacity  Maximum number of items the queue may hold at once.
 *                   Defaults to 128, which is generous for typical 30–60 fps
 *                   pipelines with a single slow consumer.
 */
template <std::size_t Capacity = 128> class MeasurementQueue final {
  static_assert(Capacity > 0, "Capacity must be at least 1.");

public:
  // ------------------------------------------------------------------
  // Construction & destruction
  // ------------------------------------------------------------------

  MeasurementQueue() = default;
  ~MeasurementQueue() = default;

  // Non-copyable, non-movable — the queue owns synchronisation objects.
  MeasurementQueue(const MeasurementQueue &) = delete;
  MeasurementQueue &operator=(const MeasurementQueue &) = delete;
  MeasurementQueue(MeasurementQueue &&) = delete;
  MeasurementQueue &operator=(MeasurementQueue &&) = delete;

  // ------------------------------------------------------------------
  // Producer API
  // ------------------------------------------------------------------

  /**
   * @brief Enqueue an item, blocking if the queue is at capacity.
   *
   * @returns `true`  if the item was enqueued.
   * @returns `false` if the queue has been shut down.
   */
  [[nodiscard]] bool push(VehicleMeasurement item) {
    std::unique_lock lock{mutex_};
    not_full_cv_.wait(lock,
                      [this] { return queue_.size() < Capacity || shutdown_; });
    if (shutdown_)
      return false;
    queue_.push_back(std::move(item));
    lock.unlock();
    not_empty_cv_.notify_one();
    return true;
  }

  /**
   * @brief Try to enqueue without blocking.
   *
   * @returns `true`  if enqueued.
   * @returns `false` if the queue is full or shut down.
   */
  [[nodiscard]] bool try_push(VehicleMeasurement item) {
    std::lock_guard lock{mutex_};
    if (queue_.size() >= Capacity || shutdown_)
      return false;
    queue_.push_back(std::move(item));
    not_empty_cv_.notify_one();
    return true;
  }

  // ------------------------------------------------------------------
  // Consumer API
  // ------------------------------------------------------------------

  /**
   * @brief Remove and return the front item, blocking until one is
   *        available.
   *
   * @returns The next `VehicleMeasurement`, or `std::nullopt` if the
   *          queue has been shut down and drained.
   */
  [[nodiscard]] std::optional<VehicleMeasurement> pop() {
    std::unique_lock lock{mutex_};
    not_empty_cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (queue_.empty())
      return std::nullopt; // shut down + drained
    VehicleMeasurement item = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    not_full_cv_.notify_one();
    return item;
  }

  /**
   * @brief Try to remove and return the front item without blocking.
   *
   * @returns The item, or `std::nullopt` if the queue is empty or shut down.
   */
  [[nodiscard]] std::optional<VehicleMeasurement> try_pop() {
    std::lock_guard lock{mutex_};
    if (queue_.empty())
      return std::nullopt;
    VehicleMeasurement item = std::move(queue_.front());
    queue_.pop_front();
    not_full_cv_.notify_one();
    return item;
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /**
   * @brief Signal all waiting threads to unblock and exit.
   *
   * After `shutdown()`, `push` returns `false` and `pop` drains any
   * remaining items before returning `std::nullopt`.  Safe to call
   * from any thread; idempotent.
   */
  void shutdown() noexcept {
    {
      std::lock_guard lock{mutex_};
      shutdown_ = true;
    }
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
  }

  // ------------------------------------------------------------------
  // Observers (const, may be stale by the time caller uses the result)
  // ------------------------------------------------------------------

  [[nodiscard]] std::size_t size() const noexcept {
    std::lock_guard lock{mutex_};
    return queue_.size();
  }

  [[nodiscard]] bool empty() const noexcept {
    std::lock_guard lock{mutex_};
    return queue_.empty();
  }

  [[nodiscard]] bool is_shutdown() const noexcept {
    return shutdown_.load(std::memory_order_relaxed);
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable not_empty_cv_;
  std::condition_variable not_full_cv_;
  std::deque<VehicleMeasurement> queue_;
  std::atomic<bool> shutdown_{false};
};

// ===========================================================================
// LockFreeMeasurementQueue  —  SPSC ring buffer (no-block consumer path)
// ===========================================================================

/**
 * @brief Single-Producer / Single-Consumer (SPSC) lock-free ring buffer.
 *
 * Guarantees
 * ----------
 * - Zero mutex contention between the producer and consumer.
 * - Cache-friendly: producer and consumer indices sit on separate cache-
 *   line-aligned storage to eliminate false sharing (requires C++17
 *   `alignas` with `hardware_destructive_interference_size`).
 * - `push` is wait-free (never blocks); returns `false` if full.
 * - `pop`  is wait-free (never blocks); returns `std::nullopt` if empty.
 *
 * @warning  This class is correct ONLY when exactly ONE thread calls
 *           `push` and exactly ONE thread calls `pop`.  MPSC / MPMC
 *           usage is undefined behaviour.
 *
 * @tparam Capacity  Size of the ring buffer.  Must be a power of two
 *                   for efficient modulo via bitwise AND.
 */
template <std::size_t Capacity = 128> class LockFreeMeasurementQueue final {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                "Capacity must be a power of two and at least 2.");

  // Align to cache-line boundary to avoid false sharing.
  static constexpr std::size_t kCacheLine =
#if defined(__cpp_lib_hardware_interference_size)
      std::hardware_destructive_interference_size;
#else
      64u;
#endif

public:
  LockFreeMeasurementQueue() : buffer_(Capacity), head_(0), tail_(0) {}

  ~LockFreeMeasurementQueue() = default;

  LockFreeMeasurementQueue(const LockFreeMeasurementQueue &) = delete;
  LockFreeMeasurementQueue &
  operator=(const LockFreeMeasurementQueue &) = delete;
  LockFreeMeasurementQueue(LockFreeMeasurementQueue &&) = delete;
  LockFreeMeasurementQueue &operator=(LockFreeMeasurementQueue &&) = delete;

  // ------------------------------------------------------------------
  // Producer API  (call from a SINGLE producer thread only)
  // ------------------------------------------------------------------

  /**
   * @brief Attempt to enqueue.  Returns `false` if the buffer is full.
   *
   * Uses `std::memory_order_release` on the head store so that the
   * consumer always sees a fully constructed item.
   */
  [[nodiscard]] bool push(const VehicleMeasurement &item) noexcept {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t next = (head + 1u) & (Capacity - 1u);
    if (next == tail_.load(std::memory_order_acquire)) {
      return false; // full
    }
    buffer_[head] = item;
    head_.store(next, std::memory_order_release);
    return true;
  }

  // ------------------------------------------------------------------
  // Consumer API  (call from a SINGLE consumer thread only)
  // ------------------------------------------------------------------

  /**
   * @brief Attempt to dequeue.  Returns `std::nullopt` if the buffer is empty.
   *
   * Uses `std::memory_order_acquire` on the head load so that the
   * consumer observes the fully written item before reading it.
   */
  [[nodiscard]] std::optional<VehicleMeasurement> pop() noexcept {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      return std::nullopt; // empty
    }
    VehicleMeasurement item = buffer_[tail];
    tail_.store((tail + 1u) & (Capacity - 1u), std::memory_order_release);
    return item;
  }

  // ------------------------------------------------------------------
  // Observers
  // ------------------------------------------------------------------

  [[nodiscard]] bool empty() const noexcept {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

private:
  std::vector<VehicleMeasurement> buffer_;

  alignas(kCacheLine) std::atomic<std::size_t> head_;
  alignas(kCacheLine) std::atomic<std::size_t> tail_;
};

} // namespace vse
