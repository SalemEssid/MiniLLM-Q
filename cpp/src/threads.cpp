#include "threads.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

class Pool {
 public:
  explicit Pool(int n) { start(n); }
  ~Pool() { stop(); }

  int size() const { return (int)workers_.size() + 1; }

  void resize(int n) {
    if (n == size()) return;
    stop();
    start(n);
  }

  void run(int n, const std::function<void(int, int)>& fn) {
    if (size() == 1 || n <= 1) {
      if (n > 0) fn(0, n);
      return;
    }
    fn_ = &fn, n_ = n;
    pending_.store(size() - 1, std::memory_order_relaxed);
    wake();  // publishes fn_ and n_ to the workers
    run_slice(0);
    while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
  }

 private:
  static constexpr int kSpins = 1 << 16;  // about 3 ms of pause instructions before sleeping

  void start(int n) {
    quit_ = false;
    // Each worker starts from the current generation, read here: a worker that read it itself could start
    // after the first run() and miss that task.
    const unsigned g = generation_.load();
    for (int id = 1; id < n; ++id) workers_.emplace_back([this, id, g] { work(id, g); });
  }

  void stop() {
    quit_ = true;
    wake();
    for (auto& w : workers_) w.join();
    workers_.clear();
  }

  // A new generation means new work (or quit). Sleeping workers are woken through the condition variable;
  // the sleepers_ count lets run() skip the mutex while every worker is still spinning. Both atomics use
  // sequentially consistent operations: a worker that registers as a sleeper after run() read sleepers_
  // is guaranteed to see the new generation in its wait predicate.
  void wake() {
    generation_.fetch_add(1);
    if (sleepers_.load() > 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_all();
    }
  }

  void run_slice(int id) {
    const int begin = (int)((long long)n_ * id / size()), end = (int)((long long)n_ * (id + 1) / size());
    if (begin < end) (*fn_)(begin, end);
  }

  void work(int id, unsigned seen) {
    for (;;) {
      for (int spins = 0; generation_.load(std::memory_order_acquire) == seen; ++spins) {
        if (spins < kSpins) {
          _mm_pause();
          continue;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        sleepers_.fetch_add(1);
        cv_.wait(lock, [&] { return generation_.load() != seen; });
        sleepers_.fetch_sub(1);
      }
      seen = generation_.load(std::memory_order_acquire);
      if (quit_) return;
      run_slice(id);
      pending_.fetch_sub(1, std::memory_order_release);
    }
  }

  std::vector<std::thread> workers_;
  std::atomic<unsigned> generation_{0};
  std::atomic<int> pending_{0}, sleepers_{0};
  std::atomic<bool> quit_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int n_ = 0;
};

Pool& pool() {
  static Pool p((int)std::max(1u, std::thread::hardware_concurrency()));
  return p;
}

}  // namespace

void set_num_threads(int n) { pool().resize(n < 1 ? 1 : n); }

int num_threads() { return pool().size(); }

void parallel_for(int n, const std::function<void(int, int)>& fn) { pool().run(n, fn); }
