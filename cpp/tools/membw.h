#pragma once
// STREAM-like read bandwidth: every thread sums its own contiguous slice of a large array.
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <new>

// Wall time in seconds. omp_get_wtime() has only 1 ms resolution with MinGW's libgomp; steady_clock is
// QueryPerformanceCounter on Windows.
inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Bandwidth {
  int threads;
  double best_gbs, mean_gbs;
};

// Four independent accumulators, so the sum is limited by loads, not by the latency of the adds.
inline float sum_slice(const float* p, std::size_t n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i));
    a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
    a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16));
    a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  float s = 0.f;
  for (float v : lanes) s += v;
  for (; i < n; ++i) s += p[i];
  return s;
}

// Reads `bytes` of memory `runs` times with `threads` threads; returns the best and mean GB/s.
inline Bandwidth measure_read_bandwidth(std::size_t bytes, int threads, int runs = 10) {
  const std::size_t n = bytes / sizeof(float) / 32 * 32;
  float* a = static_cast<float*>(::operator new(n * sizeof(float), std::align_val_t{64}));
  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i) a[i] = 1.0f;  // first touch by the threads that read

  double best = 0, sum = 0;
  volatile float sink = 0;
  for (int r = 0; r < runs; ++r) {
    const double t0 = now_s();
    float total = 0;
#pragma omp parallel reduction(+ : total)
    {
      const std::size_t nt = omp_get_num_threads(), id = omp_get_thread_num();
      const std::size_t chunk = n / nt / 32 * 32, begin = id * chunk, end = id + 1 == nt ? n : begin + chunk;
      total += sum_slice(a + begin, end - begin);
    }
    const double gbs = n * sizeof(float) / (now_s() - t0) / 1e9;
    best = std::max(best, gbs), sum += gbs;
    sink = sink + total;
  }
  ::operator delete(a, std::align_val_t{64});
  return {threads, best, sum / runs};
}
