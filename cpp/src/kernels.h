#pragma once
// FP32 kernels for the GPT-2 forward pass. All matrices are row-major; T is the number of positions
// processed together (T = 1 for a decode step).
#include <immintrin.h>

#include <cstddef>

// Dot product of two length-n vectors. With AVX2, four independent accumulators of 8 floats: a single
// accumulator would make every FMA wait for the previous one (4 cycles), far slower than memory.
inline float dot(const float* a, const float* b, int n) {
  int i = 0;
  float acc = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  const __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_movehdup_ps(h));
  acc = _mm_cvtss_f32(h);
#endif
  for (; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

// y[T, out] = x[T, in] @ W^T + b, with W stored [out, in] (one contiguous row per output); b may be null.
void linear(float* y, const float* x, const float* W, const float* b, int T, int in, int out);

// y[T, d] = (x - mean) / sqrt(var + 1e-5) * w + b over each row of x.
void layernorm(float* y, const float* x, const float* w, const float* b, int T, int d);

// In place: x = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the GELU approximation GPT-2 uses.
void gelu(float* x, std::size_t n);

// In place: x = exp(x - max x) / sum.
void softmax(float* x, int n);

// y += x
void add(float* y, const float* x, std::size_t n);
