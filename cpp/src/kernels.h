#pragma once
// FP32 kernels for the GPT-2 forward pass. All matrices are row-major; T is the number of positions
// processed together (T = 1 for a decode step).
#include <cstddef>

// Dot product of two length-n vectors. The simd reduction lets the compiler vectorize the sum.
inline float dot(const float* a, const float* b, int n) {
  float acc = 0.f;
#pragma omp simd reduction(+ : acc)
  for (int i = 0; i < n; ++i) acc += a[i] * b[i];
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
