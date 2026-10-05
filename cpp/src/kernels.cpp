#include "kernels.h"

#include <algorithm>
#include <cmath>

#include "threads.h"

void linear(float* y, const float* x, const float* W, const float* b, int T, int in, int out) {
  // Tiles of TB positions x OB outputs: the OB rows of W stay in cache while the TB rows of x stream past.
  // Each thread takes a contiguous run of tiles; with T = 1 that is a contiguous block of output rows.
  // Each output is one dot product, computed the same way whatever T is, so a decode step (T = 1) gives
  // bit-identical results to the same position inside a longer sequence.
  constexpr int TB = 64, OB = 16;
  const int nt = (T + TB - 1) / TB, no = (out + OB - 1) / OB;
  parallel_for(nt * no, [&](int begin, int end) {
    for (int tile = begin; tile < end; ++tile) {
      const int it = tile / no, io = tile % no;
      const int t1 = std::min(T, (it + 1) * TB), o1 = std::min(out, (io + 1) * OB);
      for (int t = it * TB; t < t1; ++t) {
        const float* xt = x + (std::size_t)t * in;
        float* yt = y + (std::size_t)t * out;
        for (int o = io * OB; o < o1; ++o) yt[o] = dot(xt, W + (std::size_t)o * in, in) + (b ? b[o] : 0.f);
      }
    }
  });
}

void layernorm(float* y, const float* x, const float* w, const float* b, int T, int d) {
  parallel_for(T, [&](int begin, int end) {
    for (int t = begin; t < end; ++t) {
      const float* xt = x + (std::size_t)t * d;
      float* yt = y + (std::size_t)t * d;
      // Statistics in double: the GPT-2 residual stream has a few channels in the hundreds or thousands.
      double mean = 0, var = 0;
      for (int i = 0; i < d; ++i) mean += xt[i];
      mean /= d;
      for (int i = 0; i < d; ++i) var += (xt[i] - mean) * (xt[i] - mean);
      const double rstd = 1.0 / std::sqrt(var / d + 1e-5);
      for (int i = 0; i < d; ++i) yt[i] = (float)((xt[i] - mean) * rstd) * w[i] + b[i];
    }
  });
}

void gelu(float* x, std::size_t n) {
  const float c = std::sqrt(2.0f / 3.14159265358979f);
  constexpr int kChunk = 1024;  // one task per 1024 values: tanh is slow enough to share a decode step
  parallel_for((int)((n + kChunk - 1) / kChunk), [&](int begin, int end) {
    const std::size_t i1 = std::min(n, (std::size_t)end * kChunk);
    for (std::size_t i = (std::size_t)begin * kChunk; i < i1; ++i) {
      const float v = x[i];
      x[i] = 0.5f * v * (1.f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
  });
}

void softmax(float* x, int n) {
  const float mx = *std::max_element(x, x + n);
  float sum = 0.f;
  for (int i = 0; i < n; ++i) sum += (x[i] = std::exp(x[i] - mx));
  const float inv = 1.f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void add(float* y, const float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] += x[i];
}
