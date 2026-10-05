#include "kernels.h"

#include <algorithm>
#include <cmath>

void linear(float* y, const float* x, const float* W, const float* b, int T, int in, int out) {
  // Tiles of TB positions x OB outputs: the OB rows of W stay in cache while the TB rows of x stream past.
  // Each output is one dot product, computed the same way whatever T is, so a decode step (T = 1) gives
  // bit-identical results to the same position inside a longer sequence.
  constexpr int TB = 64, OB = 16;
  const int nt = (T + TB - 1) / TB, no = (out + OB - 1) / OB;
#pragma omp parallel for collapse(2) schedule(static)
  for (int it = 0; it < nt; ++it)
    for (int io = 0; io < no; ++io) {
      const int t1 = std::min(T, (it + 1) * TB), o1 = std::min(out, (io + 1) * OB);
      for (int t = it * TB; t < t1; ++t) {
        const float* xt = x + (std::size_t)t * in;
        float* yt = y + (std::size_t)t * out;
        for (int o = io * OB; o < o1; ++o) yt[o] = dot(xt, W + (std::size_t)o * in, in) + (b ? b[o] : 0.f);
      }
    }
}

void layernorm(float* y, const float* x, const float* w, const float* b, int T, int d) {
#pragma omp parallel for schedule(static)
  for (int t = 0; t < T; ++t) {
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
}

void gelu(float* x, std::size_t n) {
  const float c = std::sqrt(2.0f / 3.14159265358979f);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i) {
    const float v = x[i];
    x[i] = 0.5f * v * (1.f + std::tanh(c * (v + 0.044715f * v * v * v)));
  }
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
