#pragma once
// Helpers shared by the gate tools: reading the binary files of data/, a minimal JSON number lookup, and
// the perplexity protocol of the Phase 1 reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpt2.h"
#include "threads.h"

namespace fs = std::filesystem;

template <class T>
std::vector<T> read_array(const fs::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  std::vector<T> v((std::size_t)f.tellg() / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
  return v;
}

inline std::vector<int> read_tokens(const fs::path& path) {
  auto u16 = read_array<uint16_t>(path);
  return {u16.begin(), u16.end()};
}

// The number after "key": in a JSON file. Enough for the flat numeric fields of the reference files.
inline double json_number(const fs::path& path, const std::string& key) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str(), pat = "\"" + key + "\":";
  const auto at = text.find(pat);
  if (at == std::string::npos) throw std::runtime_error(key + " not found in " + path.string());
  return std::stod(text.substr(at + pat.size()));
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline const char* verdict(bool ok) { return ok ? "PASS" : "FAIL"; }

// -log softmax(x)[target], in double.
inline double nll(const float* x, int n, int target) {
  const double mx = *std::max_element(x, x + n);
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += std::exp(x[i] - mx);
  return mx + std::log(sum) - x[target];
}

// Perplexity on n_windows non-overlapping windows of `window` tokens from the start of `tokens`, the
// protocol of the Phase 1 reference: each window scores its window-1 next-token predictions. Fills
// window_nll with each window's mean NLL. Sums run in a fixed order, so the result is deterministic.
inline double perplexity(GPT2& model, const std::vector<int>& tokens, int window, int n_windows,
                         std::vector<double>& window_nll) {
  const int V = model.config().vocab;
  std::vector<float> logits((std::size_t)window * V);
  std::vector<double> position_nll(window - 1);
  window_nll.clear();
  const auto t0 = std::chrono::steady_clock::now();
  for (int w = 0; w < n_windows; ++w) {
    const int* tok = &tokens[(std::size_t)w * window];
    model.forward_sequence(tok, window, logits.data());
    parallel_for(window - 1, [&](int b, int e) {
      for (int t = b; t < e; ++t) position_nll[t] = nll(&logits[(std::size_t)t * V], V, tok[t + 1]);
    });
    double sum = 0;
    for (double x : position_nll) sum += x;
    window_nll.push_back(sum / (window - 1));
    std::printf("\r    window %2d/%d  %.0f s", w + 1, n_windows, seconds_since(t0));
    std::fflush(stdout);
  }
  std::printf("\r%40s\r", "");
  double mean = 0;
  for (double x : window_nll) mean += x / n_windows;
  return std::exp(mean);
}
