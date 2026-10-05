#pragma once
// Shared by the benchmarks: process set-up, a record of the machine's state, and the decode protocol.
//
// Benchmark hygiene on Windows. Windows 11 runs processes it considers background at efficient clocks
// (power throttling, "EcoQoS"), especially on battery, and a benchmark launched from an editor or a script
// counts as background. Opt out of it, and raise the priority so background work preempts the worker
// threads less often: a preempted worker stalls the whole matrix-vector product it belongs to.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cpuid.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "gpt2.h"
#include "membw.h"

struct BenchEnv {
  bool throttling_disabled = false, high_priority = false;
};

inline BenchEnv prepare_benchmark_process() {
  BenchEnv env;
#ifdef _WIN32
  PROCESS_POWER_THROTTLING_STATE s{};
  s.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  s.StateMask = 0;  // execution speed is controlled by this process: never throttled
  env.throttling_disabled = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof s);
  env.high_priority = SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
  return env;
}

inline std::string cpu_name() {
  unsigned r[12] = {};
  for (unsigned i = 0; i < 3; ++i) __get_cpuid(0x80000002 + i, &r[4 * i], &r[4 * i + 1], &r[4 * i + 2], &r[4 * i + 3]);
  std::string s(reinterpret_cast<const char*>(r), sizeof r);
  s = s.c_str();
  return s.substr(s.find_first_not_of(' '));
}

struct SystemState {
  bool on_ac = true;
  double free_ram_gb = 0;
};

inline SystemState system_state() {
  SystemState s;
#ifdef _WIN32
  SYSTEM_POWER_STATUS ps;
  if (GetSystemPowerStatus(&ps)) s.on_ac = ps.ACLineStatus == 1;
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof ms;
  if (GlobalMemoryStatusEx(&ms)) s.free_ram_gb = ms.ullAvailPhys / 1e9;
#endif
  return s;
}

inline std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) v.push_back(std::stoi(item));
  return v;
}

// The decode protocol of the guide (Phase 2 gate): prefill the prompt, then time kDecodeSteps greedy steps.
constexpr int kDecodeSteps = 256;

struct DecodeRun {
  double tokens_per_s, linear_frac, attention_frac;
};

inline DecodeRun timed_decode(GPT2& model, const std::vector<int>& prompt, std::vector<float>& logits) {
  const int V = model.config().vocab;
  model.forward_sequence(prompt.data(), (int)prompt.size(), logits.data());
  int next = (int)(std::max_element(logits.begin() + (prompt.size() - 1) * V, logits.begin() + prompt.size() * V) -
                   logits.begin() - (prompt.size() - 1) * V);
  model.reset_timings();
  const double t0 = now_s();
  for (int i = 0; i < kDecodeSteps; ++i) {
    model.decode_step(next, logits.data());
    next = (int)(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());
  }
  const double dt = now_s() - t0;
  const auto& t = model.timings();
  return {kDecodeSteps / dt, t.linear / t.total, t.attention / t.total};
}
