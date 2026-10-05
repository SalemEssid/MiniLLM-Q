// Phase 2: FP32 decode speed against the bandwidth roofline.
// Protocol (guide, Phase 2 gate): prompt of 64 tokens, 256 decode steps, 1 warm-up and 5 timed runs per
// thread count, mean +- std. Also records the CPU, the power source and the free memory.
// Usage: phase2_bench [project root] [--threads 1,2,4,8] [--runs 5]. Writes results/phase2.json.
#include <omp.h>

#include "bench_env.h"  // includes windows.h
#include "tool_util.h"

namespace {

constexpr std::size_t kBandwidthBytes = std::size_t{256} << 20;

struct Result {
  int threads;
  double mean, std, linear_frac, attention_frac;
  std::vector<double> runs;  // tokens/s of every timed run, to spot outliers
};

}  // namespace

int main(int argc, char** argv) try {
  fs::path root = MINILLM_ROOT;
  std::vector<int> thread_counts;
  for (int t = 1; t <= omp_get_num_procs(); t *= 2) thread_counts.push_back(t);
  int runs = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) thread_counts = parse_int_list(argv[++i]);
    else if (a == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
    else root = a;
  }

  const BenchEnv env = prepare_benchmark_process();
  const SystemState sys = system_state();
  const std::string cpu = cpu_name();
  std::printf("%s, %d logical CPUs, %s, %.1f GB RAM free\n", cpu.c_str(), omp_get_num_procs(),
              sys.on_ac ? "on mains power" : "ON BATTERY (the gate needs mains power)", sys.free_ram_gb);
  std::printf("power throttling %s, priority %s\n", env.throttling_disabled ? "disabled" : "NOT disabled",
              env.high_priority ? "high" : "normal");

  // Bandwidth first, before the model takes 0.5 GB of the little free memory.
  double bw = 0;
  std::vector<Bandwidth> bws;
  for (int t : thread_counts) bws.push_back(measure_read_bandwidth(kBandwidthBytes, t)), bw = std::max(bw, bws.back().best_gbs);
  std::printf("read bandwidth (256 MB, best of 10):");
  for (const auto& b : bws) std::printf("  %dT %.1f", b.threads, b.best_gbs);
  std::printf(" GB/s\n");

  GPT2 model((root / "data" / "gpt2_124M.bin").string());
  const Config c = model.config();
  const auto prompt = read_tokens(root / "data" / "ref_prompt.u16");
  const int P = (int)prompt.size();
  std::vector<float> logits((std::size_t)P * c.vocab);

  // Bytes read per decode step: the parameters (one row of wte and wpe), plus the KV cache. A step at
  // position p reads K and V for positions 0..p; the timed steps run at positions P..P+kDecodeSteps-1.
  const double param_bytes = model.parameter_bytes_per_token();
  const double kv_bytes = 2.0 * c.n_layer * (P + (kDecodeSteps + 1) / 2.0) * c.d_model * 4;
  const double bytes_per_token = param_bytes + kv_bytes;
  const double ceiling = bw / (bytes_per_token / 1e9);
  std::printf("bytes per token: %.1f MB weights + %.1f MB KV cache; roofline ceiling %.1f tokens/s\n",
              param_bytes / 1e6, kv_bytes / 1e6, ceiling);

  std::vector<Result> results;
  for (int t : thread_counts) {
    set_num_threads(t);
    timed_decode(model, prompt, logits);  // warm-up
    std::vector<DecodeRun> rs;
    for (int r = 0; r < runs; ++r) rs.push_back(timed_decode(model, prompt, logits));
    double mean = 0, var = 0, lin = 0, att = 0;
    for (const auto& r : rs) mean += r.tokens_per_s / runs, lin += r.linear_frac / runs, att += r.attention_frac / runs;
    for (const auto& r : rs) var += (r.tokens_per_s - mean) * (r.tokens_per_s - mean) / std::max(1, runs - 1);
    std::vector<double> per_run;
    for (const auto& r : rs) per_run.push_back(r.tokens_per_s);
    results.push_back({t, mean, std::sqrt(var), lin, att, per_run});
    std::printf("  %d threads: %5.1f +- %.1f tokens/s = %4.1f%% of the ceiling  (linear %.0f%%, attention %.0f%%, "
                "other %.0f%%)\n", t, mean, std::sqrt(var), 100 * mean / ceiling, 100 * lin, 100 * att,
                100 * (1 - lin - att));
  }
  const Result best = *std::max_element(results.begin(), results.end(),
                                        [](const Result& a, const Result& b) { return a.mean < b.mean; });
  std::printf("best: %d threads, %.1f tokens/s = %.1f%% of the roofline (gate: 50-70%%)\n", best.threads,
              best.mean, 100 * best.mean / ceiling);

  fs::create_directories(root / "results");
  std::ofstream js(root / "results" / "phase2.json");
  js.precision(6);
  js << std::boolalpha << "{\n"
     << "  \"cpu\": \"" << cpu << "\",\n  \"logical_cpus\": " << omp_get_num_procs()
     << ",\n  \"on_mains_power\": " << sys.on_ac << ",\n  \"free_ram_gb\": " << sys.free_ram_gb
     << ",\n  \"power_throttling_disabled\": " << env.throttling_disabled
     << ",\n  \"high_priority\": " << env.high_priority
     << ",\n  \"protocol\": {\"prompt_tokens\": " << P << ", \"decode_steps\": " << kDecodeSteps
     << ", \"warmup_runs\": 1, \"timed_runs\": " << runs << "},\n  \"bandwidth_gbs\": {";
  for (std::size_t i = 0; i < bws.size(); ++i) js << (i ? ", " : "") << "\"" << bws[i].threads << "\": " << bws[i].best_gbs;
  js << "},\n  \"bytes_per_token\": " << bytes_per_token << ",\n  \"roofline_tokens_per_s\": " << ceiling
     << ",\n  \"decode\": [";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const Result& r = results[i];
    js << (i ? ",\n    " : "\n    ") << "{\"threads\": " << r.threads << ", \"tokens_per_s\": " << r.mean
       << ", \"std\": " << r.std << ", \"fraction_of_roofline\": " << r.mean / ceiling
       << ", \"linear_share\": " << r.linear_frac << ", \"attention_share\": " << r.attention_frac << ", \"runs\": [";
    for (std::size_t k = 0; k < r.runs.size(); ++k) js << (k ? ", " : "") << r.runs[k];
    js << "]}";
  }
  js << "\n  ],\n  \"best_threads\": " << best.threads << ",\n  \"best_fraction_of_roofline\": "
     << best.mean / ceiling << "\n}\n";
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
