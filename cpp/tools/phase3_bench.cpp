// Phase 3 microbenchmarks, for each kernel family and width:
//   compute rate C_f(b): weights per second when every thread's slice of the matrix sits in its L2 cache;
//   from DRAM: the 48 block modules of one decode step, back to back, as effective GB/s and weights/s;
//   bound: the per-module roofline of the guide (Section 5.2), sum over modules of
//          max(bytes / BW, weights / C_f(b)), which the two measurements predict.
// With --decode, also decode speed with every block module at that width and the head in FP32 (Phase 2
// protocol). Usage: phase3_bench [project root] [--threads 8] [--decode]. Writes results/phase3_bench.json.
#include "bench_env.h"  // includes windows.h
#include "quant.h"
#include "threads.h"
#include "tool_util.h"

namespace {

constexpr int kD = 768, kVocab = 50257;  // GPT-2 small, as exported
constexpr int kColumnsPerThread = 512;   // 768 x 512 at 8 bits is 400 KB: two threads per core fit in L2
constexpr int kTimedRuns = 5;

// A matrix with fixed pseudo-random codes, for the compute rate.
QLinear synthetic(int in, int out, int bits, Family family) {
  std::vector<uint8_t> codes((std::size_t)in * out);
  uint32_t s = 12345;
  for (auto& c : codes) s = s * 1664525u + 1013904223u, c = (uint8_t)((s >> 24) & ((1 << bits) - 1));
  std::vector<uint16_t> scale((std::size_t)in / kGroup * out, 0x2000), minv(scale.size(), 0xA000);  // FP16 2^-7, -2^-7
  return pack(codes.data(), scale.data(), minv.data(), in, out, bits, family);
}

// Seconds per call of f: the best of 5 rounds, each repeating f for at least 0.2 s.
template <class F>
double seconds_per_call(F&& f) {
  f();  // warm-up
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    int n = 0;
    double dt;
    const double t0 = now_s();
    do f(), ++n, dt = now_s() - t0;
    while (dt < 0.2);
    best = std::min(best, dt / n);
  }
  return best;
}

struct Row {
  Family family;
  int bits;
  double compute_gw = 0, dram_gbs = 0, dram_gw = 0, bound_gw = 0, decode_mean = 0, decode_std = 0;
};

}  // namespace

int main(int argc, char** argv) try {
  fs::path root = MINILLM_ROOT;
  int threads = 8;  // fixed by the Phase 2 sweep
  bool decode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    else if (a == "--decode") decode = true;
    else root = a;
  }
  const fs::path data = root / "data";
  const BenchEnv env = prepare_benchmark_process();
  const SystemState sys = system_state();
  std::printf("%s, %s, %.1f GB RAM free, power throttling %s, %d threads\n", cpu_name().c_str(),
              sys.on_ac ? "on mains power" : "ON BATTERY", sys.free_ram_gb,
              env.throttling_disabled ? "disabled" : "NOT disabled", threads);

  const double bw = measure_read_bandwidth(std::size_t{256} << 20, threads).best_gbs;
  set_num_threads(threads);
  std::printf("read bandwidth %.1f GB/s\n", bw);

  std::vector<float> x768(768), x3072(3072), y(std::max(kVocab, kColumnsPerThread * threads));
  for (int i = 0; i < 3072; ++i) x3072[i] = std::sin(0.1f * i), x768[i % 768] = std::cos(0.07f * i);
  const auto prompt = read_tokens(data / "ref_prompt.u16");

  std::vector<Row> rows;
  std::printf("\nfamily bits | compute rate | from DRAM: GB/s  Gweights/s | bound Gweights/s | DRAM/bound%s\n",
              decode ? " | decode tokens/s" : "");
  for (Family family : {Family::Unpack, Family::BitPlane})
    for (int bits : {2, 3, 4, 6, 8}) {
      Row r{family, bits};
      const QLinear syn = synthetic(kD, kColumnsPerThread * threads, bits, family);
      r.compute_gw = (double)syn.in * syn.out / seconds_per_call([&] { qmatvec(syn, x768.data(), y.data()); }) / 1e9;

      std::vector<QLinear> modules;
      double bytes = 0, weights = 0, t_bound = 0;
      for (int i = 0; i < 48; ++i) {
        modules.push_back(load_qlinear(data.string(), bits, i, kD, kVocab, family));
        const QLinear& m = modules.back();
        bytes += qbytes(m), weights += (double)m.in * m.out;
        t_bound += std::max(qbytes(m) / (bw * 1e9), (double)m.in * m.out / (r.compute_gw * 1e9));
      }
      const double t = seconds_per_call([&] {
        for (const QLinear& m : modules) qmatvec(m, m.in == 768 ? x768.data() : x3072.data(), y.data());
      });
      r.dram_gbs = bytes / t / 1e9, r.dram_gw = weights / t / 1e9, r.bound_gw = weights / t_bound / 1e9;
      modules.clear();

      if (decode) {
        GPT2 model((data / "gpt2_124M.bin").string(), Quantization{std::vector<int>(48, bits), 0, family, data.string()});
        std::vector<float> logits(prompt.size() * (std::size_t)kVocab);
        timed_decode(model, prompt, logits);  // warm-up
        std::vector<double> tps;
        for (int k = 0; k < kTimedRuns; ++k) tps.push_back(timed_decode(model, prompt, logits).tokens_per_s);
        for (double v : tps) r.decode_mean += v / kTimedRuns;
        for (double v : tps) r.decode_std += (v - r.decode_mean) * (v - r.decode_mean) / (kTimedRuns - 1);
        r.decode_std = std::sqrt(r.decode_std);
      }
      rows.push_back(r);
      std::printf("   %s     %d  |  %6.1f G/s   |      %5.1f      %6.1f     |      %6.1f      |   %3.0f%%", family_name(family),
                  bits, r.compute_gw, r.dram_gbs, r.dram_gw, r.bound_gw, 100 * r.dram_gw / r.bound_gw);
      if (decode) std::printf("     | %5.1f +- %.1f", r.decode_mean, r.decode_std);
      std::printf("\n");
    }

  fs::create_directories(root / "results");
  std::ofstream js(root / "results" / "phase3_bench.json");
  js.precision(6);
  js << std::boolalpha << "{\n  \"cpu\": \"" << cpu_name() << "\",\n  \"on_mains_power\": " << sys.on_ac
     << ",\n  \"free_ram_gb\": " << sys.free_ram_gb << ",\n  \"power_throttling_disabled\": " << env.throttling_disabled
     << ",\n  \"threads\": " << threads << ",\n  \"bandwidth_gbs\": " << bw << ",\n  \"rows\": [";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    js << (i ? ",\n    " : "\n    ") << "{\"family\": \"" << family_name(r.family) << "\", \"bits\": " << r.bits
       << ", \"compute_gweights_per_s\": " << r.compute_gw << ", \"dram_gb_per_s\": " << r.dram_gbs
       << ", \"dram_gweights_per_s\": " << r.dram_gw << ", \"bound_gweights_per_s\": " << r.bound_gw;
    if (decode) js << ", \"decode_tokens_per_s\": " << r.decode_mean << ", \"decode_std\": " << r.decode_std;
    js << "}";
  }
  js << "\n  ]\n}\n";
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
