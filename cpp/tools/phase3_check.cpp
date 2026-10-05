// Phase 3 gate: the quantized kernels against Python.
//   (a) For every width and both families, the C++ product on the exported codes matches x @ fake_quant(W)
//       from Python with ||dy|| / ||y|| < 1e-5. The scalar reference kernel, which decodes every code from
//       the packed layout and sums in double, must match to 1e-6: that checks the export and the packing.
//   (b) All 48 block modules at 8 bits, head in FP32: the C++ perplexity matches the Python fake-quant model
//       to 0.1%, with both families.
// Usage: phase3_check [project root] [--skip-ppl] [--ppl-widths 8,4]. Writes results/phase3.json.
#include <map>

#include "quant.h"
#include "tool_util.h"

namespace {

constexpr int kD = 768, kVocab = 50257;  // GPT-2 small, as exported

double rel_error(const std::vector<float>& y, const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < y.size(); ++i) num += (y[i] - ref[i]) * (y[i] - ref[i]), den += ref[i] * ref[i];
  return std::sqrt(num / den);
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) v.push_back(std::stoi(item));
  return v;
}

struct KernelCase {
  Family family;
  int index, bits;
  double err_reference, err_kernel;
  bool pass;
};

struct PplCase {
  Family family;
  int bits;
  double ppl, ppl_python, rel, seconds;
  bool pass;
};

}  // namespace

int main(int argc, char** argv) try {
  fs::path root = MINILLM_ROOT;
  bool skip_ppl = false;
  std::vector<int> ppl_widths = {8};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--skip-ppl") skip_ppl = true;
    else if (a == "--ppl-widths" && i + 1 < argc) ppl_widths = parse_list(argv[++i]);
    else root = a;
  }
  const fs::path data = root / "data", qref = data / "qref";
  const Family families[] = {Family::Unpack, Family::BitPlane};
#ifdef __AVX512F__
  std::printf("kernels: AVX-512\n");
#else
  std::printf("kernels: scalar reference only (no AVX-512 in this build)\n");
#endif

  // (a) One matrix-vector product per module, width and family, against Python's float64 result.
  const std::map<int, std::vector<float>> xs = {{768, read_array<float>(qref / "x768.f32")},
                                                {3072, read_array<float>(qref / "x3072.f32")}};
  std::vector<KernelCase> kernel_cases;
  bool pass_a = true;
  std::printf("(a) ||y - y_python|| / ||y_python||   (reference < 1e-6, kernel < 1e-5)\n");
  for (Family family : families)
    for (int index : {2, 3, 44, 45, 48})
      for (int bits : {2, 3, 4, 6, 8}) {
        if (index == 48 && bits != 8) continue;  // the head is exported at 8 bits only
        const QLinear q = load_qlinear(data.string(), bits, index, kD, kVocab, family);
        const auto y_python =
            read_array<double>(qref / ("y_m" + std::to_string(index) + "_q" + std::to_string(bits) + ".f64"));
        std::vector<float> y_kernel(q.out), y_reference(q.out);
        qmatvec(q, xs.at(q.in).data(), y_kernel.data());
        qmatvec_reference(q, xs.at(q.in).data(), y_reference.data());
        KernelCase k{family, index, bits, rel_error(y_reference, y_python), rel_error(y_kernel, y_python), false};
        k.pass = k.err_reference < 1e-6 && k.err_kernel < 1e-5;
        pass_a = pass_a && k.pass;
        kernel_cases.push_back(k);
        std::printf("    %s  module %2d (%4d x %5d)  %d-bit   reference %.1e   kernel %.1e   %s\n", family_name(family),
                    index, q.in, q.out, bits, k.err_reference, k.err_kernel, verdict(k.pass));
      }
  std::printf("(a) %s\n", verdict(pass_a));

  // (b) Perplexity with every block module quantized, against the Python fake-quant model.
  std::vector<PplCase> ppl_cases;
  bool pass_b = true;
  if (!skip_ppl) {
    const auto test = read_tokens(data / "wikitext2_test.u16");
    for (Family family : families)
      for (int bits : ppl_widths) {
        GPT2 model((data / "gpt2_124M.bin").string(),
                   Quantization{std::vector<int>(48, bits), 0, family, data.string()});
        std::vector<double> window_nll;
        const auto t0 = std::chrono::steady_clock::now();
        PplCase p{family, bits, perplexity(model, test, 1024, 32, window_nll),
                  json_number(data / "quant_reference.json", "ppl_q" + std::to_string(bits)), 0,
                  seconds_since(t0), false};
        p.rel = std::fabs(p.ppl - p.ppl_python) / p.ppl_python;
        p.pass = p.rel < 1e-3;
        pass_b = pass_b && p.pass;
        ppl_cases.push_back(p);
        std::printf("(b) %s  family %s, all blocks %d-bit: perplexity %.4f vs %.4f in Python (relative %.1e), %.0f s\n",
                    verdict(p.pass), family_name(family), bits, p.ppl, p.ppl_python, p.rel, p.seconds);
      }
  }

  const bool pass = pass_a && pass_b && !skip_ppl;
  std::printf("Phase 3 gate (a), (b): %s%s\n", verdict(pass), skip_ppl ? " (perplexity skipped)" : "");

  fs::create_directories(root / "results");
  std::ofstream js(root / "results" / "phase3.json");
  js.precision(8);
  js << std::boolalpha << "{\n  \"pass\": " << pass << ",\n  \"threads\": " << num_threads()
     << ",\n  \"a_kernels\": {\"pass\": " << pass_a << ", \"cases\": [";
  for (std::size_t i = 0; i < kernel_cases.size(); ++i) {
    const KernelCase& k = kernel_cases[i];
    js << (i ? ",\n    " : "\n    ") << "{\"family\": \"" << family_name(k.family) << "\", \"module\": " << k.index
       << ", \"bits\": " << k.bits << ", \"err_reference\": " << k.err_reference << ", \"err_kernel\": "
       << k.err_kernel << ", \"pass\": " << k.pass << "}";
  }
  js << "\n  ]},\n  \"b_perplexity\": {\"pass\": " << (pass_b && !skip_ppl) << ", \"cases\": [";
  for (std::size_t i = 0; i < ppl_cases.size(); ++i) {
    const PplCase& p = ppl_cases[i];
    js << (i ? ",\n    " : "\n    ") << "{\"family\": \"" << family_name(p.family) << "\", \"bits\": " << p.bits
       << ", \"ppl\": " << p.ppl << ", \"ppl_python\": " << p.ppl_python << ", \"relative_difference\": " << p.rel
       << ", \"seconds\": " << p.seconds << ", \"pass\": " << p.pass << "}";
  }
  js << "\n  ]}\n}\n";
  return pass ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
