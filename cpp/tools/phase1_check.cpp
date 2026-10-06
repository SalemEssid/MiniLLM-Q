// Phase 1 gate: the FP32 C++ engine against the PyTorch reference outputs of Phase 0.
//   (a) logits at all 64 prompt positions: cosine > 0.99999 and max |error| < 1e-3
//   (b) the 100 greedy tokens after the prompt are identical
//   (c) decode_step with the cache gives the same logits as forward_sequence at every position
//   (d) perplexity on the first 32K WikiText-2 test tokens matches Python to 0.1%
// Usage: phase1_check [project root] [--quick]. Writes results/phase1.json.
// --quick runs (a)-(c) only, in seconds, for use while optimizing; it writes no JSON.
#include "bench_env.h"  // includes windows.h
#include "tool_util.h"

namespace {

int argmax(const float* x, int n) { return (int)(std::max_element(x, x + n) - x); }

double max_abs_diff(const float* a, const float* b, int n) {
  double m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

double cosine(const float* a, const float* b, int n) {
  double ab = 0, aa = 0, bb = 0;
  for (int i = 0; i < n; ++i) ab += (double)a[i] * b[i], aa += (double)a[i] * a[i], bb += (double)b[i] * b[i];
  return ab / std::sqrt(aa * bb);
}

}  // namespace

int main(int argc, char** argv) try {
  fs::path root = MINILLM_ROOT;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--quick") quick = true;
    else root = argv[i];
  }
  const fs::path data = root / "data";
  prepare_benchmark_process();  // not timed, but throttled it runs at half speed on battery

  auto t0 = std::chrono::steady_clock::now();
  GPT2 model((data / "gpt2_124M.bin").string());
  const double load_s = seconds_since(t0);
  const int V = model.config().vocab;
  const auto prompt = read_tokens(data / "ref_prompt.u16");
  const auto ref_greedy = read_tokens(data / "ref_greedy.u16");
  const auto ref_logits = read_array<float>(data / "ref_logits.f32");
  const auto test = read_tokens(data / "wikitext2_test.u16");
  const double ref_ppl = json_number(data / "reference.json", "ppl");
  const int P = (int)prompt.size(), G = (int)ref_greedy.size();
  if (ref_logits.size() != (std::size_t)P * V) throw std::runtime_error("ref_logits.f32 has the wrong size");
  std::printf("loaded model in %.1f s, %d threads\n", load_s, num_threads());

  // (a) Logits at every prompt position.
  std::vector<float> logits((std::size_t)P * V);
  model.forward_sequence(prompt.data(), P, logits.data());
  double worst_cos = 1, worst_abs = 0;
  int worst_abs_pos = 0;
  for (int p = 0; p < P; ++p) {
    const float* a = &logits[(std::size_t)p * V];
    const float* b = &ref_logits[(std::size_t)p * V];
    worst_cos = std::min(worst_cos, cosine(a, b, V));
    const double e = max_abs_diff(a, b, V);
    if (e > worst_abs) worst_abs = e, worst_abs_pos = p;
  }
  const bool pass_a = worst_cos > 0.99999 && worst_abs < 1e-3;
  std::printf("(a) %s  logits: min cosine 1 - %.2e, max |error| %.2e (position %d)\n", verdict(pass_a),
              1 - worst_cos, worst_abs, worst_abs_pos);

  // (b) Greedy decoding from the prompt's cache.
  std::vector<float> step(V);
  std::vector<int> greedy;
  int next = argmax(&logits[(std::size_t)(P - 1) * V], V);
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < G; ++i) {
    greedy.push_back(next);
    if (i + 1 < G) model.decode_step(next, step.data()), next = argmax(step.data(), V);
  }
  const double decode_tps = (G - 1) / seconds_since(t0);
  int first_diff = -1;
  for (int i = 0; i < G && first_diff < 0; ++i)
    if (greedy[i] != ref_greedy[i]) first_diff = i;
  const bool pass_b = first_diff < 0;
  std::printf("(b) %s  greedy: %s (decode: %.1f tokens/s)\n", verdict(pass_b),
              pass_b ? "all 100 tokens identical" : ("first difference at step " + std::to_string(first_diff)).c_str(),
              decode_tps);

  // (c) Decode steps against one forward_sequence over prompt + greedy tokens.
  std::vector<int> seq(prompt);
  seq.insert(seq.end(), ref_greedy.begin(), ref_greedy.end());
  const int S = (int)seq.size();
  std::vector<float> seq_logits((std::size_t)S * V);
  model.forward_sequence(seq.data(), S, seq_logits.data());
  model.reset();
  double worst_cache = 0;
  for (int i = 0; i < S; ++i) {
    model.decode_step(seq[i], step.data());
    worst_cache = std::max(worst_cache, max_abs_diff(step.data(), &seq_logits[(std::size_t)i * V], V));
  }
  const bool pass_c = worst_cache < 1e-5;
  std::printf("(c) %s  cache: max |decode_step - forward_sequence| over %d positions = %.2e\n", verdict(pass_c), S,
              worst_cache);
  if (quick) {
    std::printf("quick check (a)-(c): %s\n", pass_a && pass_b && pass_c ? "PASS" : "FAIL");
    return pass_a && pass_b && pass_c ? 0 : 1;
  }

  // (d) Perplexity: 32 non-overlapping windows of 1024; each scores its 1023 next-token predictions.
  const int n_windows = 32;
  std::vector<double> window_nll;
  t0 = std::chrono::steady_clock::now();
  const double ppl = perplexity(model, test, 1024, n_windows, window_nll);
  const double ppl_s = seconds_since(t0), rel = std::fabs(ppl - ref_ppl) / ref_ppl;
  const bool pass_d = rel < 1e-3;
  std::printf("(d) %s  perplexity %.4f vs %.4f in Python (relative difference %.2e), %.0f s\n", verdict(pass_d),
              ppl, ref_ppl, rel, ppl_s);

  const bool pass = pass_a && pass_b && pass_c && pass_d;
  std::printf("Phase 1 gate: %s\n", pass ? "PASS" : "FAIL");

  fs::create_directories(root / "results");
  std::ofstream js(root / "results" / "phase1.json");
  js.precision(10);
  js << std::boolalpha << "{\n"
     << "  \"pass\": " << pass << ",\n"
     << "  \"threads\": " << num_threads() << ",\n"
     << "  \"a_logits\": {\"pass\": " << pass_a << ", \"min_cosine\": " << worst_cos
     << ", \"max_abs_error\": " << worst_abs << ", \"worst_position\": " << worst_abs_pos << "},\n"
     << "  \"b_greedy\": {\"pass\": " << pass_b << ", \"first_difference\": " << first_diff
     << ", \"decode_tokens_per_s\": " << decode_tps << "},\n"
     << "  \"c_cache\": {\"pass\": " << pass_c << ", \"positions\": " << S << ", \"max_abs_diff\": " << worst_cache
     << "},\n"
     << "  \"d_perplexity\": {\"pass\": " << pass_d << ", \"ppl\": " << ppl << ", \"ppl_python\": " << ref_ppl
     << ", \"relative_difference\": " << rel << ", \"seconds\": " << ppl_s << ", \"window_mean_nll\": [";
  for (int w = 0; w < n_windows; ++w) js << (w ? ", " : "") << window_nll[w];
  js << "]}\n}\n";
  return pass ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
