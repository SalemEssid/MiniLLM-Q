// Phase 2: read bandwidth for 1, 2, 4, ... threads. Usage: bandwidth [array size in MB, default 256]
#include <omp.h>

#include <cstdio>
#include <cstdlib>

#include "bench_env.h"
#include "membw.h"

int main(int argc, char** argv) {
  prepare_benchmark_process();
  const std::size_t mb = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
  std::printf("read bandwidth, %zu MB array, best and mean of 10 runs\n", mb);
  for (int t = 1; t <= omp_get_num_procs(); t *= 2) {
    const Bandwidth b = measure_read_bandwidth(mb << 20, t);
    std::printf("  %d threads: %5.1f GB/s best, %5.1f mean\n", t, b.best_gbs, b.mean_gbs);
  }
}
