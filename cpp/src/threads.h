#pragma once
// A fixed pool of worker threads for the forward pass.
//
// Why not OpenMP: with MinGW's libgomp on Windows, entering a parallel region costs 40-140 us and a barrier
// 20-170 us (measured on the i5-1135G7), and a decode step enters about 100 regions. Workers that spin
// between tasks start in about a microsecond. They spin for a few milliseconds after the last task, then
// sleep, so an idle pool does not burn the CPU.
#include <functional>

// Total number of threads, the calling thread included. The default is std::thread::hardware_concurrency().
void set_num_threads(int n);
int num_threads();

// Splits [0, n) into one contiguous slice per thread, runs fn(begin, end) on every non-empty slice (the
// caller takes the first), and returns when all slices are done. With n <= 1 the caller runs it alone.
void parallel_for(int n, const std::function<void(int, int)>& fn);
