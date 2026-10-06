#pragma once
// Weight-only quantized linear layers in two kernel families (guide, Phase 3).
//
// Format, shared with python/quant.py: asymmetric, groups of G = 128 input channels, W ~= s * q + m with an
// FP16 scale s and minimum m per group and output column, and codes q in 0..2^b - 1 for b in {2, 3, 4, 6, 8}.
//
// Family U (unpack-dequantize): each code is unpacked to an integer, converted to float and multiplied by x.
// Family P (bit-plane): bit i of every code is stored as its own plane, and the products come from lookup
// tables of sums of x, with no multiplications. The layouts are described in quant.cpp.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

enum class Family { Unpack, BitPlane };
constexpr int kGroup = 128;

const char* family_name(Family f);  // "U" or "P"

struct QLinear {
  int in = 0, out = 0, bits = 0;  // bits = 0: not quantized
  Family family = Family::Unpack;
  std::vector<uint8_t> packed;        // codes, in the family's layout
  std::vector<uint16_t> scale, minv;  // FP16 bits, in the family's layout
};

// Repacks codes from the export layout of python/export_quantized.py: codes [in, out], one byte each, and
// scales and minimums [in/G, out].
QLinear pack(const uint8_t* codes, const uint16_t* scale, const uint16_t* minv, int in, int out, int bits,
             Family family);

// Shape [in, out] of module `index` of the export files: 0..47 are the block modules in allocation order
// (c_attn, c_proj, c_fc, mlp.c_proj of block 0, then block 1, ...), 48 is the head (wte^T).
std::pair<int, int> module_shape(int index, int d_model, int vocab);

// Module `index` of dir/gpt2_q{bits}.bin, packed for `family`. The head exists only in the 8-bit file.
QLinear load_qlinear(const std::string& dir, int bits, int index, int d_model, int vocab, Family family);

// y[out] = x[in] @ W_hat, using the AVX-512 kernels when the build targets AVX-512, else the reference.
void qmatvec(const QLinear& w, const float* x, float* y);

// y[T, out] = x[T, in] @ W_hat for T rows at once. Every row gets exactly the arithmetic of qmatvec, so the
// results are bit-identical to T separate calls; batching only lets the codes come from DRAM once per chunk
// of rows instead of once per row.
void qmatmul(const QLinear& w, const float* x, float* y, int T);

// The same product, decoding every code from the packed layout and accumulating in double. Slow; it checks
// the packing (against Python) and the SIMD kernels (against it).
void qmatvec_reference(const QLinear& w, const float* x, float* y);

// Bytes a matrix-vector product reads: packed codes plus scales and minimums.
std::size_t qbytes(const QLinear& w);
