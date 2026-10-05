#include "quant.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "threads.h"

// Layouts. Both are built from the code q(j, c) of input j and output column c, in group g = j / 128.
//
// Family U: per column c and group g, one block of 16*b bytes, so the groups of a column are contiguous.
// A code is split into byte-aligned fields, low field first: 8 = 8, 6 = 4 + 2, 4 = 4, 3 = 2 + 1, 2 = 2,
// with q = low | high << (width of low). This is how llama.cpp's Q3_K and Q6_K store odd widths, and no
// field crosses a byte. A field of width w takes 16*w bytes: for m = 0..7 and lane = 0..15, the w bits of
// code j = 16m + lane sit in byte 16*(m mod w) + lane, at bit w*(m div w). One 16-byte load, one shift and
// one mask then give the field of 16 consecutive codes, for every w.
//
// Family P: per block of 16 output columns and group g, 16 quad pairs kp, each with b planes of 16 bytes
// (one byte per column): the low nibble holds bit i of the codes of inputs 8kp..8kp+3 of the group (input
// 8kp + r at bit r), the high nibble those of inputs 8kp+4..8kp+7. Scales and minimums are stored
// [block][g][16]. Columns are padded to a multiple of 16 with zero codes, scales and minimums.

namespace {

constexpr int32_t kMagic = 20261006;
constexpr std::size_t kHeaderBytes = 64;

struct Fields {
  int n, w[2];  // number of fields and their widths, low field first
};

Fields u_fields(int bits) {
  switch (bits) {
    case 8: return {1, {8, 0}};
    case 6: return {2, {4, 2}};
    case 4: return {1, {4, 0}};
    case 3: return {2, {2, 1}};
    case 2: return {1, {2, 0}};
  }
  throw std::invalid_argument("unsupported width " + std::to_string(bits));
}

// Where the field bits of code j (0..127) live in a field of width w.
void u_slot(int w, int j, int& byte, int& shift) {
  const int m = j / 16, lane = j % 16;
  byte = 16 * (m % w) + lane;
  shift = w * (m / w);
}

float half_to_float(uint16_t h) {
  const int exp = (h >> 10) & 0x1F, man = h & 0x3FF;
  float v;
  if (exp == 0) v = std::ldexp((float)man, -24);  // zero or subnormal
  else if (exp == 31) v = man ? NAN : INFINITY;
  else v = std::ldexp((float)(man | 0x400), exp - 25);
  return (h & 0x8000) ? -v : v;
}

QLinear pack_unpack(const uint8_t* codes, const uint16_t* scale, const uint16_t* minv, int in, int out,
                    int bits) {
  QLinear q;
  q.in = in, q.out = out, q.bits = bits, q.family = Family::Unpack;
  const int ng = in / kGroup;
  const Fields F = u_fields(bits);
  q.packed.assign((std::size_t)out * ng * 16 * bits, 0);
  q.scale.resize((std::size_t)out * ng), q.minv.resize((std::size_t)out * ng);
  parallel_for(out, [&](int c0, int c1) {
    for (int c = c0; c < c1; ++c)
      for (int g = 0; g < ng; ++g) {
        uint8_t* blk = q.packed.data() + ((std::size_t)c * ng + g) * 16 * bits;
        for (int j = 0; j < kGroup; ++j) {
          int v = codes[(std::size_t)(g * kGroup + j) * out + c];
          uint8_t* field = blk;
          for (int f = 0; f < F.n; ++f) {
            const int w = F.w[f];
            int byte, shift;
            u_slot(w, j, byte, shift);
            field[byte] |= (uint8_t)((v & ((1 << w) - 1)) << shift);
            v >>= w, field += 16 * w;
          }
        }
        q.scale[(std::size_t)c * ng + g] = scale[(std::size_t)g * out + c];
        q.minv[(std::size_t)c * ng + g] = minv[(std::size_t)g * out + c];
      }
  });
  return q;
}

QLinear pack_bitplane(const uint8_t* codes, const uint16_t* scale, const uint16_t* minv, int in, int out,
                      int bits) {
  QLinear q;
  q.in = in, q.out = out, q.bits = bits, q.family = Family::BitPlane;
  const int ng = in / kGroup, nb = (out + 15) / 16;
  q.packed.assign((std::size_t)nb * ng * 256 * bits, 0);  // 16 quad pairs x b planes x 16 bytes
  q.scale.assign((std::size_t)nb * ng * 16, 0), q.minv.assign((std::size_t)nb * ng * 16, 0);
  parallel_for(nb, [&](int b0, int b1) {
    for (int cb = b0; cb < b1; ++cb)
      for (int g = 0; g < ng; ++g) {
        uint8_t* blk = q.packed.data() + ((std::size_t)cb * ng + g) * 256 * bits;
        for (int l = 0; l < 16 && cb * 16 + l < out; ++l) {
          const int c = cb * 16 + l;
          for (int j = 0; j < kGroup; ++j) {
            const int v = codes[(std::size_t)(g * kGroup + j) * out + c];
            const int kp = j / 8, r = j % 8;  // quad pair; r < 4: low nibble, r >= 4: high nibble
            for (int i = 0; i < bits; ++i)
              if (v >> i & 1) blk[(kp * bits + i) * 16 + l] |= (uint8_t)(1 << r);
          }
          q.scale[((std::size_t)cb * ng + g) * 16 + l] = scale[(std::size_t)g * out + c];
          q.minv[((std::size_t)cb * ng + g) * 16 + l] = minv[(std::size_t)g * out + c];
        }
      }
  });
  return q;
}

// Decoding single codes, for the reference kernel.
int code_at(const QLinear& w, int c, int g, int j) {
  const int ng = w.in / kGroup;
  if (w.family == Family::Unpack) {
    const Fields F = u_fields(w.bits);
    const uint8_t* field = w.packed.data() + ((std::size_t)c * ng + g) * 16 * w.bits;
    int v = 0, low = 0;
    for (int f = 0; f < F.n; ++f) {
      const int wf = F.w[f];
      int byte, shift;
      u_slot(wf, j, byte, shift);
      v |= ((field[byte] >> shift) & ((1 << wf) - 1)) << low;
      low += wf, field += 16 * wf;
    }
    return v;
  }
  const uint8_t* blk = w.packed.data() + ((std::size_t)(c / 16) * ng + g) * 256 * w.bits;
  const int kp = j / 8, r = j % 8, l = c % 16;
  int v = 0;
  for (int i = 0; i < w.bits; ++i) v |= ((blk[(kp * w.bits + i) * 16 + l] >> r) & 1) << i;
  return v;
}

std::size_t scale_index(const QLinear& w, int c, int g) {
  const int ng = w.in / kGroup;
  return w.family == Family::Unpack ? (std::size_t)c * ng + g : ((std::size_t)(c / 16) * ng + g) * 16 + c % 16;
}

#ifdef __AVX512F__

// ---- Family U ----

// The field of width W of codes j = 16M..16M+15 of one group, as 16 int32 lanes.
template <int W, int M>
inline __m512i u_field(const uint8_t* f) {
  const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(f + 16 * (M % W))));
  constexpr int shift = W * (M / W);
  __m512i s = v;
  if constexpr (shift > 0) s = _mm512_srli_epi32(v, shift);
  if constexpr (shift + W < 8) s = _mm512_and_si512(s, _mm512_set1_epi32((1 << W) - 1));
  return s;
}

// Codes j = 16M..16M+15 of one group.
template <int B, int M>
inline __m512i u_codes(const uint8_t* blk) {
  if constexpr (B == 8 || B == 4 || B == 2) return u_field<B, M>(blk);
  else if constexpr (B == 6) return _mm512_or_si512(u_field<4, M>(blk), _mm512_slli_epi32(u_field<2, M>(blk + 64), 4));
  else return _mm512_or_si512(u_field<2, M>(blk), _mm512_slli_epi32(u_field<1, M>(blk + 32), 2));  // B == 3
}

template <int B, int M>
inline void u_step(const uint8_t* blk, const float* xg, __m512& p) {
  p = _mm512_fmadd_ps(_mm512_cvtepi32_ps(u_codes<B, M>(blk)), _mm512_loadu_ps(xg + 16 * M), p);
}

// sum_j x_j q_j over one group, as 16 partial sums; four accumulators so the FMAs do not wait on each other.
template <int B>
inline __m512 u_group(const uint8_t* blk, const float* xg) {
  __m512 p0 = _mm512_setzero_ps(), p1 = p0, p2 = p0, p3 = p0;
  u_step<B, 0>(blk, xg, p0), u_step<B, 1>(blk, xg, p1), u_step<B, 2>(blk, xg, p2), u_step<B, 3>(blk, xg, p3);
  u_step<B, 4>(blk, xg, p0), u_step<B, 5>(blk, xg, p1), u_step<B, 6>(blk, xg, p2), u_step<B, 7>(blk, xg, p3);
  return _mm512_add_ps(_mm512_add_ps(p0, p1), _mm512_add_ps(p2, p3));
}

template <int B>
void u_matvec(const QLinear& w, const float* x, const float* xsum, float* y) {
  const int ng = w.in / kGroup;
  parallel_for(w.out, [&](int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
      const uint8_t* blk = w.packed.data() + (std::size_t)c * ng * 16 * B;
      const uint16_t* s = w.scale.data() + (std::size_t)c * ng;
      const uint16_t* m = w.minv.data() + (std::size_t)c * ng;
      __m512 acc = _mm512_setzero_ps();  // sum over groups of s_g * (x . q), kept in 16 lanes
      float macc = 0.f;                  // sum over groups of m_g * (sum of x over the group)
      for (int g = 0; g < ng; ++g, blk += 16 * B) {
        acc = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(s[g])), u_group<B>(blk, x + g * kGroup), acc);
        macc += _cvtsh_ss(m[g]) * xsum[g];
      }
      y[c] = _mm512_reduce_add_ps(acc) + macc;
    }
  });
}

// ---- Family P ----

// For every quad k of x (inputs 4k..4k+3), table entry e = sum of x[4k + r] over the bits r set in e.
void build_tables(const float* x, int in, float* tab) {
  for (int k = 0; k < in / 4; ++k) {
    __m512 t = _mm512_setzero_ps();
    t = _mm512_mask_add_ps(t, 0xAAAA, t, _mm512_set1_ps(x[4 * k]));      // entries with bit 0 set
    t = _mm512_mask_add_ps(t, 0xCCCC, t, _mm512_set1_ps(x[4 * k + 1]));  // bit 1
    t = _mm512_mask_add_ps(t, 0xF0F0, t, _mm512_set1_ps(x[4 * k + 2]));  // bit 2
    t = _mm512_mask_add_ps(t, 0xFF00, t, _mm512_set1_ps(x[4 * k + 3]));  // bit 3
    _mm512_storeu_ps(tab + 16 * k, t);
  }
}

// 16 output columns at a time: each lookup reads one table entry per column with a single permute.
template <int B>
void p_matvec(const QLinear& w, const float* tab, const float* xsum, float* y) {
  const int ng = w.in / kGroup, nb = (w.out + 15) / 16;
  parallel_for(nb, [&](int b0, int b1) {
    for (int cb = b0; cb < b1; ++cb) {
      __m512 acc = _mm512_setzero_ps();
      for (int g = 0; g < ng; ++g) {
        const uint8_t* blk = w.packed.data() + ((std::size_t)cb * ng + g) * 256 * B;
        const float* tg = tab + (std::size_t)g * (kGroup / 4) * 16;
        __m512 lo[B], hi[B];  // per plane, sums from the low and the high nibbles
        for (int i = 0; i < B; ++i) lo[i] = hi[i] = _mm512_setzero_ps();
        for (int kp = 0; kp < 16; ++kp) {
          const __m512 t0 = _mm512_loadu_ps(tg + 32 * kp), t1 = _mm512_loadu_ps(tg + 32 * kp + 16);
          for (int i = 0; i < B; ++i) {
            const __m512i v =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + (kp * B + i) * 16)));
            lo[i] = _mm512_add_ps(lo[i], _mm512_permutexvar_ps(v, t0));  // permute reads the low 4 bits only
            hi[i] = _mm512_add_ps(hi[i], _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), t1));
          }
        }
        __m512 part = _mm512_add_ps(lo[B - 1], hi[B - 1]);  // sum_i 2^i plane_i, by Horner's rule
        for (int i = B - 2; i >= 0; --i)
          part = _mm512_fmadd_ps(part, _mm512_set1_ps(2.f), _mm512_add_ps(lo[i], hi[i]));
        const std::size_t si = ((std::size_t)cb * ng + g) * 16;
        const __m512 s = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.scale.data() + si)));
        const __m512 m = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.minv.data() + si)));
        acc = _mm512_fmadd_ps(s, part, acc);
        acc = _mm512_fmadd_ps(m, _mm512_set1_ps(xsum[g]), acc);
      }
      const int c0 = cb * 16, n = std::min(16, w.out - c0);
      _mm512_mask_storeu_ps(y + c0, (__mmask16)((1u << n) - 1), acc);
    }
  });
}

#endif  // __AVX512F__

}  // namespace

const char* family_name(Family f) { return f == Family::Unpack ? "U" : "P"; }

QLinear pack(const uint8_t* codes, const uint16_t* scale, const uint16_t* minv, int in, int out, int bits,
             Family family) {
  if (in % kGroup) throw std::invalid_argument("input size must be a multiple of 128");
  u_fields(bits);  // throws on an unsupported width
  return family == Family::Unpack ? pack_unpack(codes, scale, minv, in, out, bits)
                                  : pack_bitplane(codes, scale, minv, in, out, bits);
}

std::pair<int, int> module_shape(int index, int d, int vocab) {
  if (index == 48) return {d, vocab};
  switch (index % 4) {
    case 0: return {d, 3 * d};
    case 1: return {d, d};
    case 2: return {d, 4 * d};
    default: return {4 * d, d};
  }
}

QLinear load_qlinear(const std::string& dir, int bits, int index, int d, int vocab, Family family) {
  const std::string path = dir + "/gpt2_q" + std::to_string(bits) + ".bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int32_t h[4];
  f.read(reinterpret_cast<char*>(h), sizeof h);
  if (!f || h[0] != kMagic || h[1] != bits || h[2] != kGroup || index < 0 || index >= h[3])
    throw std::runtime_error(path + ": bad header, or no module " + std::to_string(index));
  std::size_t offset = kHeaderBytes;
  for (int i = 0; i < index; ++i) {
    const auto [in, out] = module_shape(i, d, vocab);
    offset += (std::size_t)in * out + 4 * (std::size_t)(in / kGroup) * out;  // codes, then FP16 scales and mins
  }
  const auto [in, out] = module_shape(index, d, vocab);
  std::vector<uint8_t> codes((std::size_t)in * out);
  std::vector<uint16_t> s((std::size_t)(in / kGroup) * out), m(s.size());
  f.seekg((std::streamoff)offset);
  f.read(reinterpret_cast<char*>(codes.data()), codes.size());
  f.read(reinterpret_cast<char*>(s.data()), s.size() * 2);
  f.read(reinterpret_cast<char*>(m.data()), m.size() * 2);
  if (!f) throw std::runtime_error(path + ": short read");
  return pack(codes.data(), s.data(), m.data(), in, out, bits, family);
}

void qmatvec_reference(const QLinear& w, const float* x, float* y) {
  const int ng = w.in / kGroup;
  parallel_for(w.out, [&](int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
      double acc = 0;
      for (int g = 0; g < ng; ++g) {
        double part = 0, xs = 0;
        for (int j = 0; j < kGroup; ++j) {
          const double xj = x[g * kGroup + j];
          part += xj * code_at(w, c, g, j), xs += xj;
        }
        const std::size_t si = scale_index(w, c, g);
        acc += half_to_float(w.scale[si]) * part + half_to_float(w.minv[si]) * xs;
      }
      y[c] = (float)acc;
    }
  });
}

void qmatvec(const QLinear& w, const float* x, float* y) {
#ifdef __AVX512F__
  thread_local std::vector<float> xsum, tab;
  const int ng = w.in / kGroup;
  xsum.resize(ng);
  for (int g = 0; g < ng; ++g) {
    float s = 0.f;
    for (int j = 0; j < kGroup; ++j) s += x[g * kGroup + j];
    xsum[g] = s;
  }
  if (w.family == Family::Unpack) {
    switch (w.bits) {
      case 2: return u_matvec<2>(w, x, xsum.data(), y);
      case 3: return u_matvec<3>(w, x, xsum.data(), y);
      case 4: return u_matvec<4>(w, x, xsum.data(), y);
      case 6: return u_matvec<6>(w, x, xsum.data(), y);
      case 8: return u_matvec<8>(w, x, xsum.data(), y);
    }
  } else {
    tab.resize((std::size_t)w.in / 4 * 16);
    build_tables(x, w.in, tab.data());
    switch (w.bits) {
      case 2: return p_matvec<2>(w, tab.data(), xsum.data(), y);
      case 3: return p_matvec<3>(w, tab.data(), xsum.data(), y);
      case 4: return p_matvec<4>(w, tab.data(), xsum.data(), y);
      case 6: return p_matvec<6>(w, tab.data(), xsum.data(), y);
      case 8: return p_matvec<8>(w, tab.data(), xsum.data(), y);
    }
  }
  throw std::invalid_argument("unsupported width " + std::to_string(w.bits));
#else
  qmatvec_reference(w, x, y);
#endif
}

std::size_t qbytes(const QLinear& w) { return w.packed.size() + 2 * (w.scale.size() + w.minv.size()); }
