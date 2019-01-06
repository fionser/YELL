#include "yell/defines.h"
#include "yell/params.hpp"
#include <iostream>
#include <array>
namespace yell {
struct ntt_loop_body {
  using value_type = params::value_type;
  using signed_type = params::signed_type;
  using gt_value_type = params::gt_value_type;
  const value_type p, _2p;
  const std::array<value_type, 2> mod_correct_table;
#ifdef YELL_USE_AVX_NTT
  __m256i avx_2p, avx_p, avx_80, avx_2pcomp; 
#endif

  explicit ntt_loop_body(value_type const p) : p(p), _2p(p * 2), mod_correct_table({p * 2, 0}) {
#ifdef YELL_USE_AVX_NTT
    avx_2p = _mm256_set1_epi32(p << 1);
    avx_p  = _mm256_set1_epi32(p);
    avx_80 = _mm256_set1_epi32(0x80000000);
    avx_2pcomp = _mm256_set1_epi32((2*p) - 0x80000000 - 1);
#endif
  }

  //! x'0 = x0 + x1 mod p
  //! x'1 = w * (x0 - x1) mod p
  //! Require: 0 < x0, x1 < 2 * p
  //! Ensure:  0 < x'0, x'1 < 2 * p
  inline void gs_bufferfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    value_type t0 = u0 + u1;
    t0 -= mod_correct_table[t0 < _2p]; // if (t0 >= _2p) t0 -= 2p;
    value_type t1 = u0 - u1 + _2p;
    value_type q = ((gt_value_type) t1 * wprime) >> params::kModulusRepresentationBitsize;

    *x0 = t0;
    *x1 = t1 * w - q * p;
  }

  //! x'0 = x0 + w * x1 mod p
  //! x'1 = x0 - w * x1 mod p
  //! Require: 0 < x0, x1 < 4 * p
  //! Ensure:  0 < x'0, x'1 < 4 * p
  inline void ct_bufferfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    u0 -= mod_correct_table[u0 < _2p]; // if (u0 >= 2p) u0 -= 2p;
    value_type q = ((gt_value_type) u1 * (wprime)) >> params::kModulusRepresentationBitsize;
    value_type t = u1 * w - q * p;

    *x0 = u0 + t;
    *x1 = u0 - t + _2p;
  }

#ifdef YELL_USE_AVX_NTT
  inline void avx_gs_bufferfly(value_type* x0, 
                               value_type* x1, 
                               value_type const w, 
                               value_type const wprime) const
  {
    __m256i avx_u0, avx_u1;
    __m256i avx_t0, avx_t1, avx_t2;
    __m256i avx_q, avx_cmp;

    avx_u0 = _mm256_load_si256((__m256i const*) x0);
    avx_u1 = _mm256_load_si256((__m256i const*) x1);
    const __m256i avx_w = _mm256_set1_epi32(w);
    const __m256i avx_wprime = _mm256_set1_epi32(wprime);

    //! t1 = u0 - u1 + 2p
    avx_t1 = _mm256_add_epi32(avx_2p, _mm256_sub_epi32(avx_u0, avx_u1));
    //! q = high 32 bit of (t1 * wprime)
    avx_q = avx_mm256_mul32_hi(avx_t1, avx_wprime);
    avx_t2 = _mm256_sub_epi32(_mm256_mullo_epi32(avx_t1, avx_w),
                              _mm256_mullo_epi32(avx_q, avx_p));

    avx_t0 = _mm256_add_epi32(avx_u0, avx_u1);
    avx_cmp = _mm256_cmpgt_epi32(_mm256_sub_epi32(avx_t0, avx_80), avx_2pcomp);
    avx_t0 = _mm256_sub_epi32(avx_t0, _mm256_and_si256(avx_cmp, avx_2p));

    _mm256_store_si256((__m256i*) x0, avx_t0);
    _mm256_store_si256((__m256i*) x1, avx_t2);
  }

  inline void avx_ct_bufferfly(value_type* x0, 
                               value_type* x1, 
                               value_type const w, 
                               value_type const wprime) const
  {
    __m256i avx_u0, avx_u1;
    __m256i avx_t0, avx_t1;
    __m256i avx_q, avx_cmp;

    avx_u0 = _mm256_load_si256((__m256i const*) x0);
    avx_u1 = _mm256_load_si256((__m256i const*) x1);
    const __m256i avx_w = _mm256_set1_epi32(w);
    const __m256i avx_wprime = _mm256_set1_epi32(wprime);

    //! q = high 32bit of (u1 * wprime)
    avx_q = avx_mm256_mul32_hi(avx_u1, avx_wprime);
    //! u1 * w - q * p
    avx_t1 = _mm256_sub_epi32(_mm256_mullo_epi32(avx_u1, avx_w),
                              _mm256_mullo_epi32(avx_q, avx_p));

    avx_cmp = _mm256_cmpgt_epi32(_mm256_sub_epi32(avx_u0, avx_80), avx_2pcomp);
    avx_t0 = _mm256_sub_epi32(avx_u0, _mm256_and_si256(avx_cmp, avx_2p));

    avx_u0 = _mm256_add_epi32(avx_t0, avx_t1);
    avx_u1 = _mm256_add_epi32(_mm256_sub_epi32(avx_t0, avx_t1), avx_2p);

    _mm256_store_si256((__m256i*) x0, avx_u0);
    _mm256_store_si256((__m256i*) x1, avx_u1);
  }
#endif // YELL_USE_AVX_NTT
};

void negacylic_forward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *wtab,
  const params::value_type *wtab_shoup,
  const params::value_type p)
{
  ntt_loop_body body(p);
  size_t t = degree;
  for (size_t m = 1; m < degree; m <<= 1) {
    t >>= 1u;
    const params::value_type *w = &wtab[m];
    const params::value_type *wshoup = &wtab_shoup[m];
    if (t > YELL_NTT_UNROOL_SIZE) {
      for (size_t i = 0; i != m; ++i) {
        const size_t j1 = 2 * i * t;
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
#ifndef YELL_USE_AVX_NTT
        for (size_t j = j1; j != j2; j += 4) {
          body.ct_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.ct_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.ct_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.ct_bufferfly(x0++, x1++, w[i], wshoup[i]);
        }
#else
        for (size_t j = j1; j != j2; j += YELL_AVX_BATCH_SIZE) {
          body.avx_ct_bufferfly(x0, x1, w[i], wshoup[i]);
          x0 += YELL_AVX_BATCH_SIZE;
          x1 += YELL_AVX_BATCH_SIZE;
        }
#endif
      }
    } else { //! last two layers
      for (size_t i = 0; i != m; ++i) {
        const size_t j1 = 2 * i * t;
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        for (size_t j = j1; j != j2; ++j)
          body.ct_bufferfly(x0++, x1++, w[i], wshoup[i]);
      }
    }
  }
  //! x[0 .. degree) stay in the range [0, 4p)
}

void negacylic_backward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *wtab,
  const params::value_type *wtab_shoup,
  const params::value_type p)
{
  ntt_loop_body body(p);
  size_t t = 1;
  for (size_t m = degree; m > 2; m >>= 1u) { //! 'm > 2' to skip the last layer.
    const size_t h = m >> 1u;
    size_t j1 = 0;
    const params::value_type *w = &wtab[h];
    const params::value_type *wshoup = &wtab_shoup[h];
    if (t > YELL_NTT_UNROOL_SIZE) { 
      for (size_t i = 0; i != h; ++i) {
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
#ifndef YELL_USE_AVX_NTT
        //! Unroll a little bit to reduce the number of branches.
        for (size_t j = j1; j != j2; j += 4) {
          body.gs_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.gs_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.gs_bufferfly(x0++, x1++, w[i], wshoup[i]);
          body.gs_bufferfly(x0++, x1++, w[i], wshoup[i]);
        }
#else
        for (size_t j = j1; j != j2; j += YELL_AVX_BATCH_SIZE) {
          body.avx_gs_bufferfly(x0, x1, w[i], wshoup[i]);
          x0 += YELL_AVX_BATCH_SIZE;
          x1 += YELL_AVX_BATCH_SIZE;
        }
#endif
        j1 = j1 + (t << 1u);
      }
    } else { //! first two layers
      for (size_t i = 0; i != h; ++i) {
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        for (size_t j = j1; j != j2; ++j)
          body.gs_bufferfly(x0++, x1++, w[i], wshoup[i]);
        j1 = j1 + (t << 1u);
      }
    }
    t <<= 1u;
  }
  //! x[0 .. degree) stay in the range [0, 2p)
}
} // namespace yell
