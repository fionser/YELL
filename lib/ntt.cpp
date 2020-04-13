#include "yell/defines.h"
#include "yell/params.hpp"
#include "yell/utils/math.hpp"
#include <iostream>
#include <array>
#include <immintrin.h>

namespace yell {

struct var_time_selector {
  using value_type = params::value_type;
  static inline value_type select(value_type a, value_type b, bool cond) {
    return cond ? a : b;
  }

  static inline value_type select0(value_type b, bool cond) {
    return select(0, b, cond);
  }
};

struct cnst_time_selector {
  using value_type = params::value_type;
  static inline value_type select(value_type a, value_type b, bool cond) {
    value_type c = -(value_type) cond;
    value_type x = a ^ b;
    return (x & c) ^ b;
  }

  static inline value_type select0(value_type b, bool cond) {
    return (b & -(value_type) cond) ^ b;
  }
};

struct ntt_loop_body {
  using value_type = params::value_type;
  using signed_type = params::signed_type;
  using gt_value_type = params::gt_value_type;
  const size_t bit_width = params::kModulusRepresentationBitsize;
  const value_type p, Lp;
  const value_type r, rshoup;

  explicit ntt_loop_body(value_type const p, value_type const r = 0, value_type const rshoup = 0) 
    : p(p), Lp(p * 2), r(r), rshoup(rshoup) { }

  inline value_type mulmod_shoup_lazy(value_type x, value_type y, value_type yshoup) const {
    value_type q = ((gt_value_type) x * yshoup) >> bit_width;
    return x * y - q * p;
  }

  //! x'0 = x0 + x1 mod p
  //! x'1 = w * (x0 - x1) mod p
  //! Require: 0 < x0, x1 < 2 * p
  //! Ensure:  0 < x'0, x'1 < 2 * p
  template <class selector = cnst_time_selector>
  inline void gs_butterfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wshoup) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    value_type t0 = u0 + u1;
    *x0 = t0 - selector::select0(Lp, t0 < Lp);
    *x1 = mulmod_shoup_lazy(u0 - u1 + Lp, w, wshoup);
  }

  //! x'0 = w0 * (x0 + x1) mod p
  //! x'1 = w1 * (x0 - x1) mod p
  template <class selector = cnst_time_selector>
  inline void gs_butterfly_last(value_type* x0, 
                                value_type* x1, 
                                value_type const w0, 
                                value_type const w0shoup,
                                value_type const w1, 
                                value_type const w1shoup) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    value_type t0 = u0 + u1;
    t0 -= selector::select0(Lp, t0 < Lp); //! if (t0 >= _2p) t0 -= 2p;

    *x0 = mulmod_shoup_lazy(t0, w0, w0shoup);
    *x1 = mulmod_shoup_lazy(u0 - u1 + Lp, w1, w1shoup);
  }

  template <class selector = cnst_time_selector>
  inline void gs_butterfly_last_mont(value_type* x0, 
                                     value_type* x1, 
                                     value_type const w0, 
                                     value_type const w0shoup,
                                     value_type const w1, 
                                     value_type const w1shoup) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    *x0 = mulmod_shoup_lazy(u0 + u1, w0, w0shoup);
    *x1 = mulmod_shoup_lazy(u0 - u1 + Lp, w1, w1shoup);
  }

  //! x'0 = x0 + w * x1 mod p
  //! x'1 = x0 - w * x1 mod p
  //! Require: 0 < x0, x1 < 4 * p
  //! Ensure:  0 < x'0, x'1 < 4 * p
  template <class selector = cnst_time_selector>
  inline void ct_butterfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wshoup) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    u0 -= selector::select0(Lp, u0 < Lp); //! if (u0 >= 2p) u0 -= 2p;
    value_type t = mulmod_shoup_lazy(u1, w, wshoup);
    *x0 = u0 + t;
    *x1 = u0 - t + Lp;
  }
  
  template <class selector = cnst_time_selector>
  inline void ct_butterfly_mont(value_type* x0, 
                                value_type* x1, 
                                value_type const w, 
                                value_type const wshoup) const
  {
    value_type u0 = mulmod_shoup_lazy(*x0, r, rshoup);
    value_type u1 = mulmod_shoup_lazy(*x1, w, wshoup);

    *x0 = u0 + u1;
    *x1 = u0 - u1 + Lp;
  }
};

//! x in given in standard order.
//! wtab is the powers of 2*degree primitive root in the bit-reversed order.
//! e.g., w^{2 * degree} = 1 \bmod p
void negacylic_forward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *wtab,
  const params::value_type *wtab_shoup,
  const params::value_type p) {

  using T = params::value_type;

  T r = static_cast<T>((-p) % p);
  T rshoup = ops::shoupify_p(r, p);

  ntt_loop_body body(p, r, rshoup);

  const T* w = wtab + 1;
  const T* wshoup = wtab_shoup + 1;

  { // main loop: for h >= 4
    size_t m = 1;
    size_t h = degree >> 1;
    for (; h > 2; m <<= 1, h >>= 1) {
      //! invariant: h * m = degree / 2
      //! different buttefly groups
      auto x0 = x;
      auto x1 = x0 + h; // invariant: x1 = x0 + h during the iteration
      for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
        //! buttefly group that use the same twiddle factor, i.e., w[r].
        for (size_t i = 0; i < h; i += 4) { // unrolling
          body.ct_butterfly(x0++, x1++, *w, *wshoup);
          body.ct_butterfly(x0++, x1++, *w, *wshoup);
          body.ct_butterfly(x0++, x1++, *w, *wshoup);
          body.ct_butterfly(x0++, x1++, *w, *wshoup);
        }
        x0 += h;
        x1 += h;
      }
    }
  }

  { // m = degree / 4, h = 2
    const size_t m = degree / 4;
    const size_t h = 2;
    auto x0 = x;
    auto x1 = x0 + 2;
    for (size_t r = 0; r < m; ++r, ++w, ++wshoup) { // unrolling
      body.ct_butterfly(x0++, x1++, *w, *wshoup);
      body.ct_butterfly(x0, x1, *w, *wshoup); // combine the incr to following steps
      x0 += 3;
      x1 += 3;
    }
  }

  { // m = degree / 2, h = 1
    const size_t m = degree / 2;
    const size_t h = 1;
    auto x0 = x;
    auto x1 = x0 + 1;
    for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
      body.ct_butterfly_mont(x0, x1, *w, *wshoup);
      x0 += 2;
      x1 += 2;
    }
  }
  //! x[0 .. degree) stay in the range [0, 4p)
}

void negacylic_backward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *wtab,
  const params::value_type *wtab_shoup,
  const params::value_type inv_n,
  const params::value_type inv_n_s,
  const params::value_type p)
{
  using T = params::value_type;

  ntt_loop_body body(p);

  const T *w = wtab + 1;
  const T *wshoup = wtab_shoup + 1;

  { // first loop: m = degree / 2, h = 1
    const size_t m = degree >> 1;
    const size_t h = 1;
    auto x0 = x;
    auto x1 = x0 + 1; // invariant: x1 = x0 + h during the iteration
    for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
      body.gs_butterfly(x0, x1, *w, *wshoup);
      x0 += 2;
      x1 += 2;
    }
  }

  { // second loop: m = degree / 4, h = 2
    const size_t m = degree >> 2;
    const size_t h = 2;
    auto x0 = x;
    auto x1 = x0 + 2;
    for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
      body.gs_butterfly(x0++, x1++, *w, *wshoup);
      body.gs_butterfly(x0, x1, *w, *wshoup); // combine the incr to following steps
      x0 += 3;
      x1 += 3;
    }
  }

  { // main loop: for h >= 4
    size_t m = degree >> 3;
    size_t h = 4;
    for (; m > 1; m >>= 1, h <<= 1) { // m > 1 to skip the last lazyer
      auto x0 = x;
      auto x1 = x0 + h;
      for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
        for (size_t i = 0; i < h; i += 4) { // unrolling
          body.gs_butterfly(x0++, x1++, *w, *wshoup);
          body.gs_butterfly(x0++, x1++, *w, *wshoup);
          body.gs_butterfly(x0++, x1++, *w, *wshoup);
          body.gs_butterfly(x0++, x1++, *w, *wshoup);
        }
        x0 += h;
        x1 += h;
      }
    }
  }

  auto x0 = x;
  auto x1 = x0 + (degree / 2);
  for (size_t i =(degree / 2); i < degree; ++i) {
    body.gs_butterfly_last_mont(x0++, x1++, inv_n, inv_n_s, *w, *wshoup);
  }
  //! x[0 .. degree) stay in the range [0, 2p)
}
} // namespace yell
