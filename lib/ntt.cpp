#include "yell/defines.h"
#include "yell/params.hpp"
#include <iostream>
#include <array>
#include <immintrin.h>

namespace yell {
//! return a < p ? a : a - p where p < 2^w
template <typename T>
inline T mod_correct(T a, const T p, const size_t w) {
  a -= p;
  return ((a >> (w - 1)) & p) + a;
}

struct ntt_loop_body {
  using value_type = params::value_type;
  using signed_type = params::signed_type;
  using gt_value_type = params::gt_value_type;
  const size_t bit_width = params::kModulusRepresentationBitsize;
  const value_type p, _2p;

  explicit ntt_loop_body(value_type const p) : p(p), _2p(p * 2) { }

  //! if (cond is true) return a else return b
  inline value_type const_time_select(value_type a, value_type b, bool cond) const {
      value_type c = -(value_type) cond;
      value_type x = a ^ b;
      return (x & c) ^ b;
  }

  //! x'0 = x0 + x1 mod p
  //! x'1 = w * (x0 - x1) mod p
  //! Require: 0 < x0, x1 < 2 * p
  //! Ensure:  0 < x'0, x'1 < 2 * p
  inline void gs_butterfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    value_type t0 = u0 + u1;
    t0 -= const_time_select(0, _2p, t0 < _2p); //! if (t0 >= _2p) t0 -= 2p;
    value_type t1 = u0 - u1 + _2p;
    value_type q = ((gt_value_type) t1 * wprime) >> bit_width;

    //! w is in the div_2_mod_p form, so we need to handle x0 here.
    *x0 = const_time_select(t0 + p, t0, t0 & 1) >> 1;
    *x1 = t1 * w - q * p;
  }

  //! x'0 = x0 + w * x1 mod p
  //! x'1 = x0 - w * x1 mod p
  //! Require: 0 < x0, x1 < 4 * p
  //! Ensure:  0 < x'0, x'1 < 4 * p
  inline void ct_butterfly(value_type* x0, 
                           value_type* x1, 
                           value_type const w, 
                           value_type const wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    u0 -= const_time_select(0, _2p, u0 < _2p); //! if (u0 >= 2p) u0 -= 2p;
    value_type q = ((gt_value_type) u1 * wprime) >> bit_width;
    value_type t = u1 * w - q * p;
    *x0 = u0 + t;
    *x1 = u0 - t + _2p;
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
  const params::value_type p)
{
  ntt_loop_body body(p);
  size_t h = degree >> 1u;
  for (size_t m = 1; m < degree; m <<= 1) {
    const params::value_type *w = &wtab[m];
    const params::value_type *wshoup = &wtab_shoup[m];
    //! different buttefly groups
    auto x0 = x;
    auto x1 = x0 + h;
    for (size_t r = 0; r < m; ++r) {
      //! buttefly group that use the same twiddle factor, i.e., w[r].
      for (size_t i = 0; i < h; ++i)
        body.ct_butterfly(x0++, x1++, w[r], wshoup[r]);
      x0 += h;
      x1 = x0 + h;
    }
    h >>= 1u;
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
  for (size_t m = degree; m > 1; m >>= 1u) {
    const size_t h = m >> 1u;
    const params::value_type *w = &wtab[h];
    const params::value_type *wshoup = &wtab_shoup[h];
    auto x0 = x;
    auto x1 = x0 + t;
    for (size_t i = 0; i < h; ++i) {
      for (size_t j = 0; j < t; ++j)
        body.gs_butterfly(x0++, x1++, w[i], wshoup[i]);
      x0 += t;
      x1 = x0 + t;
    }
    t <<= 1u;
  }
  //! x[0 .. degree) stay in the range [0, 2p)
}
} // namespace yell
