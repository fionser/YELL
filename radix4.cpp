#include "yell/poly.hpp"
#include "yell/utils/timer.hpp"
#include "yell/utils/math.hpp"
#include <vector>
#include <nfl.hpp>
using T = yell::params::value_type;
using gT = yell::params::gt_value_type;

uint32_t base4_reverse_bits(uint32_t v) {
  uint32_t r = v;
  int s = 30;
  for (v >>= 2; v; v >>= 2) {
    r <<= 2;
    r |= (v & 3);
    s -= 2;
  }
  return r << s;
}

uint32_t base4_reverse_bits(uint32_t v, int n) {
  uint64_t vv = base4_reverse_bits(v);
  return (uint32_t) (v >> (32 - n));
}

struct radix4_body {
  T p, _2p;
  T correct_tbl[2] = {_2p, 0};
  T _j, _jshoup;

  explicit radix4_body(T p_, T _j, T _jshoup) :
    p(p_), _2p(p_ << 1), _j(_j), _jshoup(_jshoup) {}

  inline void correct_to_double(T &u) const {
    u -= (u < _2p ? 0 : _2p);
  }

  inline void correct_to_single(T &u) const {
    u -= (u < p ? 0 : p);
  }

  //! multiply the 1*j image part.
  //! input range in [0, 4p)
  //! output range in [0, 4p)
  //! x0' = 1*j*(x0 + x1)
  //! x1' = 1*j*(x0 - x1)
  inline void ct_butterfly(T* x0, T* x1) const {
    T u0 = *x0;
    T u1 = *x1;
    correct_to_double(u0);
    T q = ((gT) u1 * _jshoup) >> yell::params::kModulusRepresentationBitsize;
    T t = u1 * _j - q * p;

    *x0 = u0 + t;
    *x1 = u0 - t + _2p;
  }

  inline T mulmod_lazy(T x, T y, T yshoup) const {
    T q = (((gT) x * yshoup) >> yell::params::kModulusRepresentationBitsize) * p;
    return (T) (x * y - q);
  }

  //! when w = 1
  void run(T *x0, T *x1, T *x2, T *x3) const
  {
    T u0 = *x0;
    T u1 = *x1;
    T u2 = *x2;
    T u3 = *x3;

    //! range in [0, 4p)
    T v0v2 = u0 - u2 + _2p;
    T v1v3 = u1 - u3 + _2p;
    ct_butterfly(&v0v2, &v1v3);
    correct_to_double(v0v2);
    correct_to_double(v1v3);

    //! range in [0, 4p)
    T u0u2 = u0 + u2;
    T u1u3 = u1 + u3;
    correct_to_double(u0u2);
    correct_to_double(u1u3);

    T t0 = u0u2 + u1u3;
    T t2 = u0u2 - u1u3 + _2p;

    correct_to_double(t0);
    correct_to_double(t2);

    *x0 = t0;
    *x1 = v0v2;
    *x2 = t2;
    *x3 = v1v3;

    assert(*x0 < _2p);
    assert(*x1 < _2p);
    assert(*x2 < _2p);
    assert(*x3 < _2p);
  }

  //! input range in [0, 2p)
  //! output range in [0, 2p)
  void run(T *x0, T *x1, T *x2, T *x3,
           const T w1, const T w1p,
           const T w2, const T w2p,
           const T w3, const T w3p) const
  {
    T u0 = *x0;
    T u1 = *x1;
    T u2 = *x2;
    T u3 = *x3;

    //! range in [0, 4p)
    T v0v2 = u0 - u2 + _2p;
    T v1v3 = u1 - u3 + _2p;
    ct_butterfly(&v0v2, &v1v3);

    //! range in [0, 4p)
    T u0u2 = u0 + u2;
    T u1u3 = u1 + u3;
    correct_to_double(u0u2);
    correct_to_double(u1u3);
    T t2 = u0u2 - u1u3 + _2p;
    T t0 = u0u2 + u1u3;
    correct_to_double(t0);

    *x0 = t0;
    *x1 = mulmod_lazy(v0v2, w1, w1p);
    *x2 = mulmod_lazy(  t2, w2, w2p);
    *x3 = mulmod_lazy(v1v3, w3, w3p);

    assert(*x0 < _2p);
    assert(*x1 < _2p);
    assert(*x2 < _2p);
    assert(*x3 < _2p);
  }
};

void negacylic_forward_lazy(
  T *x, 
  const size_t degree,
  const T *wtab,
  const T *wtab_shoup,
  const T p)
{
  T neg = wtab[degree / 4];
  T neg_prime = wtab_shoup[degree / 4];
  radix4_body body(p, neg, neg_prime);
  size_t h = degree >> 2u;
  for (size_t m = 1; m < degree; m <<= 2) {
    //TODO exchange for-loops for a better locality.
    for (size_t r = 0; r < m; ++r) {
      auto x0 = &x[4 * h * r];
      auto x1 = x0 + h;
      auto x2 = x1 + h;
      auto x3 = x2 + h;
      body.run(x0++, x1++, x2++, x3++);
      for (size_t i = 1; i < h; ++i) {
        auto w1 = wtab[i * m];
        auto w2 = wtab[2 * i * m];
        auto w3 = wtab[3 * i * m];
        auto w1p = wtab_shoup[i * m];
        auto w2p = wtab_shoup[2 * i * m];
        auto w3p = wtab_shoup[3 * i * m];
        body.run(x0++, x1++, x2++, x3++, w1, w1p, w2, w2p, w3, w3p);
      }
      // std::cout << "-----\n";
    }
    h >>= 2u;
  }

  // std::transform(x, x + degree, x,
  //                [p](T v) {
  //                  v -= (v < p ? 0 : p);
  //                  return v;
  //                });
  // int J = 0;
  // int N2 = degree>>2;
  // int N1;
  // for (int I=0; I < degree-1; I++) {
  //   if (I < J)
  //     std::swap(x[I], x[J]);
  //   N1 = N2;
  //   while ( J >= 3*N1 ) {
  //     J -= 3*N1;
  //     N1 >>= 2;
  //   }
  //   J += N1;
  // }
  //! x[0 .. degree) stay in the range [0, 4p)
}

void prep_wtab(T *wtab, T *wtab_shoup, size_t n, T w, size_t cm)
{
  unsigned K = n;
  yell::ops::mulmod mulmod;
  while (K >= 2) {
    T wi = 1;
    for (size_t i = 0; i < K / 2; ++i) {
      *wtab++ = wi;
      std::cout<< wi << " ";
      *wtab_shoup = yell::ops::shoupify(wi, cm);
      wi = mulmod(wi, w, cm);
    }
    std::cout << "\n";
    w = mulmod(w, w, cm);
    K >>= 1;
  }
}

int main() {
  constexpr size_t degree = 16384;
  T w[degree], wprime[degree], invw[degree], tmp[degree + 1];
  auto tbl = yell::ntt<degree>::init_table(0);

  yell::permute<degree>::compute(tmp, tbl->phiTbl);
  //prep_wtab(w, wprime, degree, tmp[1], 0);
  std::memcpy(w, tmp, sizeof(w));
  yell::permute<degree>::compute(tmp, tbl->shoupPhiTbl);
  std::memcpy(wprime, tmp, sizeof(wprime));
  yell::permute<degree>::compute(tmp, tbl->invphiTbl);
  std::memcpy(invw, tmp, sizeof(invw));

  yell::poly<degree> A(1, yell::uniform{});
  for (long i = 0; i < A.degree; ++i)
    assert(A(0, i) < 2 * yell::params::P[0]);
  auto cpy(A);

  yell::ops::addmod addmod;
  yell::ops::mulmod mulmod;
  yell::ops::muladd muladd;

  for (int k = 0; k < A.degree; ++k) {
    mulmod.compute(w[k], w[k], 0);
    wprime[k] = yell::ops::shoupify(w[k], 0);
  }

  if (degree < 1024) {
    for (int k = 0; k < A.degree; ++k) {
      T sum{A(0, 0)};
      T w_origin = w[k];
      T w_ = w_origin;
      for (int i = 1; i < A.degree; ++i) {
        muladd.compute(sum, w_, A(0, i), 0);
        mulmod.compute(w_, w_origin, 0);
      }
      std::cout << sum << " ";
    }
    std::cout << "\n";

    negacylic_forward_lazy(A.ptr_at(0), A.degree, w, wprime, yell::params::P[0]);
    std::cout << A << "\n";
  }

  double rd4 = 0., rd2 = 0.;

  for (long i = 0; i < 1000; ++i) {
    if (i & 1) {
      {
        AutoTimer timer(&rd2);
        cpy.forward_lazy();
      }
      {
        AutoTimer timer(&rd4);
        negacylic_forward_lazy(A.ptr_at(0), A.degree, w, wprime, 0);
      }
    } else {
      {
        AutoTimer timer(&rd4);
        negacylic_forward_lazy(A.ptr_at(0), A.degree, w, wprime, 0);
      }
      {
        AutoTimer timer(&rd2);
        cpy.forward_lazy();
      }
    }
  }

  std::cout << "rd4 : rd2\n";
  std::cout << rd4 << " " << rd2 << "\n";
}
