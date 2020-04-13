#pragma once
#include "yell/params.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
namespace yell {
#define YELL_BINARY_OPERATOR(Op, Name) \
template<size_t degree> \
poly<degree> Op(poly<degree> const& op0, \
                poly<degree> const& op1) {\
  assert(op0.moduli_count() == op1.moduli_count()); \
  size_t nmoduli = std::min(op0.moduli_count(), \
                            op1.moduli_count()); \
  ops::Name op; \
  poly<degree> rop(nmoduli); \
  for (size_t cm = 0; cm < nmoduli; ++cm) { \
    auto dst = rop.ptr_at(cm); \
    auto op0_ptr = op0.cptr_at(cm); \
    auto op1_ptr = op1.cptr_at(cm); \
    for (size_t i = 0; i < degree; ++i) \
      *dst++ = op(*op0_ptr++, *op1_ptr++, cm);\
  } \
  return rop; \
}

#define YELL_SELF_BINARY_OPERATOR(Op, Name) \
template<size_t degree> \
poly<degree>& Op(poly<degree> &op0, \
                 poly<degree> const& op1) {\
  assert(op0.moduli_count() == op1.moduli_count()); \
  size_t nmoduli = std::min(op0.moduli_count(), \
                            op1.moduli_count()); \
  ops::Name op; \
  for (size_t cm = 0; cm < nmoduli; ++cm) { \
    auto dst = op0.ptr_at(cm); \
    auto op1_ptr = op1.cptr_at(cm); \
    for (size_t i = 0; i < degree; ++i) \
      op.compute(*dst++, *op1_ptr++, cm);\
  } \
  return op0; \
}

namespace ops
{

/* a -= (a >= p) ? p : 0 */
template<typename T>
static inline void mod_correct(T &a, const T p)
{
  using ST = typename std::make_signed<T>::type;
  a -= (p & static_cast<T>(-static_cast<ST>(a >= p)));
}

struct addmod {
  using T = typename params::value_type;

  T operator()(T x, T y, size_t cm) const {
    compute(x, y, cm);
    return x;
  }

  inline void compute(T &x, T y, size_t cm) const {
    auto const p = params::P[cm];
    assert(x < p); 
    assert(y < p);
    x += y;
    mod_correct(x, p);
  }
};

struct submod {
  using T = typename params::value_type;
  inline T operator()(T x, T y, size_t cm) const {
    compute(x, y, cm);
    return x;
  }

  inline void compute(T &x, T y, size_t cm) const {
    auto const p = params::P[cm];
    assert(x < p && y < p);
    x += (p - y);
    mod_correct(x, p);
  }
};

/*
   Barret Reduction.
   Reduce (*x) % params::P[cm].

   X * r = X * (4*2^64 + pm) = 4*X * 2^64 + X * pm is a 196-bit value.
   represent as two parts X := X1 * 2^64 + X0; then
   X * pm = X1 * pm * 2^64 + X0 * pm is a 196-bit value. We only care
   the highest 64-bit, i.e., (X1 * pm) >> 64;

   Final value is (4 * X * 2^64 + X * pm) >> 128. We do not have 192-bit word.
   So we compute the highest 64-bit as
      (4 * X + (X * pm) >> 64) >> 64
   -> (4*X + ((X >> 64) * pm)) >> 64;
*/
void barret_reduction(params::gt_value_type *x, size_t cm);

yell::params::value_type shoupify(yell::params::value_type x, size_t cm);

yell::params::value_type shoupify_p(yell::params::value_type x, yell::params::value_type p);

struct mulmod {
  using T = typename params::value_type;
  using gt_value_type = typename params::gt_value_type;
  inline T operator()(T x, T y, size_t cm) const 
  {
    assert(cm < params::kMaxNbModuli);
    gt_value_type res = (gt_value_type) x * y;
    barret_reduction(&res, cm);
    return (T) res;
  }

  inline void compute(T &x, T y, size_t cm) const {
    assert(cm < params::kMaxNbModuli);
    gt_value_type res = (gt_value_type) x * y;
    barret_reduction(&res, cm);
    x = (T) res;
  }
};

struct mulmod_shoup {
  using T = typename params::value_type;
  inline T operator()(T x, T y, T yprime, size_t cm) const
  {
    using gt_value_type = typename params::gt_value_type;
    auto const p = params::P[cm];
    T q = (((gt_value_type) x * yprime) >> params::kModulusRepresentationBitsize) * p;
    T res = (T) (x * y - q);
    mod_correct(res, p);
    return res;
  }

  inline void compute(T &x, T y, T yprime, size_t cm) const
  {
    using gt_value_type = typename params::gt_value_type;
    auto const p = params::P[cm];
    T q = (((gt_value_type) x * yprime) >> params::kModulusRepresentationBitsize) * p;
    x = x * y - q;
    mod_correct(x, p);
  }
};

struct mulmod_mont {
  using T = typename params::value_type;
  using gT = typename params::gt_value_type;

  T m;
  explicit mulmod_mont(T m) : m(m) { }

  T operator()(T a, T b, size_t cm) const {
    T p = params::P[cm];
    gT t = static_cast<gT>(a) * b;
    T u = static_cast<T>(t) * m;
    gT c = static_cast<gT>(u) * p + t;
    T r = static_cast<T>(c >> params::kModulusRepresentationBitsize);
    mod_correct(r, p);
    return r;
  }
};

struct muladd_mont {
  using T = typename params::value_type;
  using gT = typename params::gt_value_type;

  T m;
  explicit muladd_mont(T m) : m(m) { }

  inline T operator()(T rop, T x, T y, size_t cm) const {
    compute(rop, x, y, cm);
    return rop;
  }

  inline void compute(T &rop, T a, T b, size_t cm) const {
    T p = params::P[cm];
    gT t = static_cast<gT>(a) * b;
    T u = static_cast<T>(t) * m;
    gT c = static_cast<gT>(u) * p + t;
    T r = static_cast<T>(c >> params::kModulusRepresentationBitsize);
    mod_correct(r, p);
    rop += r;
    mod_correct(rop, p);
  }
};

struct muladd {
  using T = typename params::value_type;
  using gt_value_type = typename params::gt_value_type;
  inline T operator()(T rop, T x, T y, size_t cm) const {
    compute(rop, x, y, cm);
    return rop;
  }

  inline void compute(T &rop, T x, T y, size_t cm) const {
    gt_value_type res = (gt_value_type) x * y;
    res += rop;
    barret_reduction(&res, cm);
    rop = (T) res;
  }
};

} // namespace ops
} // namespace yell
