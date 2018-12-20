#pragma once
#include <gmp.h>
#include <array>
#include <iosfwd>
#include <cassert>
#include <iterator>
#include "yell/ntt.hpp"
#include "yell/params.hpp"
#include "yell/prng/fastrandombytes.h"
#include "yell/prng/FastGaussianNoise.hpp"
#ifdef USE_MEM_POOL
#include <boost/pool/poolfwd.hpp>
#endif

namespace yell {
struct uniform {};
/***
 * Generators to initialize random polynomials
 */
struct hwt_dist { // hamming weight distribution.
  uint32_t hwt;
  explicit hwt_dist(uint32_t hwt_) : hwt(hwt_) {}
};

struct ZO_dist { // zero distribution.
  uint8_t rho; // P(1) = P(-1) = (rho/0xFF)/2, P(0) = 1 - P(1) - P(-1)
  explicit ZO_dist(uint8_t rho_ = 0x7F) : rho(rho_) {}
};

template<class in_class, unsigned _lu_depth>
struct gaussian {
  FastGaussianNoise<in_class, _lu_depth> *fg_prng;
  uint64_t amplifier;
  explicit gaussian(FastGaussianNoise<in_class, _lu_depth> *prng) 
    : fg_prng{prng}, amplifier{1} {}

  gaussian(FastGaussianNoise<in_class, _lu_depth> *prng, uint64_t amp) 
    : fg_prng{prng}, amplifier{amp} {}
};

template <size_t degree_, size_t nmoduli_>
class poly {
  static_assert(degree_ < yell::params::kMaxPolyDegree, "");
  static_assert(nmoduli_ < yell::params::kMaxNbModuli, "");
  std::array<params::value_type *, nmoduli_> _data;
#ifdef USE_MEM_POOL
  static boost::pool<> _mem_pool; // memory pool.
#endif

public:
  using value_type = params::value_type;
  using gt_value_type = params::gt_value_type;
  using signed_type = params::signed_type;
  using pointer_type = uint64_t *;
  using const_pointer_type = uint64_t const*;
  using iterator = pointer_type;
  using const_iterator = const_pointer_type;
  static constexpr size_t degree = degree_;
  static constexpr size_t nmoduli = nmoduli_;
  static constexpr size_t logn = yell::static_log2<degree_>::value;
  /* constructor
   */
  poly();
  ~poly();
  poly(poly const& oth);
  poly(poly &&oth);
  poly& operator=(poly const& oth);
  explicit poly(value_type v); // init as all 'v'.
  explicit poly(uniform const& mode);
  explicit poly(hwt_dist const& mode);
  explicit poly(ZO_dist const& mode);
  template <class in_class, unsigned _lu_depth> 
  explicit poly(gaussian<in_class, _lu_depth> const& mode);

  void set(uniform const& mode);
  void set(hwt_dist const& mode);
  void set(ZO_dist const& mode);
  template <class in_class, unsigned _lu_depth> 
  void set(gaussian<in_class, _lu_depth> const& mode);

  /* iterators
   */
  iterator ptr_at(size_t cm) { return _data.at(cm); }
  iterator ptr_end(size_t cm) { return ptr_at(cm) + degree; }
  const_iterator cptr_at(size_t cm) const { return _data.at(cm); }
  const_iterator cptr_end(size_t cm) const { return cptr_at(cm) + degree; }
  /* operators
   */
  bool operator==(poly const& oth) const;
  bool operator!=(poly const& oth) const { return !(*this == oth); }
  // *this += op0 * op1. Use lazy reduction for a better performance.
  poly<degree_, nmoduli_>& add_product_of(const poly<degree_, nmoduli_>& op0, 
                                          const poly<degree_, nmoduli_>& op1);
  void negate();
  
  void automorphism_inplace(const size_t k);

  /* polynomial indexing
   */
  value_type const& operator()(size_t cm, size_t i) const { 
    assert(cm < nmoduli && i < degree);
    return _data[cm][i];
  }
  
  value_type& operator()(size_t cm, size_t i) { 
    assert(cm < nmoduli && i < degree);
    return _data[cm][i];
  }

  /* NTT related.
   */
  void forward();
  void forward_lazy(); // skip the final correction.
  void backward();
  /* misc
   */
  value_type** raw_data() { return _data.data(); }
  value_type* const* raw_data() const { return _data.data(); }

  value_type get_modulus(size_t cm) { assert(cm < nmoduli); return params::P[cm]; }

  void clear();

  constexpr size_t size() const { return sizeof(_data); }
  /* The caller should clean up the mpz_t array.
   */
  std::array<mpz_t, degree_> poly2mpz() const;
};

/* stream operator
 */
template<size_t Degree, size_t NbModuli>
std::ostream& operator<<(std::ostream& os, 
                         yell::poly<Degree, NbModuli> const& p);

/* Recasting. Caller should know the memory layout of poly 
   which is just the concatnation of "nmoduli" moduli with "degree" coefficients each: 
   [moduli 1 || moduli 2 || .... || moduli L]
*/
template <size_t L0, size_t degree, size_t nmoduli>
poly<degree, L0>* take_head_moduli(poly<degree, nmoduli> &op) {
  static_assert(L0 <= nmoduli, "");
  return reinterpret_cast<poly<degree, L0> *>(op.raw_data());
}

template <size_t L0, size_t degree, size_t nmoduli>
const poly<degree, L0>* take_head_moduli(poly<degree, nmoduli> const& op) {
  static_assert(L0 <= nmoduli, "");
  return reinterpret_cast<const poly<degree, L0> *>(op.raw_data());
}

template <size_t L0, size_t degree, size_t nmoduli>
poly<degree, L0>* take_tail_moduli(poly<degree, nmoduli> &op) {
  static_assert(L0 <= nmoduli, "");
  return reinterpret_cast<poly<degree, L0> *>(op.raw_data() + (nmoduli - L0));
}

template <size_t L0, size_t degree, size_t nmoduli>
const poly<degree, L0>* take_tail_moduli(poly<degree, nmoduli> const& op) {
  static_assert(L0 <= nmoduli, "");
  return reinterpret_cast<const poly<degree, L0> *>(op.raw_data() + (nmoduli - L0));
}

template <typename value_type, size_t degree>
struct ArrayPointer {
  using T = std::array<value_type, degree> *;
  using cT = const T;
};

/* recast the specific moduli as an array. */
template <size_t degree, size_t nmoduli> typename ArrayPointer<yell::params::value_type, degree>::T
recast_as_array(yell::poly<degree, nmoduli> &op, size_t cm) {
  assert(cm < op.nmoduli);
  return reinterpret_cast<typename ArrayPointer<yell::params::value_type, degree>::T>(op.ptr_at(cm));
}

template <size_t degree, size_t nmoduli> typename ArrayPointer<yell::params::value_type, degree>::cT
recast_as_array(yell::poly<degree, nmoduli> const& op, size_t cm) {
  assert(cm < op.nmoduli);
  return reinterpret_cast<typename ArrayPointer<yell::params::value_type, degree>::cT>(op.cptr_at(cm));
}

DECLARE_BINARY_OPERATOR(operator*, mulmod);
DECLARE_BINARY_OPERATOR(operator+, addmod);
DECLARE_BINARY_OPERATOR(operator-, submod);
DECLARE_SELF_BINARY_OPERATOR(operator*=, mulmod);
DECLARE_SELF_BINARY_OPERATOR(operator+=, addmod);
DECLARE_SELF_BINARY_OPERATOR(operator-=, submod);

} // namespace yell

#include "yell/core.hpp"

