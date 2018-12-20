#pragma once

#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include "yell/gmp.hpp"
#include "yell/utils/math.hpp"

#ifdef USE_MEM_POOL
#include <boost/pool/pool.hpp>
template<size_t degree, size_t nmoduli>
boost::pool<> nfl::poly<degree, nmoduli>::_mem_pool(degree * sizeof(value_type));
#define ALLOC_MEM(bytes) _mem_pool.malloc()
#define RELEASE_MEM(ptr) _mem_pool.free(ptr)
#else
#include "yell/utils/mem.hpp"
#define ALLOC_MEM(bytes) mem_alloc(bytes)
#define RELEASE_MEM(ptr) mem_free(ptr)
#endif

namespace yell {

// *********************************************************
//    Constructor
// *********************************************************

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly() {
  constexpr size_t bytes = degree * sizeof(value_type);
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    _data[cm] = (value_type *) ALLOC_MEM(bytes);
    std::memset(_data[cm], 0, bytes);
  }
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(poly const& oth) {
  constexpr size_t bytes = degree * sizeof(value_type);
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    assert(oth._data[cm]); 
    _data[cm] = (value_type *) ALLOC_MEM(bytes);
    std::memcpy(_data[cm], oth.cptr_at(cm), bytes);
  }
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(poly && oth) : _data(oth._data) {
  for (auto& ptr : oth._data)
    ptr = nullptr;
}

template<size_t degree, size_t nmoduli> poly<degree, nmoduli>& 
poly<degree, nmoduli>::operator=(poly const& oth) 
{
  constexpr size_t bytes = degree * sizeof(value_type);
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    if (!_data.at(cm))
      _data[cm] = (value_type *) ALLOC_MEM(bytes);
    std::memcpy(_data[cm], oth.cptr_at(cm), bytes);
  }
  return *this;
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::~poly() {
  for (auto& ptr : _data) {
    RELEASE_MEM(ptr);
    ptr = nullptr;
  }
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(value_type v) : poly<degree, nmoduli>() {
  if (v == 0) 
    return ; // already init as 0.
  for (size_t cm = 0; cm < nmoduli; ++cm) 
    std::fill(ptr_at(cm), ptr_end(cm), v);
}

// ****************************************************
// operators
// ****************************************************
template<size_t degree_, size_t nmoduli_>
bool poly<degree_, nmoduli_>::operator==(poly<degree_, nmoduli_> const& oth) const{
  constexpr size_t bytes = degree_ * sizeof(value_type);
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    if (std::memcmp(cptr_at(cm), oth.cptr_at(cm), bytes) != 0)
      return false;
  }
  return true;
}

template<size_t degree_, size_t nmoduli_>
void poly<degree_, nmoduli_>::clear() {
  for (size_t cm = 0; cm < nmoduli; ++cm) 
    std::fill(ptr_at(cm), ptr_end(cm), 0);
}

template<size_t degree_, size_t nmoduli_> poly<degree_, nmoduli_>& 
poly<degree_, nmoduli_>::add_product_of(const poly<degree_, nmoduli_>& op0, 
                                        const poly<degree_, nmoduli_>& op1)
{
  nfl::ops::muladd muladd;
  for (size_t cm = 0; cm < nmoduli_; ++cm) {
    auto dst  = ptr_at(cm);
    auto op0_ = op0.cptr_at(cm);
    auto op1_ = op1.cptr_at(cm);
    for (size_t d = 0; d < degree_; ++d)
      muladd.compute(*dst++, *op0_++, *op1_++, cm);
  }
  return *this;
}

template <size_t degree_, size_t nmoduli_>
void poly<degree_, nmoduli_>::negate() {
  for (size_t cm = 0; cm < nmoduli_; ++cm) {
    auto P = get_modulus(cm);
    std::transform(cptr_at(cm), cptr_end(cm), ptr_at(cm),
                   [P](value_type v) { return P - v; });
  }
}

template <size_t degree_, size_t nmoduli_>
void poly<degree_, nmoduli_>::automorphism_inplace(const size_t k) {
  assert((k & 1) && k < 2 * degree_);
  if (k == 1) 
    return;
  const size_t mod_mask = ((2 << logn) - 1);
  for (size_t cm = 0; cm < nmoduli_; ++cm) {
    std::array<value_type, degree_> tmp;
    auto src_ptr = cptr_at(cm);
    for (size_t d = 0; d < degree_; ++d) {
      uint64_t reversed = math::reverse_bits(d, logn);
      uint64_t index_raw = k * (2 * reversed + 1);
      index_raw &= mod_mask;
      tmp[d] = src_ptr[math::reverse_bits((index_raw - 1) >> 1, logn)];
    }
    std::memcpy(ptr_at(cm), tmp.begin(), sizeof(tmp));
  }
}

// ********************
// GMP
// ********************
template<size_t degree_, size_t nmoduli_>
std::array<mpz_t, degree_> poly<degree_, nmoduli_>::poly2mpz() const {
  return gmp::poly2mpz<degree_, nmoduli_>(*this);
}

// ****************************************************
// NTT related 
// ****************************************************
template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::forward() {
#pragma omp parallel num_threads(4)
#pragma omp for 
    for (size_t cm = 0; cm < nmoduli; ++cm)
      ntt<degree>::forward(ptr_at(cm), cm);
}

template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::forward_lazy() {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (size_t cm = 0; cm < nmoduli; ++cm) 
    ntt<degree>::forward_lazy(ptr_at(cm), cm);
}
 
template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::backward() {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (size_t cm = 0; cm < nmoduli; ++cm)
    ntt<degree>::backward(ptr_at(cm), cm);
}

// ****************************************************
// Random polynomial generation functions
// ****************************************************

// Sets a pre-allocated random polynomial in FFT form
// uniformly random, else the coefficients are uniform below the bound

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(uniform const& u) : poly<degree, nmoduli>() {
  set(u);
}

template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::set(uniform const &) {
  // In uniform mode we need randomness for all the polynomials in the CRT
  for (size_t cm = 0; cm < nmoduli; ++cm)
    fastrandombytes((unsigned char *) ptr_at(cm), degree * sizeof(value_type));

  for (unsigned int cm = 0; cm < nmoduli; cm++) {
    // In the uniform case, instead of getting a big random (within the general
    // moduli), We rather prefer, for performance issues, to get smaller
    // randoms for each module The mask should be the same for all moduli
    // (because they are the same size) But for generality we prefer to compute
    // it for each moduli so that we could have moduli of different bitsize

    value_type mask = (1ULL << (int)(std::floor(std::log2(get_modulus(cm))) + 1)) - 1;

    auto dst = ptr_at(cm);
    for (size_t i = 0; i < degree; i++) {
      // First remove the heavy weight bits we dont need
      value_type tmp = *dst & mask;

      // When the random is still too large, reduce it
      if (tmp >= get_modulus(cm))
        tmp -= get_modulus(cm);
      *dst++ = tmp;
    }
  }

#ifdef DEVELOP_CHECK
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    auto _ptr = cptr_at(cm);
    for (size_t d = 0; d < degree; ++d)
      assert(*_ptr++ < get_modulus(cm));
  }
#endif
}

template<size_t degree, size_t nmoduli>
template<class in_class, unsigned _lu_depth>
poly<degree, nmoduli>::poly(gaussian<in_class, _lu_depth> const& mode) : poly<degree, nmoduli>() {
  set(mode);
}

template<size_t degree, size_t nmoduli>
template<class in_class, unsigned _lu_depth>
void poly<degree, nmoduli>::set(gaussian<in_class, _lu_depth> const& mode) {
#ifndef NDEBUG
#ifdef NOISE_FREE
  clear();
  return;
#endif
#endif

  uint64_t const amplifier = mode.amplifier;

  // We play with the rnd pointer (in the uniform case), and thus
  // we need to remember the allocated pointer to free it at the end
  signed_type rnd[degree];

  // Get some randomness from the PRNG
  mode.fg_prng->getNoise((value_type *)rnd, degree);

  if (amplifier != 1) {
    for (signed_type & r : rnd) 
      r *= amplifier;
  }
  for (size_t cm = 0; cm < nmoduli; cm++) {
    auto p = get_modulus(cm);
    auto dst = ptr_at(cm);
    for (auto r : rnd)
      *dst++ = r < 0 ? p + r : r;
  }
#ifdef DEVELOP_CHECK
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    auto _ptr = cptr_at(cm);
    for (size_t d = 0; d < degree; ++d)
      assert(*_ptr++ < get_modulus(cm));
  }
#endif
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(ZO_dist const& mode) : poly<degree, nmoduli>() {
  set(mode);
}

template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::set(ZO_dist const& mode) {
  uint8_t rnd[degree];
  fastrandombytes(rnd, sizeof(rnd));
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    const value_type p = get_modulus(cm);
    const value_type p_min_1 = p - 1U;
    auto dst = ptr_at(cm);
    for (auto r : rnd)
      *dst++ = r <= mode.rho ? (r & 2) ? p_min_1 : 1U : 0U;
  }
#ifdef DEVELOP_CHECK
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    auto _ptr = cptr_at(cm);
    for (size_t d = 0; d < degree; ++d)
      assert(*_ptr++ < get_modulus(cm));
  }
#endif
}

template<size_t degree, size_t nmoduli>
poly<degree, nmoduli>::poly(hwt_dist const& mode) : poly<degree, nmoduli>() {
  set(mode);
}

template<size_t degree, size_t nmoduli>
void poly<degree, nmoduli>::set(hwt_dist const& mode) {
  assert(mode.hwt > 0 && mode.hwt <= degree);
  std::vector<size_t> hitted(mode.hwt);
  std::iota(hitted.begin(), hitted.end(), 0U); // select the first hwt positions.
  std::vector<size_t> rnd(hitted.size());
  auto rnd_end = rnd.end();
  auto rnd_ptr = rnd_end;
  /* Reservoir Sampling: uniformly select hwt coefficients. */
  for (size_t k = mode.hwt; k < degree; ++k) 
  {
    size_t pos = 0;
    size_t reject_sample = std::numeric_limits<size_t>::max() / k;
    /* sample uniformly from [0, k) using reject sampling. */
    for (;;) {
      if (rnd_ptr == rnd_end)
      {
        fastrandombytes((unsigned char *)rnd.data(), rnd.size() * sizeof(size_t));
        rnd_ptr = rnd.begin();
      }
      pos = *rnd_ptr++;
      if (pos <= reject_sample * k) {
        pos %= k;
        break;
      }
    }
    if (pos < mode.hwt)
      hitted[pos] = k;
  }

  std::sort(hitted.begin(), hitted.end()); // for better locality ?
  clear(); // clear up all
  fastrandombytes((unsigned char *)rnd.data(), rnd.size() * sizeof(size_t));
  for (size_t cm = 0, offset = 0; cm < nmoduli; ++cm) {
    rnd_ptr = rnd.begin();
    const value_type p = get_modulus(cm);
    const value_type p_min_1 = p - 1u;
    auto dst = ptr_at(cm);
    for (size_t pos : hitted)
      dst[pos] = ((*rnd_ptr++) & 2) ? 1 : p_min_1; // {-1, 1}
  }
  std::memset(hitted.data(), 0x0, hitted.size() * sizeof(size_t)); // erase from memory
#ifdef DEVELOP_CHECK
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    auto _ptr = cptr_at(cm);
    for (size_t d = 0; d < degree; ++d)
      assert(*_ptr++ < get_modulus(cm));
  }
#endif
}

// *********************************************************
// Helper functions
// *********************************************************

template<size_t degree, size_t nmoduli>
std::ostream& operator<<(std::ostream& outs, poly<degree, nmoduli> const& p)
{
  std::string term = "ULL";
  outs << "{";
  for (size_t cm = 0; cm < nmoduli; ++cm) {
    auto ptr = p.cptr_at(cm);
    outs << "{";
    for (size_t i = 0; i + 1 < degree; ++i) {
      outs << *ptr++ << term << ", ";
    }
    outs << *ptr++ << term << "}";
    if (cm + 1 < nmoduli)
      outs << "\n";
  }
  outs << "}";
  return outs;
}
} // namespace nfl
