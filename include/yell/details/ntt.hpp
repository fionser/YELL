#pragma once
#include <cassert>
#include <cstring>
#include "yell/ops.hpp"
#include "yell/meta.hpp"
#include "yell/params.hpp"
#include "yell/utils/math.hpp"
namespace yell {
//! @see lib/ntt.cpp
void negacylic_forward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *w,
  const params::value_type *wshoup,
  const params::value_type p);

//! @see lib/ntt.cpp
void negacylic_backward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *w,
  const params::value_type *wshoup,
  const params::value_type inv_n,
  const params::value_type inv_n_s,
  const params::value_type p);

template<size_t degree_> ntt<degree_> ntt<degree_>::instance_;

void compute_twiddle_factor_table(params::value_type *tbl, size_t degree, params::value_type w0, size_t cm) {
  params::value_type temp{1UL};
  ops::mulmod mulmod;
  for (unsigned int i = 0; i < degree; i++) {
    tbl[i] = temp;
    mulmod.compute(temp, w0, cm);
  }
  math::revbin_permute(tbl, degree);
}

template <size_t degree>
void ntt<degree>::ntt_precomputed::init(size_t cm) {
  assert(cm < params::kMaxNbModuli);
  const T prime  = yell::params::P[cm];
  ops::mulmod mulmod;
  ops::mulmod_shoup mulmod_s;

  T r = static_cast<T>((-prime) % prime); // r= 2^64 mod p
  T rshoup = ops::shoupify(r, cm);

  // (n * 2^64)^(-1) mod p
  invDegree = mulmod_s(degree, r, rshoup, cm);
  invDegree = math::inv_mod_prime(invDegree, cm);
  shoupInvDegree = ops::shoupify(invDegree, cm);

  T phi;
  phi = params::primitive_roots[cm];
  for (unsigned int i = 0 ; i < static_log2<params::kMaxPolyDegree>::value - static_log2<degree>::value; i++) {
    mulmod.compute(phi, phi, cm);
  }

  const T invphi = math::inv_mod_prime(phi, cm);
  
  compute_twiddle_factor_table(phiTbl, degree, phi, cm);
  compute_twiddle_factor_table(invphiTbl, degree, invphi, cm);

  for (size_t i = 1; i < degree / 2; ++i) {
    shoupPhiTbl[i] = ops::shoupify(phiTbl[i], cm);
  }
  // merge r mod p for the last layer of NTT
  for (size_t i = degree / 2; i < degree; ++i) {
    mulmod_s.compute(phiTbl[i], r, rshoup, cm);
    shoupPhiTbl[i] = ops::shoupify(phiTbl[i], cm);
  }

  // merge n^{-1} step of the last layer of invNTT
  mulmod_s.compute(invphiTbl[1], invDegree, shoupInvDegree, cm);
  for (size_t i = 1; i < degree; ++i)
    shoupInvphiTbl[i] = ops::shoupify(invphiTbl[i], cm);

  // Reordering inv_phi so that the access pattern at iNTT is sequential.
  std::vector<T> tmp(degree);
  uint64_t *ptr = tmp.data() + 1;
  for (size_t i = degree / 2; i > 0; i >>= 1) {
    for (size_t j = i; j < i * 2; ++j)
      *ptr++ = invphiTbl[j];
  }
  std::copy(tmp.cbegin(), tmp.cend(), invphiTbl);

  ptr = tmp.data() + 1;
  for (size_t i = degree / 2; i > 0; i >>= 1) {
    for (size_t j = i; j < i * 2; ++j)
      *ptr++ = shoupInvphiTbl[j];
  }
  std::copy(tmp.cbegin(), tmp.cend(), shoupInvphiTbl);
}

template<size_t degree> const typename ntt<degree>::ntt_precomputed* 
ntt<degree>::init_table(size_t cm)
{
  assert(cm < params::kMaxNbModuli);
  instance_.guard.lock_shared();
  auto kv = instance_.tables.find(cm);
  if (kv != instance_.tables.end()) {
    instance_.guard.unlock_shared();
    return kv->second;
  }

  instance_.guard.unlock_shared(); // release the R-lock
  instance_.guard.lock(); // apply the W-lock
  kv = instance_.tables.find(cm); 
  // double-check, the table might be updated when waiting for the W-lock
  if (kv != instance_.tables.end()) {
    instance_.guard.unlock();
    return kv->second;
  }
  
  ntt_precomputed *tbl = new ntt_precomputed();
  tbl->init(cm);
  instance_.tables.insert({cm, tbl});
  instance_.guard.unlock();
  return tbl;
}

template <size_t degree>
void ntt<degree>::forward(T *op, size_t cm)
{
  forward_lazy(op, cm);
  const T p = params::P[cm];
  T mod_table[2][2] = {{2 * p, 0}, 
                       {    p, 0}};
  std::transform(op, op + degree, op,
                 [&mod_table, p](T v) { 
                 v -= mod_table[0][v < 2 * p];
                 v -= mod_table[1][v < p];
                 return v;
                 });
// #ifndef NDEBUG
  for (size_t d = 0; d < degree; ++d)
    assert(op[d] < yell::params::P[cm]);
// #endif
}

template <size_t degree>
void ntt<degree>::forward_lazy(T *op, size_t cm)
{
  assert(op && cm < params::kMaxNbModuli);
  const auto tbl = init_table(cm);
  const T *phiTbl = tbl->phiTbl;
  const T *shoupPhiTbl = tbl->shoupPhiTbl;

  negacylic_forward_lazy(op, degree, phiTbl, shoupPhiTbl, params::P[cm]);
}

template <size_t degree>
void ntt<degree>::backward(T *op, size_t cm)
{
  assert(op && cm < params::kMaxNbModuli);
  const auto tbl = init_table(cm);
  const T p = params::P[cm];
  const T *invphiTbl = tbl->invphiTbl;
  const T *shoupInvphiTbl= tbl->shoupInvphiTbl;

  negacylic_backward_lazy(op, degree, 
                          invphiTbl, shoupInvphiTbl, 
                          tbl->invDegree, tbl->shoupInvDegree,
                          params::P[cm]);

  T mod_table[2]{p, 0};
  std::transform(op, op + degree, op,
                 [&mod_table, p](T v) -> T { 
                 v -= mod_table[v < p];
                 return v;
                 });

// #ifndef NDEBUG
  for (size_t d = 0; d < degree; ++d)
    assert(op[d] < yell::params::P[cm]);
// #endif
  //! final result should range in [0, p)
}
} // namespace yell
