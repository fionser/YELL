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
  const params::value_type p);

template<size_t degree_> ntt<degree_> ntt<degree_>::instance_;

template <size_t degree>
void ntt<degree>::ntt_precomputed::init(size_t cm) {
  assert(cm < params::kMaxNbModuli);
  T phi, temp;
  ops::mulmod mulmod;
  // We start by computing phi
  // The roots in the array are primitve
  // X=2*params::kMaxPolyDegree-th roots
  // Squared log2(X)-log2(degree) times they become
  // degree-th roots as required by the NTT
  // But first we get phi = sqrt(omega) squaring them
  // log2(X/2)-log2(degree) times
  phi = params::primitive_roots[cm];
  for (unsigned int i = 0 ; i < static_log2<params::kMaxPolyDegree>::value 
                                - static_log2<degree>::value; i++) {
    mulmod.compute(phi, phi, cm);
  }

  // Now that temp = phi we initialize the array of phi**i values
  // Initialized to phi**0
  temp = 1;
  for (unsigned int i = 0 ; i < degree; i++) {
    phiTbl[i]      = temp;
    shoupPhiTbl[i] = ops::shoupify(temp, cm);
    // phi**(i+1)
    mulmod.compute(temp, phi, cm);
  }
  // At the end of the loop temp = phi**degree

  // Computation of invphi
  // phi^(2*degree) = 1 -> temp * phi^(degree-1) = phi^(-1)
  const T invphi = mulmod(temp, phiTbl[degree - 1], cm);
  const T prime  = yell::params::P[cm];
  temp = 1;
  for (unsigned int i = 0 ; i < degree; i++) {
    invphiTbl[i]      = temp;
    shoupInvphiTbl[i] = ops::shoupify(invphiTbl[i], cm);
    mulmod.compute(temp, invphi, cm);
  }

  // Bit-reverse phi and inv_phi
  math::revbin_permute(phiTbl, degree);
  math::revbin_permute(shoupPhiTbl, degree);
  math::revbin_permute(invphiTbl, degree);
  math::revbin_permute(shoupInvphiTbl, degree);

  invDegree            = math::inv_mod_prime(degree, cm);
  invphiInvDegree      = mulmod(invDegree, invphiTbl[1], cm);
  shoupInvDegree       = yell::ops::shoupify(invDegree, cm);
  shoupInvphiInvDegree = yell::ops::shoupify(invphiInvDegree, cm);
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
#ifndef NDEBUG
  for (size_t d = 0; d < degree; ++d)
    assert(op[d] < yell::params::P[cm]);
#endif
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

  negacylic_backward_lazy(op, degree, tbl->invphiTbl, tbl->shoupInvphiTbl, params::P[cm]);

  yell::ops::mulmod_shoup mulmod_s;
  const T invN = tbl->invDegree;
  const T invN_s = tbl->shoupInvDegree;
  const T invNinvPhi = tbl->invphiInvDegree;
  const T invNinvPhi_s = tbl->shoupInvphiInvDegree;
  const T _2p = yell::params::P[cm] << 1;

  T *x0 = op;
  T *x1 = op + degree / 2;
  T const* end = op + degree;
  while (x1 != end) {
      T u = *x0;
      T v = *x1;
      *x0++ = mulmod_s(u + v, invN, invN_s, cm);
      *x1++ = mulmod_s(u - v + _2p, invNinvPhi, invNinvPhi_s, cm);
  }

#ifndef NDEBUG
  for (size_t d = 0; d < degree; ++d)
    assert(op[d] < yell::params::P[cm]);
#endif
  //! final result should range in [0, p)
}
} // namespace yell
