#pragma once
#include "yell/ops.hpp"
#include "yell/meta.hpp"
#include <cstring>
namespace yell {
template<size_t degree> ntt<degree> ntt<degree>::instance_;

inline unsigned char add_uint64_generic(uint64_t operand1, 
                                        uint64_t operand2,
                                        unsigned char carry, 
                                        uint64_t * result)
{
  operand1 += operand2;
  *result = operand1 + carry;
  return (operand1 < operand2) || (~operand1 < carry);
}

static params::value_type div2mod(params::value_type x, size_t cm)
{
  if (x & 1) {
    params::value_type temp;
    int carry = add_uint64_generic(x, params::P[cm], 0, &temp);
    x = temp >> 1u;
    if (carry)
      return x | (params::value_type(1) << (params::kModulusRepresentationBitsize - 1));
    return x;
  }
  return x >> 1u;
}

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
    phis[i] = temp;
    shoupphis[i] = ops::shoupify(temp, cm);
    // phi**(i+1)
    mulmod.compute(temp, phi, cm);
  }
  // At the end of the loop temp = phi**degree

  // Computation of invphi
  // phi^(2*degree) = 1 -> temp * phi^(degree-1) = phi^(-1)
  const T invphi = mulmod(temp, phis[degree - 1], cm);

  // Computation of the inverse of degree using the inverse of kMaxPolyDegree
  invDegree = mulmod(params::invkMaxPolyDegree[cm], (uint64_t)(params::kMaxPolyDegree/degree), cm);
  shoupinvDegree = ops::shoupify(invDegree, cm);

  temp = 1;
  for (unsigned int i = 0 ; i < degree; i++) {
    invphis[i] = temp;
    shoupinvphis[i] = ops::shoupify(invphis[i], cm);
    mulmod.compute(temp, invphi, cm);
  }

  // Bit-reverse phis and inv_phis
  params::value_type tmp[degree + 1];
  permute<degree>::compute(tmp, phis);
  std::memcpy(phis, tmp, sizeof(phis));

  permute<degree>::compute(tmp, shoupphis);
  std::memcpy(shoupphis, tmp, sizeof(phis));

  permute<degree>::compute(tmp, invphis);
  std::memcpy(invphis, tmp, sizeof(phis));

  permute<degree>::compute(tmp, shoupinvphis);
  std::memcpy(shoupinvphis, tmp, sizeof(phis));
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

static void negacylic_forward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *w,
  const params::value_type *wshoup,
  const params::value_type p);

static void negacylic_backward_lazy(
  params::value_type *x, 
  const size_t degree,
  const params::value_type *w,
  const params::value_type *wshoup,
  const params::value_type p);

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
}

template <size_t degree>
void ntt<degree>::forward_lazy(T *op, size_t cm)
{
  assert(op && cm < params::kMaxNbModuli);
  const auto tbl = init_table(cm);
  T *dst = op;
  const T *phi = tbl->phis;
  const T *shoup_phi = tbl->shoupphis;

  negacylic_forward_lazy(op, degree, phi, shoup_phi, params::P[cm]);
}

template <size_t degree>
void ntt<degree>::backward(T *op, size_t cm)
{
  assert(op && cm < params::kMaxNbModuli);
  const auto tbl = init_table(cm);
  T *dst = op;
  const T *invphi = tbl->invphis;
  const T *shoup_invphi = tbl->shoupinvphis;
  const T p = params::P[cm];
  const T mod_table[2] = {p, 0};

  negacylic_backward_lazy(op, degree, invphi, shoup_invphi, p);
  std::transform(op, op + degree, op,
                 [&mod_table, p](T v) { 
                   v -= mod_table[v < p];
                   return v;
                 });
  // TODO: we might elimite this degree^-1 step.
  const T inv_degree = tbl->invDegree;
  const T shoup_inv_degree = tbl->shoupinvDegree;
  ops::mulmod_shoup mulmod;
  for (size_t d = 0; d < degree; ++d)
    mulmod.compute(op[d], inv_degree, shoup_inv_degree, cm);
}

struct ntt_loop_body {
  using value_type = params::value_type;
  using signed_type = params::signed_type;
  using gt_value_type = params::gt_value_type;
  const value_type p;
  const std::array<value_type, 2> mod_correct_table;

  explicit ntt_loop_body(value_type const p) : p(p), mod_correct_table({p * 2, 0}) {}

  //! x'0 = x0 + x1 mod p
  //! x'1 = w * (x0 - x1) mod p
  //! Require: 0 < x0, x1 < 2 * p
  //! Ensure:  0 < x'0, x'1 < 2 * p
  inline void gs_bufferfly(value_type* x0, 
                           value_type* x1, 
                           value_type const *w, 
                           value_type const *wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    value_type t0 = u0 + u1;
    t0 -= mod_correct_table[t0 < p * 2]; // if (t0 >= _2p) t0 -= 2p;
    value_type t1 = u0 - u1 + p * 2;
    value_type q = ((gt_value_type) t1 * (*wprime)) >> params::kModulusRepresentationBitsize;

    *x0 = t0;
    *x1 = t1 * (*w) - q * p;
  }

  void four_gs_butterfly(size_t j1, size_t j2,
                         value_type* x0,
                         value_type* x1,
                         value_type const* w,
                         value_type const* wprime) const 
  {
    for (size_t j = j1; j != j2; j += 4) {
      gs_bufferfly(x0++, x1++, w, wprime);
      gs_bufferfly(x0++, x1++, w, wprime);
      gs_bufferfly(x0++, x1++, w, wprime);
      gs_bufferfly(x0++, x1++, w, wprime);
    }
  }
  //! x'0 = x0 + w * x1 mod p
  //! x'1 = x0 - w * x1 mod p
  //! Require: 0 < x0, x1 < 4 * p
  //! Ensure:  0 < x'0, x'1 < 4 * p
  inline void ct_bufferfly(value_type* x0, 
                           value_type* x1, 
                           value_type const *w, 
                           value_type const *wprime) const
  {
    value_type u0 = *x0;
    value_type u1 = *x1;

    u0 -= mod_correct_table[u0 < p * 2]; // if (u0 >= 2p) u0 -= 2p;
    value_type q = ((gt_value_type) u1 * (*wprime)) >> params::kModulusRepresentationBitsize;
    value_type t = u1 * (*w) - q * p;

    *x0 = u0 + t;
    *x1 = u0 - t + p * 2;
  }

  void four_ct_butterfly(size_t j1, size_t j2,
                         value_type* x0, 
                         value_type* x1,
                         value_type const* w,
                         value_type const* wprime) const 
  {
    for (size_t j = j1; j != j2; j += 4) {
      ct_bufferfly(x0++, x1++, w, wprime);
      ct_bufferfly(x0++, x1++, w, wprime);
      ct_bufferfly(x0++, x1++, w, wprime);
      ct_bufferfly(x0++, x1++, w, wprime);
    }
  }

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
    if (t >= 4) {
      for (size_t i = 0; i != m; ++i) {
        const size_t j1 = 2 * i * t;
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        body.four_ct_butterfly(j1, j2, x0, x1, w, wshoup);
        x0 += 4u;
        x1 += 4u;
        ++w;
        ++wshoup;
      }
    } else {
      for (size_t i = 0; i != m; ++i) {
        const size_t j1 = 2 * i * t;
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        for (size_t j = j1; j != j2; ++j)
          body.ct_bufferfly(x0++, x1++, w, wshoup);
        ++w;
        ++wshoup;
      }
    }
  }
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
    size_t j1 = 0;
    const params::value_type *w = &wtab[h];
    const params::value_type *wshoup = &wtab_shoup[h];
    if (t >= 4) {
      for (size_t i = 0; i != h; ++i) {
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        //! Unroll a little bit to reduce the number of branches.
        body.four_gs_butterfly(j1, j2, x0, x1, w, wshoup);
        x0 += 4u;
        x1 += 4u;
        ++w;
        ++wshoup;
        j1 = j1 + (t << 1u);
      }
    } else { //! last two layers
      for (size_t i = 0; i != h; ++i) {
        const size_t j2 = j1 + t;
        auto x0 = &x[j1];
        auto x1 = &x[j2];
        for (size_t j = j1; j != j2; ++j)
          body.gs_bufferfly(x0++, x1++, w, wshoup);
        ++w;
        ++wshoup;
        j1 = j1 + (t << 1u);
      }
    }
    t <<= 1u;
  }
}

} // namespace yell
