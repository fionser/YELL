#pragma once
#include "yell/ops.hpp"
#include "yell/utils/math.hpp"

namespace yell {
montgomery montgomery::instance_;
montgomery::mot_precomputed::mot_precomputed(const size_t cm) 
  : w(params::kModulusRepresentationBitsize),
    cm(cm),
    prime(params::P[cm]),
    mod_tab({prime, 0})
{
  //! mu = -p^{-1} \bmod 2^w
  const gT ONE{1};
  const gT r = ONE << w;
  T y = 1;
  auto N = params::P[cm];
  for (int i = 2; i <= w; ++i) {
    auto mask = (T) ((ONE << i) - 1);
    auto d = (N * y) & mask;
    if (d != 1)
      y += (1ULL) << (i - 1);
  }
  mu = (T) (r - y);
  //! invR = (2^{w})^{-1} \bmod p
  ops::mulmod mulmod;
  auto inv_2w = math::inv_mod_prime<T>(1L << (w - 1), cm); // 2^{-w+1}
  auto inv_2  = math::inv_mod_prime<T>(1L << 1, cm); // 2^{-1}
  invR = mulmod(inv_2w, inv_2, cm);
  shoupInvR = ops::shoupify(invR, cm);
}

void montgomery::mot_precomputed::to_montgomery(T *op, const size_t degree) const
{
  if (!op) return;
  const size_t cm = this->cm;
  const size_t  w = this->w;
  std::transform(op, op + degree, op, 
                 [w, cm](T x) {
                   auto k = (gT) x << w; // 2^w * x mod p
                   yell::ops::barret_reduction(&k, cm);
                   assert(k < yell::params::P[cm]);
                   return (T) k;
                 });
}

void montgomery::mot_precomputed::from_montgomery(T* op, const size_t degree) const
{
  if (!op) return;
  yell::ops::mulmod_shoup mulmod;
  //! 2^{-w} * x mod p
  for (size_t d = 0; d < degree; ++d)
    mulmod.compute(*op++, invR, shoupInvR, cm);
}

const montgomery::mot_precomputed* montgomery::init_table(const size_t cm) 
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
  // double-check, the table might be updated when waiting for the W-
  if (kv != instance_.tables.end()) {
    instance_.guard.unlock();
    return kv->second;
  }

  auto *tbl = new mot_precomputed(cm);
  instance_.tables.insert({cm, tbl});
  instance_.guard.unlock();
  return tbl;
}
} // namespace yell
