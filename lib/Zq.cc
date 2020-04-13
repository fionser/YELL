#include "yell/Zq.h"
#include <gmpxx.h>
#include <iostream>
namespace yell {

std::unordered_map<U64, Zq> CachedZq::memo;
sf::contention_free_shared_mutex<> CachedZq::lock;

Zq CachedZq::Get(U64 q) {
  lock.lock_shared();
  auto kv = memo.find(q);
  if (kv != memo.end()) {
    lock.unlock_shared();
    return kv->second;
  }

  lock.unlock_shared();
  lock.lock();
  // two-stage opt lock
  kv = memo.find(q);
  if (kv != memo.end()) {
    lock.unlock();
    return kv->second;
  }

  Zq zq(q);
  memo.emplace(q, zq);
  lock.unlock();
  return zq;
}

Zq::Zq(U64 q) : q_(q) {
  const int nbits = static_cast<int>(std::ceil(std::log2(q * 1.)));
  if (nbits > kMaxModuliBit) {
    throw std::invalid_argument("Zq: moduli out-of-bound");
  }

  montgomeryParm_ = [](U64 p) {
    U64 e = -(1UL << 63U) - 1;
    U64 a = -p, r = 1;
    while (e > 0) {
      if (e & 1U) r *= a;
      a *= a;
      e >>= 1U;
    }
    return r;
  }(q);

  Rq_ = static_cast<U64>((-q_) % q_);
  Rqshoup_ = arith::shoupify(Rq_, q_);

  assert((q * montgomeryParm_) + 1 == 0);

  mpz_t u;
  mpz_init2(u, 128);
  mpz_set_ui(u, 1);
  mpz_mul_2exp(u, u, 128);
  mpz_div_ui(u, u, q);
  barretParm_ = static_cast<U64>(mpz_get_ui(u));

  barretShift_ = kWordSize - nbits;
  mpz_clear(u);
}

Zq::Zq(const Zq& oth)
    : q_(oth.q_),
      montgomeryParm_(oth.montgomeryParm_),
      Rq_(oth.Rq_),
      Rqshoup_(oth.Rqshoup_),
      barretParm_(oth.barretParm_),
      barretShift_(oth.barretShift_) {}

U64 Zq::pow(U64 a, U64 e) const {
  U64 r{1};
  while (e > 0) {
    if (e & 1u) r = mulmod_barrett(r, a);
    a = mulmod_barrett(a, a);
    e >>= 1u;
  }
  return r;
}

bool Zq::find_2nth_root_of_unity(U64* u, size_t n) const {
  constexpr U64 MaxNiterRootFinding = 1UL << 20U;
  if (!u) return false;
  const U64 e = (q_ - 1) / static_cast<U64>(n << 1u);

  U64 a;
  for (U64 a_ = 2; a_ < MaxNiterRootFinding; ++a_) {
    a = a_;
    if (is_2nth_root_of_unity(a, n)) {
      *u = a;
      return true;
    }

    a = pow(a, e);
    if (is_2nth_root_of_unity(a, n)) {
      *u = a;
      return true;
    }
  }

  return false;
}

bool Zq::is_2nth_root_of_unity(U64 a, size_t n) const {
  assert(n > 2 && arith::is_power_of_2(n));
  int log2n = static_cast<int>(std::log2(1. * n));
  for (int i = 0; i <= log2n; ++i) {
    if (a == 1) return false;
    a = mulmod_barrett(a, a);
  }
  return a == 1;
}

U64 Zq::Shoupify(const U64 a) const {
  assert(a < q_);
  U128 _a;
  _a.u64[0] = a;
  _a.u64[1] = 0;
  return static_cast<U64>((_a.u128 << 64U) / q_);
}

I64 Zq::Signed(U64 a) const {
  assert(a < q_);
  if (a >= q_ / 2)
    return static_cast<I64>(a) - static_cast<I64>(q_);
  return static_cast<I64>(a);
}

}  // namespace yell
