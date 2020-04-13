#pragma once
#include <unordered_map>
#include "yell/types.h"
#include "yell/util/arith.h"
#include "yell/utils/safe_ptr.h"
namespace yell {

template <typename T>
struct Reducer;

template <>
struct Reducer<std::true_type> {
  static inline U64 reduce(U64 a, U64 p) { return a - arith::select0(p, a < p); }
};

template <>
struct Reducer<std::false_type> {
  static inline U64 reduce(U64 a, U64) { return a; }
};

class Zq {
 private:
  friend class CachedZq;
  explicit Zq(U64 q); // Access via CachedZq::Get()

 public:
  static constexpr int kWordSize = 64;
  static constexpr int kMaxModuliBit = 59;

  using LazyRdc = std::false_type;
  using FullRdc = std::true_type;

  Zq(const Zq& oth);

  inline U64 addmod(const U64 a, const U64 b) const { 
    assert(a < q_ && b < q_);
    return Reducer<FullRdc>::reduce(a + b, q_); 
  }

  inline U64 submod(const U64 a, const U64 b) const { 
    assert(a < q_ && b < q_);
    return Reducer<FullRdc>::reduce(a + q_ - b, q_); 
  }

  U64 Shoupify(const U64 a) const;

  U64 negate(const U64 a) const { 
    assert(a < q_);
    return Reducer<FullRdc>::reduce(q_ - a, q_); 
  }

  I64 Signed(U64 a) const;

  template <typename lazy = FullRdc>
  inline U64 redc_barret(U128 a) const {
    const U64 lo64 = a.u64[0];
    const U64 hi64 = a.u64[1];

    U128 t;
    arith::mul_u64_u64(&t, barretParm_, hi64);
    arith::lshiftu128(&a, barretShift_);

    arith::add_u128_u128(&t, t, a);
    return Reducer<lazy>::reduce(lo64 - t.u64[1] * q_, q_);
  }

  template <typename lazy = FullRdc>
  inline U64 redc_barret(U64 a) const {
    U128 t;
    t.u64[0] = a;
    t.u64[1] = 0;
    arith::lshiftu128(&t, barretShift_);
    return Reducer<lazy>::reduce(a - t.u64[1] * q_, q_);
  }

  template <typename lazy = FullRdc>
  inline U64 redc_mont(U128 const& t) const {
    U128 c;
    U64 u;
    u = t.u64[0] * montgomeryParm_;
    arith::mul_u64_u64(&c, u, q_);
    arith::add_u128_u128(&c, c, t);
    return Reducer<lazy>::reduce(c.u64[1], q_);
  }

  template <typename lazy = FullRdc>
  inline U64 mulmod_barrett(const U64 a, const U64 b) const {
    assert(a < q_ &&  b < q_);
    U128 c;
    arith::mul_u64_u64(&c, a, b);
    return redc_barret<lazy>(c);
  }

  template <typename lazy = FullRdc>
  inline U64 mulmod_mont(U64 a, U64 b) const {
    assert(a < q_ &&  b < q_);
    U128 t;
    arith::mul_u64_u64(&t, a, b);
    return redc_mont<lazy>(t);
  }

  template <typename lazy = FullRdc>
  inline U64 mulmod_shoup(U64 x, U64 y, U64 yshoup) const {
    assert(x < q_ &&  y < q_);
    U128 t;
    arith::mul_u64_u64(&t, x, yshoup);
    return Reducer<lazy>::reduce(x * y - t.u64[1] * q_, q_);
  }

  U64 pow(U64 a, U64 e) const;

  U64 inv(U64 a) const { return pow(a, q_ - 2); }

  inline U64 ToMontgomery(U64 a) const { return mulmod_shoup(a, Rq_, Rqshoup_); }

  inline U64 FromMontgomery(U64 a) const { return mulmod_mont(a, 1); }

  inline U64 Moduli() const { return q_; }

  bool find_2nth_root_of_unity(U64* u, size_t n) const;

 private:
  bool is_2nth_root_of_unity(U64 a, size_t n) const;

  U64 q_;
  // -(q^(-1)) mod 2^64
  U64 montgomeryParm_;
  // 2^64 mod q
  U64 Rq_, Rqshoup_;
  // lo64(floor(2^128 / q))
  U64 barretParm_;
  // 64 - floor(log2(q))
  size_t barretShift_;
};

class CachedZq {
 public:
  static Zq Get(U64 q);

  ~CachedZq();

 private:
  explicit CachedZq();
  CachedZq(CachedZq&& oth) = delete;
  CachedZq(CachedZq const& oth) = delete;
  CachedZq& operator=(CachedZq oth) = delete;

  static sf::contention_free_shared_mutex<> lock;
  static std::unordered_map<U64, Zq> memo;
};

}  // namespace yell

