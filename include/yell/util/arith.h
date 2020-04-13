#pragma once
#include "yell/types.h"

namespace yell {
namespace arith {

// return 0 if cond is true, else return x
inline U64 select0(U64 x, bool cond) { return (x & -static_cast<U64>(cond)) ^ x; }

inline void mul_u64_u64(U128 *u, U64 a, U64 b) { u->u128 = static_cast<decltype(u->u128)>(a) * b; }

inline void add_u128_u128(U128 *u, U128 const &a, U128 const &b) { u->u128 = a.u128 + b.u128; }

inline void add_u128_u64(U128 *u, U128 const &a, U64 const b) { u->u128 = a.u128 + b; }

inline void lshiftu128(U128 *u, size_t s) { u->u128 = u->u128 << s; }

// floor(2^64 * a / p)
inline U64 shoupify(U64 a, U64 p) {
  U128 _a;
  _a.u64[0] = a;
  _a.u64[1] = 0;
  return static_cast<U64>((_a.u128 << 64U) / p);
}

// compute bit reverse of i of length len
// ex.) bit_reverse(0x05, 8) = 0xa0
size_t bit_reverse(const size_t i, const size_t len);

constexpr bool is_power_of_2(size_t N) {
  return (N & (N - 1)) == 0;
}

};  // namespace arith
};  // namespace yell

