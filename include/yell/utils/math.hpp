#pragma once
#include "yell/ops.hpp"
#include <cmath>
namespace math {
// compute a^{-1} mod prime
// i.e., a^{prime - 2}
template <typename T>
T inv_mod_prime(T a, size_t cm)
{
  yell::ops::mulmod mulmod;
  T ret{1};
  T y = yell::params::P[cm] - 2;
  while (y > 0) {
    if (y & 1)
      ret = mulmod(ret, a, cm);
    y >>= 1u;
    a = mulmod(a, a, cm);
  }
  return ret;
}

template <typename T, typename F>
inline T round(F a)
{
  return a >= 0 ? (T)(a + 0.5f) : (T) (a - 0.5f);
}

inline bool is_same(double a, double b) 
{
  if (std::abs(a - b) < 1e-8)
    return true;
  return std::abs(std::abs((a - b)) / std::max(a, b)) < 1e-8;
}

/* Reverse 32-bit */
uint32_t reverse_bits(uint32_t operand) 
{
  operand = (((operand & 0xaaaaaaaa) >> 1) | ((operand & 0x55555555) << 1));
  operand = (((operand & 0xcccccccc) >> 2) | ((operand & 0x33333333) << 2));
  operand = (((operand & 0xf0f0f0f0) >> 4) | ((operand & 0x0f0f0f0f) << 4));
  operand = (((operand & 0xff00ff00) >> 8) | ((operand & 0x00ff00ff) << 8));
  return((operand >> 16) | (operand << 16));
}

/* The operand is less than 32 */
uint32_t reverse_bits(uint32_t operand, int32_t bit_count)  
{
  assert(bit_count < 32);
  return (uint32_t) (((uint64_t) reverse_bits(operand)) >> (32 - bit_count));
}
} // namespace math

