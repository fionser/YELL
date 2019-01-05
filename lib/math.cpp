#include <cassert>
namespace yell { namespace math {

uint32_t reverse_bits(uint32_t v)
{
  //! taken from bit twiddling hacks
  // swap odd and even bits
  v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
  // swap consecutive pairs
  v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
  // swap nibbles ...
  v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
  // swap bytes
  v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
  // swap 2-byte long pairs
  v = ( v >> 16             ) | ( v               << 16);
  return v;
}

/* The operand is less than 32 */

uint32_t reverse_bits(uint32_t operand, int32_t bit_count)
{
  assert(bit_count < 32);
  return (uint32_t) (((uint64_t) reverse_bits(operand)) >> (32 - bit_count));
}
}} // namespace yell::math
