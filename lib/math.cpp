#include <cassert>
namespace yell { namespace math {

uint32_t reverse_bits(uint32_t b)
{
  //! from bit twiddling hacks
  return ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
}

/* The operand is less than 32 */

uint32_t reverse_bits(uint32_t operand, int32_t bit_count)
{
  assert(bit_count < 32);
  return (uint32_t) (((uint64_t) reverse_bits(operand)) >> (32 - bit_count));
}
}} // namespace yell::math
