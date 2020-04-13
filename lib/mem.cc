#include "yell/util/mem.h"
#include <new>
namespace yell {
namespace util {

static constexpr size_t AVX_MEMORY_ALIGNMENT = 32;

// compute bit reverse of i of length len
// ex.) bit_reverse(0x05, 8) = 0xa0
size_t bit_reverse(const size_t i, const size_t len) {
  size_t v = i, r_ = i;
  size_t s = len - 1;

  for (v >>= 1U; v; v >>= 1U) {
    r_ <<= 1U;
    r_ |= v & 1U;
    s--;
  }
  return (r_ << s) & ((1UL << len) - 1);
}

void *mem_alloc(const size_t size) {
#ifdef __USE_ISOC11
  void *ptr = aligned_alloc(AVX_MEMORY_ALIGNMENT, size);
  if (ptr == nullptr) throw std::bad_alloc();
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, AVX_MEMORY_ALIGNMENT, size)) throw std::bad_alloc();
#endif
  return ptr;
}

void mem_free(void *p) {
  if (!p) {
    free(p);
  }
}

}  // namespace util
}  // namespace yell
