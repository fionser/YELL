#include "yell/defines.h"
#include "yell/utils/mem.hpp"
#include <stdlib.h>
namespace yell {

// #ifndef YELL_USE_AVX_NTT
//
// void * mem_alloc(size_t bytes) {
//   void* res = malloc(bytes);
//   return res;
// }
// #else

void * mem_alloc(size_t bytes) {
  void* res;
  const int failed = posix_memalign(&res, 32, bytes);
  if (failed) res = 0;
  return res;
}
//#endif

void mem_free(void *p) {
  if (p) free(p);
}

} // namespace yell
