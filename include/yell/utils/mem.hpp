#pragma once
#if 0
#include <gperftools/tcmalloc.h>
#include <memory>
#include <cassert>

void * mem_alloc(size_t bytes) {
  void *p = tc_malloc(bytes);
  assert(p);
  return p;
}

void mem_free(void *p) {
  if (p) tc_free(p);
}

#else
#include <memory>
#include <stdlib.h>
#include <cassert>

void * mem_alloc(size_t bytes) {
  void* res;
  const int failed = posix_memalign(&res, 32, bytes);
  if (failed) res = 0;
  return res;
}

void mem_free(void *p) {
  if (p) free(p);
}
#endif
