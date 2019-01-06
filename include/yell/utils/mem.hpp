#pragma once
#include <cstddef>
namespace yell {
void * mem_alloc(size_t bytes);
void mem_free(void *p);
} // namespace yell
