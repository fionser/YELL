#pragma once
#include "yell/types.h"

namespace yell {
namespace util {

void *mem_alloc(size_t bytes);
void mem_free(void *p);

}  // namespace util
}  // namespace yell

