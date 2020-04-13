#pragma once
#include <cstdint>
#include <cstdlib>
namespace yell {
using U32 = uint32_t;
using U64 = uint64_t;
using I64 = int64_t;
using _U128 = unsigned __int128;

typedef union {
  U64 u64[2];
  unsigned __int128 u128;
} U128;

}  // namespace yell
