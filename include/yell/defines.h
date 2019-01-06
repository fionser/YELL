#pragma once
#if defined(YELL_USE_32BITS_MODULI) && defined(YELL_USE_AVX)
#undef YELL_USE_MEM_POOL
#define YELL_USE_AVX_NTT
#define YELL_AVX_BATCH_SIZE 8
#include "yell/avx2.h"
#endif

#ifndef YELL_USE_AVX
#define YELL_NTT_UNROOL_SIZE 3
#else
#define YELL_NTT_UNROOL_SIZE (YELL_AVX_BATCH_SIZE - 1)
#endif
