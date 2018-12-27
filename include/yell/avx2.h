#pragma once
#include <immintrin.h>
//! multiply the packed 64 bit integers in A, B;
//! and store the high 64-bit result.
__m256i avx_mm256_mul64_hi(__m256i const& A, __m256i const& B);
