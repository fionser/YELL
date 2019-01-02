#pragma once
#include <gmp.h>
#include "yell/meta.hpp"
#include "yell/params.hpp"
namespace yell {
gmp gmp::instance_;

gmp::gmp_precomputed::gmp_precomputed(size_t nmoduli)
  : nmoduli_(nmoduli)
{
  mpz_init_set_ui(moduli_product_, 1);
  for (size_t j = 0; j < nmoduli; j++) {
    T p = yell::params::P[j];
    mpz_mul_ui(moduli_product_, moduli_product_, p);
  }

  bits_in_moduli_product = mpz_sizeinbase(moduli_product_, 2);

  // Compute Shoup value for optimized reduction modulo "moduli_product"
  shift_modulus_shoup = bits_in_moduli_product +
                        yell::params::kModulusRepresentationBitsize +
                        (T) std::log2(nmoduli) + 1;

  mpz_init2(modulus_shoup, shift_modulus_shoup);
  mpz_ui_pow_ui(modulus_shoup, 2, shift_modulus_shoup);
  mpz_tdiv_q(modulus_shoup, modulus_shoup, moduli_product_);

  bits_in_modulus_shoup = mpz_sizeinbase(modulus_shoup, 2);

  // Compute the lifting coefficients
  mpz_t quotient, current_modulus;
  mpz_inits(quotient, current_modulus, nullptr);

  lifting_integers = new mpz_t[nmoduli];
  for (size_t j = 0; j < nmoduli; j++) {
    // Current modulus
    T p = yell::params::P[j];
    mpz_set_ui(current_modulus, p);

    // compute the product of primes except the current one
    mpz_divexact(quotient, moduli_product_, current_modulus);

    // Compute the inverse of the product
    mpz_init2(lifting_integers[j], bits_in_moduli_product);
    mpz_invert(lifting_integers[j], quotient, current_modulus);

    // Multiply by the quotient
    mpz_mul(lifting_integers[j], lifting_integers[j], quotient);
  }

  // Clear
  mpz_clears(quotient, current_modulus, nullptr);
}

gmp::gmp_precomputed::~gmp_precomputed()
{
  mpz_clears(moduli_product_, modulus_shoup, nullptr);
  for (size_t i = 0; i < nmoduli_; ++i)
    mpz_clear(lifting_integers[i]);
  delete []lifting_integers;
}

const gmp::gmp_precomputed* gmp::init_table(size_t nmoduli)
{
  instance_.guard.lock_shared();
  auto kv = instance_.tables.find(nmoduli);
  if (kv != instance_.tables.end()) {
    instance_.guard.unlock_shared();
    return kv->second;
  }

  instance_.guard.unlock_shared(); // release the R-lock
  instance_.guard.lock(); // apply the W-lock
  kv = instance_.tables.find(nmoduli); 
  // double-check, the table might be updated when waiting for the W-lock
  if (kv != instance_.tables.end()) {
    instance_.guard.unlock();
    return kv->second;
  }

  auto tbl = new gmp_precomputed(nmoduli);
  instance_.tables.insert({nmoduli, tbl});
  instance_.guard.unlock();
  return tbl;
}

template <size_t degree_>
std::array<mpz_t, degree_> gmp::poly2mpz(poly<degree_> const& op)
{
  const size_t nmoduli_ = op.nmoduli;
  auto tbl = init_table(op.nmoduli_);
  std::array<mpz_t, degree_> rop;
  for (size_t i = 0; i < degree_; ++i)
    mpz_init2(rop[i], tbl->shift_modulus_shoup - 1);

  mpz_t tmp;
  mpz_init2(tmp, tbl->shift_modulus_shoup - 1 + tbl->bits_in_modulus_shoup);
  // Loop on each coefficient
  for (size_t i = 0; i < degree_; i++) {
    mpz_set_ui(rop[i], 0);
    for (size_t j = 0; j < nmoduli_; j++) {
      if (op(j, i) != 0)
        mpz_addmul_ui(rop[i], tbl->lifting_integers[j], op(j, i));
    }

    // Modular reduction using Shoup
    mpz_mul(tmp, rop[i], tbl->modulus_shoup);
    mpz_tdiv_q_2exp(tmp, tmp, tbl->shift_modulus_shoup);
    mpz_submul(rop[i], tmp, tbl->moduli_product_);
    if (mpz_cmp(rop[i], tbl->moduli_product_) >= 0) {
       mpz_sub(rop[i], rop[i], tbl->moduli_product_);
    }
  }
  // Clean
  mpz_clear(tmp);
  return rop;
}
} // namespace yell
