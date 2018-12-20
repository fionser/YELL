#pragma once
#include "yell/utils/safe_ptr.h"
#include <gmp.h>
#include <array>
#include <unordered_map>

namespace yell {
template <size_t degree, size_t nmoduli> class poly; // forward declaration
class gmp {
private:
  using T = typename params::value_type;

  gmp() {}
  gmp(gmp const& oth) = delete;
  gmp& operator=(gmp const& oth) = delete;

  ~gmp() {
    for (auto kv : tables) delete kv.second;
  }
  static gmp instance_; // singleton

  struct gmp_precomputed {
    gmp_precomputed(size_t nmoduli);
    ~gmp_precomputed();
    mpz_t moduli_product_;
    mpz_t modulus_shoup;
    size_t bits_in_moduli_product;
    size_t bits_in_modulus_shoup;
    size_t shift_modulus_shoup;
    mpz_t *lifting_integers;
    const size_t nmoduli_;
  };

  sf::contention_free_shared_mutex<> guard;
  std::unordered_map<size_t, gmp_precomputed *> tables;

public:
  static const gmp_precomputed* init_table(size_t nmoduli);

  template <size_t degree_, size_t nmoduli_>
  static std::array<mpz_t, degree_> poly2mpz(poly<degree_, nmoduli_> const& op);
}; 
} // namespace yell

#include "yell/gmp_impl.hpp"
