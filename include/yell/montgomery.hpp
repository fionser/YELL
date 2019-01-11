#pragma once
#include "yell/params.hpp"
#include "yell/utils/safe_ptr.h"
#include <unordered_map>
namespace yell {
class montgomery {
private:
  using T = params::value_type;
  using gT = params::gt_value_type;

  montgomery() {}
  montgomery(montgomery const& oth) = delete;
  montgomery(montgomery && oth) = delete;
  montgomery& operator=(montgomery const& oth) = delete;

  ~montgomery() {
    for (auto kv : tables) delete kv.second;
  }
  static montgomery instance_; // singleton

  struct mot_precomputed {
  public:
    explicit mot_precomputed(const size_t cm);
    ~mot_precomputed() {}

    int w; //! R := 2^w
    size_t cm;
    T prime;
    std::array<T, 2> mod_tab;
    T mu; //! -p^{-1} mod 2^w
    T invR; //! 2^{-w} mod prime
    T shoupInvR;
    void to_montgomery(T *op, const size_t degree) const;
    void from_montgomery(T *op, const size_t degree) const;

    //! lazy reduce the to [0, 2 * prime)
    inline T reduce_lazy(const gT P) const {
      T q = (T) P * mu; // mod 2^w
      return (T)((P + (gT) q * yell::params::P[cm]) >> w);
    } 

    //! reduce the to [0, prime)
    inline T reduce(const gT P) const {
      T r = reduce_lazy(P);
      r -= mod_tab[r < prime];
      return r;
    }

    //! Use lazy reduction in mulmod. The result range in [0, 2 * prime).
    inline void mulmod_lazy(T *rop, T const* op0, T const* op1, 
                       const size_t degree) const 
    {
      if (!rop || !op0 || !op1) return;
      for (size_t d = 0; d < degree; ++d)
        *rop++ = reduce_lazy((gT) (*op0++) * (*op1++));
    }

    //! Use full reduction in mulmod. The result range in [0, prime).
    inline void mulmod(T *rop, T const* op0, T const* op1, 
                       const size_t degree) const 
    {
      if (!rop || !op0 || !op1) return;
      for (size_t d = 0; d < degree; ++d)
        *rop++ = reduce((gT) (*op0++) * (*op1++));
    }
  };

  sf::contention_free_shared_mutex<> guard;
  std::unordered_map<size_t, mot_precomputed *> tables;

public:
  static const mot_precomputed* init_table(size_t cm);
};
} // namespace yell

#include "yell/details/montgomery.hpp"
