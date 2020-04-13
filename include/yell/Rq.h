#pragma once
#include <vector>
#include "yell/types.h"
#include "yell/util/safe_ptr.h"

#include <iostream>
#include <unordered_map>
namespace yell {

enum class RqForm : int { PowerBasis, NTTBasis };

class ArithOptimizer;

/**
 * Representation of Rq := Z_q[X] / (X^n + 1)
 */
class Rq {
 private:
  explicit Rq(size_t degree, U64 moduli, RqForm form);

  Rq();

  friend class ArithOptimizer;

 protected:
  inline bool IsValidOp(Rq const& oth) const {
    return degree_ == oth.degree_ && moduli_ == oth.moduli_ && form_ == oth.form_;
  }

  U64* begin();

  inline U64* end() { return begin() + degree_; }

  Rq& convertToNTT();

  Rq& convertFromNTT();

 public:
  const U64* cbegin() const;

  inline const U64* cend() const { return cbegin() + degree_; }

  static Rq Zero(size_t degree, U64 moduli, RqForm form);

  static Rq Ternary(size_t degree, U64 moduli, RqForm form);

  static Rq Uniform(size_t degree, U64 moduli, RqForm form);

  static Rq Normal(double variance, size_t degree, U64 moduli, RqForm form);

  static Rq Constant(U64 cnst, size_t degree, U64 moduli, RqForm form);

  static Rq Assign(Rq const& oth, U64 moduli);

  ~Rq();

  Rq(Rq&& oth);

  Rq(Rq const& oth);

  Rq& operator=(Rq oth);

  Rq& Redc();

  bool Equal(Rq const& oth) const {
    return IsValidOp(oth) && (0 == std::memcmp(cbegin(), oth.cbegin(), sizeof(U64) * degree_));
  }

  bool IsZero() const {
    return std::all_of(cbegin(), cend(), [](U64 v) { return 0 == v; });
  }

  constexpr bool IsMontgomery() const {
#ifdef YELL_MONTGOMERY_NTT
    return true;
#else
    return false;
#endif
  }

  bool operator==(Rq const& oth) const { return Equal(oth); }

  bool operator!=(Rq const& oth) const { return !Equal(oth); }

  friend void swap(Rq& dst, Rq& src);

  Rq& NTT() { return convertToNTT(); }

  Rq& INTT() { return convertFromNTT(); }

  Rq& Add(Rq const& oth);

  Rq& Sub(Rq const& oth);

  Rq& PMul(Rq const& oth);

  Rq& Negate();

  Rq& AddScalar(U64 v);

  Rq& MulScalar(U64 v);

  Rq& SubScalar(U64 v);

  void SetCoeff(size_t pos, U64 v);

  RqForm Form() const { return form_; }

  size_t Degree() const { return degree_; }

  U64 Moduli() const { return moduli_; }

 private:
  U64 moduli_;
  RqForm form_;
  size_t degree_;
  std::vector<U64> coeff_u64_;
};

}  // namespace yell

