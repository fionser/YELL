#include "yell/Rq.h"
#include "yell/Zq.h"

#include "yell/prng/fastrandombytes.h"
#include "yell/prng/FastGaussianNoise.hpp"

#include "yell/util/arith.h"
#include "yell/util/mem.h"
#include "yell/util/safe_ptr.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace yell {
static void RevBinPermute(U64 *array, size_t length);

inline static U64 select0(U64 b, bool cond) { return (b & -(U64)cond) ^ b; }

struct DFTTable {
 public:
  explicit DFTTable(size_t degree, U64 q) : degree_(degree), zq_(CachedZq::Get(q)) {
    if (!(degree > 1 && arith::is_power_of_2(degree))) {
      throw std::invalid_argument("DFTTable: invalid degree");
    }
    generate();
  }

  ~DFTTable() {
    util::mem_free(roots_);
    util::mem_free(roots_shoup_);
    util::mem_free(inv_roots_);
    util::mem_free(inv_roots_shoup_);
    roots_ = nullptr;
    roots_shoup_ = nullptr;
    inv_roots_ = nullptr;
    inv_roots_shoup_ = nullptr;
  }

  std::array<const U64 *, 2> Roots() const { return std::array<const U64 *, 2>{roots_, roots_shoup_}; }

  std::array<const U64 *, 2> InvRoots() const { return std::array<const U64 *, 2>{inv_roots_, inv_roots_shoup_}; }

  std::array<U64, 2> InvDegree() const { return std::array<U64, 2>{inv_degree_, inv_degree_shoup_}; }

  ::yell::Zq const &Zq() const { return zq_; }

 private:
  void computeTwiddleFactor(U64 *tbl, const U64 w) const {
    assert(tbl);
    U64 t = w;
    tbl[0] = 1UL;
    for (size_t i = 1; i < degree_; ++i) {
      assert(t != 1UL);
      tbl[i] = t;
      t = zq_.mulmod_barrett(t, w);
    }

    if ((t + 1) != zq_.Moduli()) {
      throw std::logic_error("compute_twiddle_factor: w is not 2n-th root");
    }

    RevBinPermute(tbl, degree_);
  }

  void sequentialReorder(U64 *array, size_t n) {
    std::vector<U64> tmp(n);
    U64 *ptr = tmp.data() + 1;
    for (size_t i = n / 2; i > 0; i /= 2) {
      for (size_t j = i; j < i * 2; ++j) *ptr++ = array[j];
    }
    for (size_t i = 1; i < n; ++i) array[i] = tmp[i];
  }

  void generate() {
    inv_degree_ = zq_.inv(degree_);
#ifdef YELL_MONTGOMERY_NTT
    inv_degree_ = zq_.FromMontgomery(inv_degree_);
#endif
    inv_degree_shoup_ = zq_.Shoupify(inv_degree_);

    roots_ = (U64 *)util::mem_alloc(sizeof(U64) * degree_);
    roots_shoup_ = (U64 *)util::mem_alloc(sizeof(U64) * degree_);
    inv_roots_ = (U64 *)util::mem_alloc(sizeof(U64) * degree_);
    inv_roots_shoup_ = (U64 *)util::mem_alloc(sizeof(U64) * degree_);

    U64 w;
    if (!zq_.find_2nth_root_of_unity(&w, degree_)) {
      throw std::runtime_error("DFTTable: could not find 2n-th root of unity");
    }

    computeTwiddleFactor(roots_, w);
    computeTwiddleFactor(inv_roots_, zq_.inv(w));
#ifdef YELL_MONTGOMERY_NTT
    // convert the last-half to Montgomery domain
    for (size_t i = degree_ / 2; i < degree_; ++i) {
      roots_[i] = zq_.ToMontgomery(roots_[i]);
    }
#endif

    // reordering so that the access pattern at inverse NTT is sequential.
    // merge n^(-1)
    inv_roots_[1] = zq_.mulmod_shoup(inv_roots_[1], inv_degree_, inv_degree_shoup_);
    sequentialReorder(inv_roots_, degree_);

    auto Shoupify = [this](U64 x) -> U64 { return zq_.Shoupify(x); };
    std::transform(roots_, roots_ + degree_, roots_shoup_, Shoupify);
    std::transform(inv_roots_, inv_roots_ + degree_, inv_roots_shoup_, Shoupify);
  }

  size_t degree_;
  ::yell::Zq zq_;
  U64 inv_degree_, inv_degree_shoup_;
  U64 *roots_{nullptr};
  U64 *roots_shoup_{nullptr};
  U64 *inv_roots_{nullptr};
  U64 *inv_roots_shoup_{nullptr};
};

struct CachedDFTTables {
 public:
  ~CachedDFTTables();

  static std::shared_ptr<DFTTable> Get(size_t degree, U64 p);

 private:
  explicit CachedDFTTables();
  CachedDFTTables(CachedDFTTables &&oth) = delete;
  CachedDFTTables(CachedDFTTables const &oth) = delete;
  CachedDFTTables &operator=(CachedDFTTables oth) = delete;

  using key_t = std::pair<size_t, U64>;

  struct hasher {
    size_t operator()(const key_t &k) const {
      U32 u0 = static_cast<U32>(k.second);
      U32 u1 = static_cast<U32>(k.second >> 32u);
      U32 u2 = static_cast<U32>(k.first);
      std::hash<U32> h;
      return h(u0) ^ h(u1) ^ h(u2);
    }
  };

  static sf::contention_free_shared_mutex<> lock;
  static std::unordered_map<key_t, std::shared_ptr<DFTTable>, hasher> tables;
};

std::unordered_map<CachedDFTTables::key_t, std::shared_ptr<DFTTable>, CachedDFTTables::hasher> CachedDFTTables::tables;

sf::contention_free_shared_mutex<> CachedDFTTables::lock;

void RevBinPermute(U64 *array, size_t length) {
  if (!array || length <= 2) return;
  for (size_t i = 1, j = 0; i < length; ++i) {
    size_t bit = length >> 1;
    for (; j >= bit; bit >>= 1) {
      j -= bit;
    }
    j += bit;

    if (i < j) {
      std::swap(array[i], array[j]);
    }
  }
}

std::shared_ptr<DFTTable> CachedDFTTables::Get(size_t degree, U64 p) {
  lock.lock_shared();
  key_t key{degree, p};
  auto kv = tables.find(key);
  if (kv != tables.end()) {
    lock.unlock_shared();
    return kv->second;
  }

  lock.unlock_shared();
  lock.lock();
  kv = tables.find(key);
  if (kv != tables.end()) {
    lock.unlock();
    return kv->second;
  }

  auto table = std::make_shared<DFTTable>(degree, p);
  tables.insert({key, table});
  lock.unlock();
  return table;
}

Rq::Rq() {}

Rq::Rq(size_t degree, U64 moduli, RqForm form) : moduli_(moduli), form_(form), degree_(degree) {
  coeff_u64_.resize(degree_);
}

Rq::Rq(Rq &&oth) : Rq() { swap(*this, oth); }

Rq::Rq(Rq const &oth) : moduli_(oth.moduli_), form_(oth.form_), degree_(oth.degree_), coeff_u64_(oth.coeff_u64_) {}

Rq &Rq::operator=(Rq oth) {
  std::swap(*this, oth);
  return *this;
}

Rq &Rq::Redc() {
  Zq zq = CachedZq::Get(moduli_);
  std::transform(cbegin(), cend(), begin(), [&zq](U64 v) -> U64 { return zq.redc_barret(v); });
  return *this;
}

Rq::~Rq() {}

Rq Rq::Zero(size_t degree, U64 moduli, RqForm form) {
  Rq zero(degree, moduli, form);
  std::fill(zero.begin(), zero.end(), 0);
  return zero;
}

Rq Rq::Uniform(size_t degree, U64 moduli, RqForm form) {
  Rq u(degree, moduli, form);
  fastrandombytes((unsigned char *)u.begin(), degree * sizeof(U64));
  // we sample from [0, 2^63) then apply redc_barret()
  std::transform(u.cbegin(), u.cend(), u.begin(), [](U64 v) -> U64 { return v & 0x7FFFFFFFFFFFFFFFULL; });
  u.Redc();
  return u;
}

Rq Rq::Ternary(size_t degree, U64 moduli, RqForm form) {
  Rq u(degree, moduli, RqForm::PowerBasis);
  fastrandombytes((unsigned char *)u.begin(), degree * sizeof(U64));

  // Note : (2^64 - 1) is multipler of 3.
  // So, sample x from [1, 2^64), then x % 3 is uniform in [0, 1, 2]
  std::transform(u.cbegin(), u.cend(), u.begin(), [](U64 v) -> U64 { return ((v | 1) % 3) - 1; });
  u.Redc();
  if (form == RqForm::NTTBasis)
    u.NTT();
  return u;
}

Rq Rq::Normal(double stdv, size_t degree, U64 moduli, RqForm form) {
  if (stdv < 0.)
    throw std::invalid_argument("Rq::Normal: requires positive stdv");
  constexpr int security = 40;
  constexpr int samples = 1 << 20;
  std::vector<int32_t> rnd(degree);

  Rq u(degree, moduli, RqForm::PowerBasis);
  FastGaussianNoise<uint16_t, int32_t, 2> fgn(stdv, security, samples);
  fgn.getNoise(rnd.data(), degree);

  std::transform(rnd.cbegin(), rnd.cend(), u.begin(), [moduli](int32_t v) -> U64 {
                 if (v == 0) return 0;
                 return v < 0 ? moduli + v : v;
                 });

  std::fill(rnd.begin(), rnd.end(), 0);

  u.Redc();
  if (form == RqForm::NTTBasis)
    u.NTT();
  return u;
}

Rq Rq::Constant(U64 cnst, size_t degree, U64 moduli, RqForm form) {
  Rq u = Rq::Zero(degree, moduli, RqForm::PowerBasis);
  u.SetCoeff(0, cnst % moduli);
  if (form == RqForm::NTTBasis)
    u.NTT();
  return u;
}

Rq Rq::Assign(Rq const &oth, U64 moduli) {
  Rq u{oth};

  if (oth.Moduli() != moduli) { 
    if (oth.Form() == RqForm::PowerBasis) {
      u.moduli_ = moduli;
      u.Redc();
    } else {
      u.INTT(); // work over oth.Moduli()
      u.moduli_ = moduli;
      u.Redc().NTT();
    }
  }

  return u;
}

void Rq::SetCoeff(size_t pos, U64 value) {
  if (pos >= degree_) {
    throw std::invalid_argument("Rq::SetCoeff: pos out of bound");
  }
  if (value >= moduli_) {
    throw std::invalid_argument("Rq::SetCoeff: value out of bound");
  }
  coeff_u64_[pos] = value;
}

Rq &Rq::Add(Rq const &oth) {
  if (!IsValidOp(oth)) {
    throw std::invalid_argument("Rq::Add: incompatible operand");
  }

  auto zq = CachedZq::Get(moduli_);
  std::transform(cbegin(), cend(), oth.cbegin(), begin(), [&zq](U64 a, U64 b) -> U64 { return zq.addmod(a, b); });
  return *this;
}

Rq &Rq::Sub(Rq const &oth) {
  if (!IsValidOp(oth)) {
    throw std::invalid_argument("Rq::Sub: incompatible operand");
  }

  auto zq = CachedZq::Get(moduli_);
  std::transform(cbegin(), cend(), oth.cbegin(), begin(), [&zq](U64 a, U64 b) -> U64 { return zq.submod(a, b); });
  return *this;
}

Rq &Rq::PMul(Rq const &oth) {
  if (!IsValidOp(oth)) {
    throw std::invalid_argument("Rq::PMul: incompatible operand");
  }

  if (form_ != RqForm::NTTBasis) {
    throw std::invalid_argument("Rq::PMul: requires RqForm::NTTBasis");
  }

  auto zq = CachedZq::Get(moduli_);
  if (IsMontgomery()) {
    std::transform(cbegin(), cend(), oth.cbegin(), begin(),
                   [&zq](U64 a, U64 b) -> U64 { return zq.mulmod_mont(a, b); });
  } else {
    std::transform(cbegin(), cend(), oth.cbegin(), begin(),
                   [&zq](U64 a, U64 b) -> U64 { return zq.mulmod_barrett(a, b); });
  }
  return *this;
}

Rq &Rq::Negate() {
  auto zq = CachedZq::Get(moduli_);
  std::transform(cbegin(), cend(), begin(), [&zq](U64 a) -> U64 { return zq.negate(a); });
  return *this;
}

Rq &Rq::AddScalar(U64 v) {
  if (form_ != RqForm::PowerBasis) {
    throw std::invalid_argument("Rq::AddScalar: requires RqForm::PowerBasis");
  }

  auto zq = CachedZq::Get(moduli_);
  if (v >= moduli_) v = zq.redc_barret(v);

  std::transform(cbegin(), cend(), begin(), [&zq, v](U64 a) -> U64 { return zq.addmod(a, v); });
  return *this;
}

Rq &Rq::SubScalar(U64 v) {
  if (form_ != RqForm::PowerBasis) {
    throw std::invalid_argument("Rq::SubScalar: requires RqForm::PowerBasis");
  }

  auto zq = CachedZq::Get(moduli_);
  if (v >= moduli_) v = zq.redc_barret(v);

  std::transform(cbegin(), cend(), begin(), [&zq, v](U64 a) -> U64 { return zq.submod(a, v); });
  return *this;
}

Rq &Rq::MulScalar(U64 v) {
  auto zq = CachedZq::Get(moduli_);
  if (v >= moduli_) v = zq.redc_barret(v);

  U64 v_s = zq.Shoupify(v);
  std::transform(cbegin(), cend(), begin(), [&zq, v, v_s](U64 a) -> U64 { return zq.mulmod_shoup(a, v, v_s); });

  return *this;
}

const U64 *Rq::cbegin() const {
  if (coeff_u64_.empty()) {
    throw std::invalid_argument("Rq::cbegin: empty coefficients");
  }
  return coeff_u64_.data();
}

U64 *Rq::begin() {
  if (coeff_u64_.empty()) {
    throw std::invalid_argument("Rq::begin: empty coefficients");
  }
  return coeff_u64_.data();
}

void swap(Rq &dst, Rq &src) {
  std::swap(dst.moduli_, src.moduli_);
  std::swap(dst.form_, src.form_);
  std::swap(dst.degree_, src.degree_);
  std::swap(dst.coeff_u64_, src.coeff_u64_);
}

struct NTTBody {
  const Zq &zq;
  const U64 Lq;
  NTTBody(const Zq &zq, size_t L) : zq(zq), Lq(L * zq.Moduli()) {}

  /**
   * x0 = x0 + w * x1
   * x1 = x0 - w * x1
   */
  inline void Forward(U64 *x0, U64 *x1, U64 w, U64 w_s) const {
    U64 u0 = *x0;
    U64 u1 = *x1;

    U64 t = zq.mulmod_shoup<Zq::LazyRdc>(u1, w, w_s);
    *x0 = u0 + t;
    *x1 = u0 + Lq - t;
  }

  inline void ForwardLast(U64 *x0, U64 *x1, U64 w, U64 w_s) const {
    // x0 \in [0, K*p) -> [0, 2p)
#ifdef YELL_MONTGOMERY_NTT
    U64 u0 = zq.ToMontgomery(*x0);
#else
    U64 u0 = zq.redc_barret<Zq::LazyRdc>(*x0);
#endif
    U64 u1 = *x1;
    U64 t = zq.mulmod_shoup<Zq::LazyRdc>(u1, w, w_s);
    *x0 = u0 + t;
    *x1 = u0 + Lq - t;
  }

  // x0 = x0 + x1
  // x1 = w * (x0 - x1)
  inline void Backward(U64 *x0, U64 *x1, U64 w, U64 w_s) const {
    U64 u0 = *x0;
    U64 u1 = *x1;

    *x0 = u0 + u1;
    *x1 = zq.mulmod_shoup<Zq::LazyRdc>(u0 + Lq - u1, w, w_s);
  }

  inline void BackwardCorrect(U64 *x0, U64 *x1, U64 w, U64 w_s) const {
    U64 u0 = zq.redc_barret<Zq::LazyRdc>(*x0);
    U64 u1 = zq.redc_barret<Zq::LazyRdc>(*x1);

    *x0 = u0 + u1;
    *x1 = zq.mulmod_shoup<Zq::LazyRdc>(u0 + Lq - u1, w, w_s);
  }

  inline void BackwardLast(U64 *x0, U64 *x1, U64 w0, U64 w0_s, U64 w1, U64 w1_s) const {
    U64 u0 = *x0;
    U64 u1 = *x1;

    *x0 = zq.mulmod_shoup<Zq::FullRdc>(u0 + u1, w0, w0_s);
    *x1 = zq.mulmod_shoup<Zq::FullRdc>(u0 + Lq - u1, w1, w1_s);
  }

  inline void BackwardCorrectLast(U64 *x0, U64 *x1, U64 w0, U64 w0_s, U64 w1, U64 w1_s) const {
    U64 u0 = zq.redc_barret<Zq::LazyRdc>(*x0);
    U64 u1 = zq.redc_barret<Zq::LazyRdc>(*x1);

    *x0 = zq.mulmod_shoup<Zq::LazyRdc>(u0 + u1, w0, w0_s);
    *x1 = zq.mulmod_shoup<Zq::LazyRdc>(u0 + Lq - u1, w1, w1_s);
  }
};

Rq &Rq::convertToNTT() {
  if (form_ == RqForm::NTTBasis) return *this;
  auto dftTable = CachedDFTTables::Get(degree_, moduli_);
  size_t L = 2;

  NTTBody body(dftTable->Zq(), L);

  U64 *const x = begin();
  auto roots = dftTable->Roots();
  const U64 *w = roots[0] + 1;
  const U64 *w_s = roots[1] + 1;

  // invariant: h * m = degree_ / 2
  {
    size_t m = 1;
    size_t h = degree_ / 2;
    for (; h > 2; m <<= 1, h >>= 1) {
      auto x0 = x;
      auto x1 = x0 + h;
      for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
        for (size_t i = 0; i < h; i += 4) {
          body.Forward(x0++, x1++, *w, *w_s);
          body.Forward(x0++, x1++, *w, *w_s);
          body.Forward(x0++, x1++, *w, *w_s);
          body.Forward(x0++, x1++, *w, *w_s);
        }
        x0 += h;
        x1 += h;
      }
    }
  }

  {
    const size_t m = degree_ / 4;
    constexpr size_t h = 2;
    auto x0 = x;
    auto x1 = x0 + h;
    for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
      body.Forward(x0++, x1++, *w, *w_s);
      body.Forward(x0, x1, *w, *w_s);
      x0 += 3;
      x1 += 3;
    }
  }

  {
    const size_t m = degree_ / 2;
    constexpr size_t h = 1;
    auto x0 = x;
    auto x1 = x + h;
    for (size_t r = 0; r < m; ++r) {
      body.ForwardLast(x0, x1, *w++, *w_s++);
      x0 += 2;
      x1 += 2;
    }
  }

  // values in [0, 4p)

  form_ = RqForm::NTTBasis;
  return *this;
}

Rq &Rq::convertFromNTT() {
  if (form_ != RqForm::NTTBasis) return *this;
  auto dftTable = CachedDFTTables::Get(degree_, moduli_);
  auto inv_roots = dftTable->InvRoots();
  const U64 *w = inv_roots[0] + 1;
  const U64 *w_s = inv_roots[1] + 1;

  constexpr size_t L = 1UL << (Zq::kWordSize - Zq::kMaxModuliBit - 1);

  NTTBody body(dftTable->Zq(), L);
  U64 *const x = begin();

  {
    const size_t m = degree_ / 2;
    constexpr size_t h = 1;
    auto x0 = x;
    auto x1 = x0 + h;
    for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
      body.BackwardCorrect(x0, x1, *w, *w_s);
      x0 += 2;
      x1 += 2;
    }
  }

  {
    const size_t m = degree_ / 4;
    constexpr size_t h = 2;
    auto x0 = x;
    auto x1 = x0 + h;
    for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
      body.Backward(x0++, x1++, *w, *w_s);
      body.Backward(x0, x1, *w, *w_s);
      x0 += 3;
      x1 += 3;
    }
  }

  {
    size_t m = degree_ / 8;
    size_t h = 4;
    for (; m >= (degree_ / L); m >>= 1, h <<= 1) {
      auto x0 = x;
      auto x1 = x0 + h;
      for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
        for (size_t i = 0; i < h; i += 4) {
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
        }
        x0 += h;
        x1 += h;
      }
    }

    // m > 1 to skip the last layer
    for (; m > 1; m >>= 1, h <<= 1) {
      auto x0 = x;
      auto x1 = x0 + h;
      const size_t h0 = (degree_ / L) / (2 * m);
      for (size_t r = 0; r < m; ++r, ++w, ++w_s) {
        size_t i = 0;
        for (; i < h0; i += 4) {
          body.BackwardCorrect(x0++, x1++, *w, *w_s);
          body.BackwardCorrect(x0++, x1++, *w, *w_s);
          body.BackwardCorrect(x0++, x1++, *w, *w_s);
          body.BackwardCorrect(x0++, x1++, *w, *w_s);
        }

        for (; i < h; i += 4) {
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
          body.Backward(x0++, x1++, *w, *w_s);
        }

        x0 += h;
        x1 += h;
      }
    }
  }

  {
    const U64 inv_n = dftTable->InvDegree()[0];
    const U64 inv_n_s = dftTable->InvDegree()[1];
    constexpr size_t m = 1;
    const size_t h = degree_ / 2;
    auto x0 = x;
    auto x1 = x0 + h;
    for (size_t i = 0; i < h; i += 4) {
      body.BackwardLast(x0++, x1++, inv_n, inv_n_s, *w, *w_s);
      body.BackwardLast(x0++, x1++, inv_n, inv_n_s, *w, *w_s);
      body.BackwardLast(x0++, x1++, inv_n, inv_n_s, *w, *w_s);
      body.BackwardLast(x0++, x1++, inv_n, inv_n_s, *w, *w_s);
    }
  }

  form_ = RqForm::PowerBasis;
  return *this;
}
}  // namespace yell

