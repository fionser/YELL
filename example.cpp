#include "yell/poly.hpp"
#include "yell/utils/timer.hpp"
constexpr size_t Deg = 8192;
constexpr size_t NdM = 2;

namespace global {
  using gauss_struct = yell::gaussian<uint16_t, yell::params::value_type, 2>;
  using gauss_t = yell::FastGaussianNoise<uint16_t, yell::params::value_type, 2>;
  gauss_t fg_prng(3.2, 128, 1 << 14);
}

struct SK {

  explicit SK() : sx(NdM, yell::hwt_dist(64)) {
    sx.forward();
  }

  yell::poly<Deg> sx;
};

struct PK {

  explicit PK(SK &sk) : bx(NdM), ax(NdM, yell::uniform{}) {
    bx.set(global::gauss_struct(&global::fg_prng));
    bx.forward();
    bx.add_product_of(ax, sk.sx);
    bx.negate();
  }

  yell::poly<Deg> bx, ax;
};

struct CTXT {
  yell::poly<Deg> bx, ax;

  explicit CTXT(yell::poly<Deg> const& msg, PK const& pk)
    :bx(NdM), ax(NdM)
  {
    yell::poly<Deg> u(NdM, yell::ZO_dist{});
    u.forward();
    bx.set(global::gauss_struct(&global::fg_prng));
    bx += msg;
    bx.forward();
    bx.add_product_of(u, pk.bx);

    ax.set(global::gauss_struct(&global::fg_prng));
    ax.forward();
    ax.add_product_of(u, pk.ax);
  }
};

void decrypt(yell::poly<Deg> *rop, CTXT const& ctx, SK const& sk)
{
  if (!rop) return;
  (*rop) = ctx.bx;
  rop->add_product_of(sk.sx, ctx.ax);
  rop->backward();
}

static uint64_t compute_montgomery(const uint64_t p) {
  uint64_t e = -(1UL << 63U) - 1; // Z/(2^64)Z^* has a multiplicative order of (2^64 - 2^63)
  uint64_t a = -p, r = 1;
  while(e > 0) {
    if(e & 1U) r *= a;
    a *= a;
    e >>= 1U;
  }
  return r;
}

int main() {
  double btime{0.}, mtime{0.};

  for (int _i = 0; _i < 100; _i++) {
    yell::poly<Deg> c0(NdM, yell::uniform{});
    yell::poly<Deg> c1(NdM, yell::uniform{});

    yell::poly<Deg> c2(NdM);

    for (size_t cm = 0; cm < c2.moduli_count(); ++cm) {
      AutoTimer timer(&btime);
      yell::ops::mulmod mulmod;
      auto op0 = c0.cptr_at(cm);
      auto op1 = c1.cptr_at(cm);
      auto dst = c2.ptr_at(cm);
      for (size_t d = 0; d < Deg; ++d) {
        *dst++ = mulmod(*op0++, *op1++, cm);
      }
    }

    yell::poly<Deg> c3(NdM);
    for (size_t cm = 0; cm < c2.moduli_count(); ++cm) {
      uint64_t m = compute_montgomery(yell::params::P[cm]);
      AutoTimer timer(&mtime);
      yell::ops::mulmod_mont mulmod(m);
      auto op0 = c0.cptr_at(cm);
      auto op1 = c1.cptr_at(cm);
      auto dst = c3.ptr_at(cm);
      for (size_t d = 0; d < Deg; ++d) {
        *dst++ = mulmod(*op0++, *op1++, cm);
      }
    }
  }

  std::cout << btime << " " << mtime << "\n";

  {
    SK sk;
    PK pk(sk);
    yell::poly<Deg> msg(NdM, yell::uniform{});

    CTXT ctx(msg, pk);
    yell::poly<Deg> plain(NdM);
    auto coeffs = plain.poly2mpz();

    decrypt(&plain, ctx, sk);

    std::cout << msg(0, 0) << " " << plain(0, 0) << "\n";
  }
  return 0;
}
