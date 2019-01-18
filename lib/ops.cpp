#include "yell/params.hpp"
#include <cassert>
#include <cstddef>
namespace yell{ namespace ops {
void barret_reduction(params::gt_value_type *x, size_t cm)
{
  if (!x) return;
  using T  = params::value_type;
  using gT = params::gt_value_type;
  using ST = params::signed_type;

  const gT _x = *x;
  const T pm = params::Pn[cm];
  constexpr size_t shift = params::kModulusRepresentationBitsize;
  constexpr size_t delta = shift - params::kModulusBitsize;
  gT q = ((gT) pm * (_x >> shift)) + (_x << delta) ;

  const T p[2] = {params::P[cm], 0};
  T r = (T) _x - ((T) (q >> shift)) * p[0];
  r -= p[r < p[0]]; //! if (r >= params::P[cm]) r -= params::P[cm];
  *x = (gT) r;
} 

yell::params::value_type shoupify(yell::params::value_type x, size_t cm)
{
  using gT = params::gt_value_type;
  return ((gT) x << params::kModulusRepresentationBitsize) / params::P[cm];
}
}} // namespace yell::ops
