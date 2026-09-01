#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "math_pswf.h"

// Small, deterministic bridge to the exact MathPSWF implementation used by
// the development ESP code.  The first output line contains the normalization
// constants.  Each subsequent input value x produces psi_0^c(x) and its
// integral from 0 to x.  The Python driver only requests 0 <= x <= 1; the
// outside continuation is evaluated from the finite-Fourier identity.
int main(int argc, char **argv)
{
  if (argc != 2) return 2;
  const double c = std::atof(argv[1]);
  const double psi0 = LAMMPS_NS::MathPSWF::prolate0_eval(c, 0.0);
  const double psi1 = LAMMPS_NS::MathPSWF::prolate0_eval(c, 1.0);
  const double c0 = LAMMPS_NS::MathPSWF::prolate0_int_eval(c, 1.0);
  // At x=0, lambda*psi(0)=2*C0 for the even ground-state PSWF.
  const double lambda = 2.0 * c0 / psi0;
  std::cout << std::setprecision(17);
  std::cout << "constants " << psi0 << ' ' << psi1 << ' ' << c0 << ' ' << lambda << '\n';

  double x;
  while (std::cin >> x) {
    if (x < 0.0 || x > 1.0) return 3;
    std::cout << x << ' ' << LAMMPS_NS::MathPSWF::prolate0_eval(c, x) << ' '
              << LAMMPS_NS::MathPSWF::prolate0_int_eval(c, x) << '\n';
  }
  return 0;
}
