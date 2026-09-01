#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "math_pswf.h"

using LAMMPS_NS::MathPSWF::fourier_poly;
using LAMMPS_NS::MathPSWF::spread_fourier_poly;
using LAMMPS_NS::MathPSWF::spread_real_poly;

static void dump(const char *kind, double lambda, const std::vector<double> &coeff)
{
  std::cout << kind << ",lambda," << std::setprecision(17) << lambda << "\n";
  for (std::size_t i = 0; i < coeff.size(); ++i)
    std::cout << kind << "," << i << "," << std::setprecision(17) << coeff[i] << "\n";
}

int main(int argc, char **argv)
{
  if (argc != 6) return 2;
  const double split_tol = std::atof(argv[1]);
  const double spread_tol = std::atof(argv[2]);
  const double csplit = std::atof(argv[3]);
  const double cspread = std::atof(argv[4]);
  const int order = std::atoi(argv[5]);
  double lambda = 0.0;
  std::vector<double> coeff;
  fourier_poly(split_tol, csplit, lambda, coeff);
  dump("split", lambda, coeff);
  coeff.clear();
  spread_fourier_poly(spread_tol, cspread, lambda, coeff);
  dump("spread", lambda, coeff);
  coeff.clear();
  spread_real_poly(order, spread_tol, cspread, coeff);
  dump("real", static_cast<double>(order), coeff);
  return 0;
}
