#ifndef MATH_PSWF_H
#define MATH_PSWF_H

#include <unordered_map>
#include <vector>

static constexpr int MAX_CHEB_ORDER = 40;
static constexpr int MAX_MONO_ORDER = 40;

// prolate functions
void prolc180(double eps, double &c);
void prolc180_der3(double eps, double &der3);
// prolate0 functor
struct Prolate0Fun;

double prolate0_eval_derivative(double c, double x);
/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
double prolate0_eval(double c, double x);

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
double prolate0_int_eval(double c, double r);

// approximation functions
void force_poly(double tol, double r_tol, const double &c, std::vector<double> &coeffs);
void energy_poly(double tol, double r_tol, const double &c, std::vector<double> &coeffs);
void fourier_poly(double tol, double r_tol, const double &c, double &lambda,
                  std::vector<double> &coeffs);
void spread_fourier_poly(double tol, double r_tol, const double &c, double &lambda,
                         std::vector<double> &coeffs);
void spread_real_poly(int P, double tol, double r_tol, const double &c,
                      std::vector<double> &coeffs);

#endif    // MATH_PSWF_H
