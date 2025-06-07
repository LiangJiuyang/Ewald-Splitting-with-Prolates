#ifndef MATH_PSWF_H
#define MATH_PSWF_H

#include <iostream>
#include <unordered_map>
#include <vector>

#define Long int64_t
#define MAX_MONO_ORDER 20

// blas, lapack math functions used
extern "C" {
void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda, double* s, double* u,
             int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* info);
void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s, double* u,
             int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
void dgemm_(char* TransA, char* TransB, int* M, int* N, int* K, double* alpha, double* A, int* lda,
            double* B, int* ldb, double* beta, double* C, int* ldc);
}

// prolate functions
void prolc180_der3(double eps, double& der3);
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
void force_poly(double tol, int order, double &c, std::vector<double>& coeffs);
void energy_poly(double tol, int order, double &c, std::vector<double>& coeffs);
void fourier_poly(double tol, int order, double &c, double &lambda, std::vector<double>& coeffs);
void spread_fourier_poly(double tol, int order, double &c, double &lambda, std::vector<double>& coeffs);
void spread_real_poly(int P, double tol, int order, double& c, std::vector<double>& coeffs);
#endif  // MATH_PSWF_H
