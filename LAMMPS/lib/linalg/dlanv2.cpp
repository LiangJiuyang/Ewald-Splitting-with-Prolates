#ifdef __cplusplus
extern "C" {
#endif
#include "lmp_f2c.h"
static doublereal c_b3 = 1.;
int dlanv2_(doublereal *a, doublereal *b, doublereal *c__, doublereal *d__, doublereal *rt1r,
            doublereal *rt1i, doublereal *rt2r, doublereal *rt2i, doublereal *cs, doublereal *sn)
{
    doublereal d__1, d__2;
    double d_lmp_sign(doublereal *, doublereal *), sqrt(doublereal);
    doublereal p, z__, aa, bb, cc, dd, cs1, sn1, sab, sac, eps, tau, temp, scale, bcmax, bcmis,
        sigma;
    extern doublereal dlapy2_(doublereal *, doublereal *), dlamch_(char *, ftnlen);
    eps = dlamch_((char *)"P", (ftnlen)1);
    if (*c__ == 0.) {
        *cs = 1.;
        *sn = 0.;
    } else if (*b == 0.) {
        *cs = 0.;
        *sn = 1.;
        temp = *d__;
        *d__ = *a;
        *a = temp;
        *b = -(*c__);
        *c__ = 0.;
    } else if (*a - *d__ == 0. && d_lmp_sign(&c_b3, b) != d_lmp_sign(&c_b3, c__)) {
        *cs = 1.;
        *sn = 0.;
    } else {
        temp = *a - *d__;
        p = temp * .5;
        d__1 = abs(*b), d__2 = abs(*c__);
        bcmax = max(d__1, d__2);
        d__1 = abs(*b), d__2 = abs(*c__);
        bcmis = min(d__1, d__2) * d_lmp_sign(&c_b3, b) * d_lmp_sign(&c_b3, c__);
        d__1 = abs(p);
        scale = max(d__1, bcmax);
        z__ = p / scale * p + bcmax / scale * bcmis;
        if (z__ >= eps * 4.) {
            d__1 = sqrt(scale) * sqrt(z__);
            z__ = p + d_lmp_sign(&d__1, &p);
            *a = *d__ + z__;
            *d__ -= bcmax / z__ * bcmis;
            tau = dlapy2_(c__, &z__);
            *cs = z__ / tau;
            *sn = *c__ / tau;
            *b -= *c__;
            *c__ = 0.;
        } else {
            sigma = *b + *c__;
            tau = dlapy2_(&sigma, &temp);
            *cs = sqrt((abs(sigma) / tau + 1.) * .5);
            *sn = -(p / (tau * *cs)) * d_lmp_sign(&c_b3, &sigma);
            aa = *a * *cs + *b * *sn;
            bb = -(*a) * *sn + *b * *cs;
            cc = *c__ * *cs + *d__ * *sn;
            dd = -(*c__) * *sn + *d__ * *cs;
            *a = aa * *cs + cc * *sn;
            *b = bb * *cs + dd * *sn;
            *c__ = -aa * *sn + cc * *cs;
            *d__ = -bb * *sn + dd * *cs;
            temp = (*a + *d__) * .5;
            *a = temp;
            *d__ = temp;
            if (*c__ != 0.) {
                if (*b != 0.) {
                    if (d_lmp_sign(&c_b3, b) == d_lmp_sign(&c_b3, c__)) {
                        sab = sqrt((abs(*b)));
                        sac = sqrt((abs(*c__)));
                        d__1 = sab * sac;
                        p = d_lmp_sign(&d__1, c__);
                        tau = 1. / sqrt((d__1 = *b + *c__, abs(d__1)));
                        *a = temp + p;
                        *d__ = temp - p;
                        *b -= *c__;
                        *c__ = 0.;
                        cs1 = sab * tau;
                        sn1 = sac * tau;
                        temp = *cs * cs1 - *sn * sn1;
                        *sn = *cs * sn1 + *sn * cs1;
                        *cs = temp;
                    }
                } else {
                    *b = -(*c__);
                    *c__ = 0.;
                    temp = *cs;
                    *cs = -(*sn);
                    *sn = temp;
                }
            }
        }
    }
    *rt1r = *a;
    *rt2r = *d__;
    if (*c__ == 0.) {
        *rt1i = 0.;
        *rt2i = 0.;
    } else {
        *rt1i = sqrt((abs(*b))) * sqrt((abs(*c__)));
        *rt2i = -(*rt1i);
    }
    return 0;
}
#ifdef __cplusplus
}
#endif
