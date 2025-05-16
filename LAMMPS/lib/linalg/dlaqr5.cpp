#ifdef __cplusplus
extern "C" {
#endif
#include "lmp_f2c.h"
static doublereal c_b7 = 0.;
static doublereal c_b8 = 1.;
static integer c__3 = 3;
static integer c__1 = 1;
static integer c__2 = 2;
int dlaqr5_(logical *wantt, logical *wantz, integer *kacc22, integer *n, integer *ktop,
            integer *kbot, integer *nshfts, doublereal *sr, doublereal *si, doublereal *h__,
            integer *ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz,
            doublereal *v, integer *ldv, doublereal *u, integer *ldu, integer *nv, doublereal *wv,
            integer *ldwv, integer *nh, doublereal *wh, integer *ldwh)
{
    integer h_dim1, h_offset, u_dim1, u_offset, v_dim1, v_offset, wh_dim1, wh_offset, wv_dim1,
        wv_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7;
    doublereal d__1, d__2, d__3, d__4, d__5;
    integer i__, j, k, m, i2, j2, i4, j4, k1;
    doublereal h11, h12, h21, h22;
    integer m22, ns, nu;
    doublereal vt[3], scl;
    integer kdu, kms;
    doublereal ulp;
    integer knz, kzs;
    doublereal tst1, tst2, beta;
    logical blk22, bmp22;
    integer mend, jcol, jlen, jbot, mbot;
    doublereal swap;
    integer jtop, jrow, mtop;
    doublereal alpha;
    logical accum;
    extern int dgemm_(char *, char *, integer *, integer *, integer *, doublereal *, doublereal *,
                      integer *, doublereal *, integer *, doublereal *, doublereal *, integer *,
                      ftnlen, ftnlen);
    integer ndcol, incol, krcol, nbmps;
    extern int dtrmm_(char *, char *, char *, char *, integer *, integer *, doublereal *,
                      doublereal *, integer *, doublereal *, integer *, ftnlen, ftnlen, ftnlen,
                      ftnlen),
        dlaqr1_(integer *, doublereal *, integer *, doublereal *, doublereal *, doublereal *,
                doublereal *, doublereal *),
        dlabad_(doublereal *, doublereal *);
    extern doublereal dlamch_(char *, ftnlen);
    extern int dlarfg_(integer *, doublereal *, doublereal *, integer *, doublereal *),
        dlacpy_(char *, integer *, integer *, doublereal *, integer *, doublereal *, integer *,
                ftnlen);
    doublereal safmin;
    extern int dlaset_(char *, integer *, integer *, doublereal *, doublereal *, doublereal *,
                       integer *, ftnlen);
    doublereal safmax, refsum;
    integer mstart;
    doublereal smlnum;
    --sr;
    --si;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    wh_dim1 = *ldwh;
    wh_offset = 1 + wh_dim1;
    wh -= wh_offset;
    if (*nshfts < 2) {
        return 0;
    }
    if (*ktop >= *kbot) {
        return 0;
    }
    i__1 = *nshfts - 2;
    for (i__ = 1; i__ <= i__1; i__ += 2) {
        if (si[i__] != -si[i__ + 1]) {
            swap = sr[i__];
            sr[i__] = sr[i__ + 1];
            sr[i__ + 1] = sr[i__ + 2];
            sr[i__ + 2] = swap;
            swap = si[i__];
            si[i__] = si[i__ + 1];
            si[i__ + 1] = si[i__ + 2];
            si[i__ + 2] = swap;
        }
    }
    ns = *nshfts - *nshfts % 2;
    safmin = dlamch_((char *)"SAFE MINIMUM", (ftnlen)12);
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_((char *)"PRECISION", (ftnlen)9);
    smlnum = safmin * ((doublereal)(*n) / ulp);
    accum = *kacc22 == 1 || *kacc22 == 2;
    blk22 = ns > 2 && *kacc22 == 2;
    if (*ktop + 2 <= *kbot) {
        h__[*ktop + 2 + *ktop * h_dim1] = 0.;
    }
    nbmps = ns / 2;
    kdu = nbmps * 6 - 3;
    i__1 = *kbot - 2;
    i__2 = nbmps * 3 - 2;
    for (incol = (1 - nbmps) * 3 + *ktop - 1; i__2 < 0 ? incol >= i__1 : incol <= i__1;
         incol += i__2) {
        ndcol = incol + kdu;
        if (accum) {
            dlaset_((char *)"ALL", &kdu, &kdu, &c_b7, &c_b8, &u[u_offset], ldu, (ftnlen)3);
        }
        i__4 = incol + nbmps * 3 - 3, i__5 = *kbot - 2;
        i__3 = min(i__4, i__5);
        for (krcol = incol; krcol <= i__3; ++krcol) {
            i__4 = 1, i__5 = (*ktop - 1 - krcol + 2) / 3 + 1;
            mtop = max(i__4, i__5);
            i__4 = nbmps, i__5 = (*kbot - krcol) / 3;
            mbot = min(i__4, i__5);
            m22 = mbot + 1;
            bmp22 = mbot < nbmps && krcol + (m22 - 1) * 3 == *kbot - 2;
            i__4 = mbot;
            for (m = mtop; m <= i__4; ++m) {
                k = krcol + (m - 1) * 3;
                if (k == *ktop - 1) {
                    dlaqr1_(&c__3, &h__[*ktop + *ktop * h_dim1], ldh, &sr[(m << 1) - 1],
                            &si[(m << 1) - 1], &sr[m * 2], &si[m * 2], &v[m * v_dim1 + 1]);
                    alpha = v[m * v_dim1 + 1];
                    dlarfg_(&c__3, &alpha, &v[m * v_dim1 + 2], &c__1, &v[m * v_dim1 + 1]);
                } else {
                    beta = h__[k + 1 + k * h_dim1];
                    v[m * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
                    v[m * v_dim1 + 3] = h__[k + 3 + k * h_dim1];
                    dlarfg_(&c__3, &beta, &v[m * v_dim1 + 2], &c__1, &v[m * v_dim1 + 1]);
                    if (h__[k + 3 + k * h_dim1] != 0. || h__[k + 3 + (k + 1) * h_dim1] != 0. ||
                        h__[k + 3 + (k + 2) * h_dim1] == 0.) {
                        h__[k + 1 + k * h_dim1] = beta;
                        h__[k + 2 + k * h_dim1] = 0.;
                        h__[k + 3 + k * h_dim1] = 0.;
                    } else {
                        dlaqr1_(&c__3, &h__[k + 1 + (k + 1) * h_dim1], ldh, &sr[(m << 1) - 1],
                                &si[(m << 1) - 1], &sr[m * 2], &si[m * 2], vt);
                        alpha = vt[0];
                        dlarfg_(&c__3, &alpha, &vt[1], &c__1, vt);
                        refsum =
                            vt[0] * (h__[k + 1 + k * h_dim1] + vt[1] * h__[k + 2 + k * h_dim1]);
                        if ((d__1 = h__[k + 2 + k * h_dim1] - refsum * vt[1], abs(d__1)) +
                                (d__2 = refsum * vt[2], abs(d__2)) >
                            ulp * ((d__3 = h__[k + k * h_dim1], abs(d__3)) +
                                   (d__4 = h__[k + 1 + (k + 1) * h_dim1], abs(d__4)) +
                                   (d__5 = h__[k + 2 + (k + 2) * h_dim1], abs(d__5)))) {
                            h__[k + 1 + k * h_dim1] = beta;
                            h__[k + 2 + k * h_dim1] = 0.;
                            h__[k + 3 + k * h_dim1] = 0.;
                        } else {
                            h__[k + 1 + k * h_dim1] -= refsum;
                            h__[k + 2 + k * h_dim1] = 0.;
                            h__[k + 3 + k * h_dim1] = 0.;
                            v[m * v_dim1 + 1] = vt[0];
                            v[m * v_dim1 + 2] = vt[1];
                            v[m * v_dim1 + 3] = vt[2];
                        }
                    }
                }
            }
            k = krcol + (m22 - 1) * 3;
            if (bmp22) {
                if (k == *ktop - 1) {
                    dlaqr1_(&c__2, &h__[k + 1 + (k + 1) * h_dim1], ldh, &sr[(m22 << 1) - 1],
                            &si[(m22 << 1) - 1], &sr[m22 * 2], &si[m22 * 2], &v[m22 * v_dim1 + 1]);
                    beta = v[m22 * v_dim1 + 1];
                    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 * v_dim1 + 1]);
                } else {
                    beta = h__[k + 1 + k * h_dim1];
                    v[m22 * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
                    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 * v_dim1 + 1]);
                    h__[k + 1 + k * h_dim1] = beta;
                    h__[k + 2 + k * h_dim1] = 0.;
                }
            }
            if (accum) {
                jbot = min(ndcol, *kbot);
            } else if (*wantt) {
                jbot = *n;
            } else {
                jbot = *kbot;
            }
            i__4 = jbot;
            for (j = max(*ktop, krcol); j <= i__4; ++j) {
                i__5 = mbot, i__6 = (j - krcol + 2) / 3;
                mend = min(i__5, i__6);
                i__5 = mend;
                for (m = mtop; m <= i__5; ++m) {
                    k = krcol + (m - 1) * 3;
                    refsum = v[m * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] +
                                                  v[m * v_dim1 + 2] * h__[k + 2 + j * h_dim1] +
                                                  v[m * v_dim1 + 3] * h__[k + 3 + j * h_dim1]);
                    h__[k + 1 + j * h_dim1] -= refsum;
                    h__[k + 2 + j * h_dim1] -= refsum * v[m * v_dim1 + 2];
                    h__[k + 3 + j * h_dim1] -= refsum * v[m * v_dim1 + 3];
                }
            }
            if (bmp22) {
                k = krcol + (m22 - 1) * 3;
                i__4 = k + 1;
                i__5 = jbot;
                for (j = max(i__4, *ktop); j <= i__5; ++j) {
                    refsum = v[m22 * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] +
                                                    v[m22 * v_dim1 + 2] * h__[k + 2 + j * h_dim1]);
                    h__[k + 1 + j * h_dim1] -= refsum;
                    h__[k + 2 + j * h_dim1] -= refsum * v[m22 * v_dim1 + 2];
                }
            }
            if (accum) {
                jtop = max(*ktop, incol);
            } else if (*wantt) {
                jtop = 1;
            } else {
                jtop = *ktop;
            }
            i__5 = mbot;
            for (m = mtop; m <= i__5; ++m) {
                if (v[m * v_dim1 + 1] != 0.) {
                    k = krcol + (m - 1) * 3;
                    i__6 = *kbot, i__7 = k + 3;
                    i__4 = min(i__6, i__7);
                    for (j = jtop; j <= i__4; ++j) {
                        refsum =
                            v[m * v_dim1 + 1] * (h__[j + (k + 1) * h_dim1] +
                                                 v[m * v_dim1 + 2] * h__[j + (k + 2) * h_dim1] +
                                                 v[m * v_dim1 + 3] * h__[j + (k + 3) * h_dim1]);
                        h__[j + (k + 1) * h_dim1] -= refsum;
                        h__[j + (k + 2) * h_dim1] -= refsum * v[m * v_dim1 + 2];
                        h__[j + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 + 3];
                    }
                    if (accum) {
                        kms = k - incol;
                        i__4 = 1, i__6 = *ktop - incol;
                        i__7 = kdu;
                        for (j = max(i__4, i__6); j <= i__7; ++j) {
                            refsum =
                                v[m * v_dim1 + 1] * (u[j + (kms + 1) * u_dim1] +
                                                     v[m * v_dim1 + 2] * u[j + (kms + 2) * u_dim1] +
                                                     v[m * v_dim1 + 3] * u[j + (kms + 3) * u_dim1]);
                            u[j + (kms + 1) * u_dim1] -= refsum;
                            u[j + (kms + 2) * u_dim1] -= refsum * v[m * v_dim1 + 2];
                            u[j + (kms + 3) * u_dim1] -= refsum * v[m * v_dim1 + 3];
                        }
                    } else if (*wantz) {
                        i__7 = *ihiz;
                        for (j = *iloz; j <= i__7; ++j) {
                            refsum =
                                v[m * v_dim1 + 1] * (z__[j + (k + 1) * z_dim1] +
                                                     v[m * v_dim1 + 2] * z__[j + (k + 2) * z_dim1] +
                                                     v[m * v_dim1 + 3] * z__[j + (k + 3) * z_dim1]);
                            z__[j + (k + 1) * z_dim1] -= refsum;
                            z__[j + (k + 2) * z_dim1] -= refsum * v[m * v_dim1 + 2];
                            z__[j + (k + 3) * z_dim1] -= refsum * v[m * v_dim1 + 3];
                        }
                    }
                }
            }
            k = krcol + (m22 - 1) * 3;
            if (bmp22) {
                if (v[m22 * v_dim1 + 1] != 0.) {
                    i__7 = *kbot, i__4 = k + 3;
                    i__5 = min(i__7, i__4);
                    for (j = jtop; j <= i__5; ++j) {
                        refsum =
                            v[m22 * v_dim1 + 1] * (h__[j + (k + 1) * h_dim1] +
                                                   v[m22 * v_dim1 + 2] * h__[j + (k + 2) * h_dim1]);
                        h__[j + (k + 1) * h_dim1] -= refsum;
                        h__[j + (k + 2) * h_dim1] -= refsum * v[m22 * v_dim1 + 2];
                    }
                    if (accum) {
                        kms = k - incol;
                        i__5 = 1, i__7 = *ktop - incol;
                        i__4 = kdu;
                        for (j = max(i__5, i__7); j <= i__4; ++j) {
                            refsum = v[m22 * v_dim1 + 1] *
                                     (u[j + (kms + 1) * u_dim1] +
                                      v[m22 * v_dim1 + 2] * u[j + (kms + 2) * u_dim1]);
                            u[j + (kms + 1) * u_dim1] -= refsum;
                            u[j + (kms + 2) * u_dim1] -= refsum * v[m22 * v_dim1 + 2];
                        }
                    } else if (*wantz) {
                        i__4 = *ihiz;
                        for (j = *iloz; j <= i__4; ++j) {
                            refsum = v[m22 * v_dim1 + 1] *
                                     (z__[j + (k + 1) * z_dim1] +
                                      v[m22 * v_dim1 + 2] * z__[j + (k + 2) * z_dim1]);
                            z__[j + (k + 1) * z_dim1] -= refsum;
                            z__[j + (k + 2) * z_dim1] -= refsum * v[m22 * v_dim1 + 2];
                        }
                    }
                }
            }
            mstart = mtop;
            if (krcol + (mstart - 1) * 3 < *ktop) {
                ++mstart;
            }
            mend = mbot;
            if (bmp22) {
                ++mend;
            }
            if (krcol == *kbot - 2) {
                ++mend;
            }
            i__4 = mend;
            for (m = mstart; m <= i__4; ++m) {
                i__5 = *kbot - 1, i__7 = krcol + (m - 1) * 3;
                k = min(i__5, i__7);
                if (h__[k + 1 + k * h_dim1] != 0.) {
                    tst1 = (d__1 = h__[k + k * h_dim1], abs(d__1)) +
                           (d__2 = h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
                    if (tst1 == 0.) {
                        if (k >= *ktop + 1) {
                            tst1 += (d__1 = h__[k + (k - 1) * h_dim1], abs(d__1));
                        }
                        if (k >= *ktop + 2) {
                            tst1 += (d__1 = h__[k + (k - 2) * h_dim1], abs(d__1));
                        }
                        if (k >= *ktop + 3) {
                            tst1 += (d__1 = h__[k + (k - 3) * h_dim1], abs(d__1));
                        }
                        if (k <= *kbot - 2) {
                            tst1 += (d__1 = h__[k + 2 + (k + 1) * h_dim1], abs(d__1));
                        }
                        if (k <= *kbot - 3) {
                            tst1 += (d__1 = h__[k + 3 + (k + 1) * h_dim1], abs(d__1));
                        }
                        if (k <= *kbot - 4) {
                            tst1 += (d__1 = h__[k + 4 + (k + 1) * h_dim1], abs(d__1));
                        }
                    }
                    d__2 = smlnum, d__3 = ulp * tst1;
                    if ((d__1 = h__[k + 1 + k * h_dim1], abs(d__1)) <= max(d__2, d__3)) {
                        d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)),
                        d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(d__2));
                        h12 = max(d__3, d__4);
                        d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)),
                        d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(d__2));
                        h21 = min(d__3, d__4);
                        d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(d__1)),
                        d__4 =
                            (d__2 = h__[k + k * h_dim1] - h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
                        h11 = max(d__3, d__4);
                        d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(d__1)),
                        d__4 =
                            (d__2 = h__[k + k * h_dim1] - h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
                        h22 = min(d__3, d__4);
                        scl = h11 + h12;
                        tst2 = h22 * (h11 / scl);
                        d__1 = smlnum, d__2 = ulp * tst2;
                        if (tst2 == 0. || h21 * (h12 / scl) <= max(d__1, d__2)) {
                            h__[k + 1 + k * h_dim1] = 0.;
                        }
                    }
                }
            }
            i__4 = nbmps, i__5 = (*kbot - krcol - 1) / 3;
            mend = min(i__4, i__5);
            i__4 = mend;
            for (m = mtop; m <= i__4; ++m) {
                k = krcol + (m - 1) * 3;
                refsum = v[m * v_dim1 + 1] * v[m * v_dim1 + 3] * h__[k + 4 + (k + 3) * h_dim1];
                h__[k + 4 + (k + 1) * h_dim1] = -refsum;
                h__[k + 4 + (k + 2) * h_dim1] = -refsum * v[m * v_dim1 + 2];
                h__[k + 4 + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 + 3];
            }
        }
        if (accum) {
            if (*wantt) {
                jtop = 1;
                jbot = *n;
            } else {
                jtop = *ktop;
                jbot = *kbot;
            }
            if (!blk22 || incol < *ktop || ndcol > *kbot || ns <= 2) {
                i__3 = 1, i__4 = *ktop - incol;
                k1 = max(i__3, i__4);
                i__3 = 0, i__4 = ndcol - *kbot;
                nu = kdu - max(i__3, i__4) - k1 + 1;
                i__3 = jbot;
                i__4 = *nh;
                for (jcol = min(ndcol, *kbot) + 1; i__4 < 0 ? jcol >= i__3 : jcol <= i__3;
                     jcol += i__4) {
                    i__5 = *nh, i__7 = jbot - jcol + 1;
                    jlen = min(i__5, i__7);
                    dgemm_((char *)"C", (char *)"N", &nu, &jlen, &nu, &c_b8, &u[k1 + k1 * u_dim1], ldu,
                           &h__[incol + k1 + jcol * h_dim1], ldh, &c_b7, &wh[wh_offset], ldwh,
                           (ftnlen)1, (ftnlen)1);
                    dlacpy_((char *)"ALL", &nu, &jlen, &wh[wh_offset], ldwh,
                            &h__[incol + k1 + jcol * h_dim1], ldh, (ftnlen)3);
                }
                i__4 = max(*ktop, incol) - 1;
                i__3 = *nv;
                for (jrow = jtop; i__3 < 0 ? jrow >= i__4 : jrow <= i__4; jrow += i__3) {
                    i__5 = *nv, i__7 = max(*ktop, incol) - jrow;
                    jlen = min(i__5, i__7);
                    dgemm_((char *)"N", (char *)"N", &jlen, &nu, &nu, &c_b8, &h__[jrow + (incol + k1) * h_dim1],
                           ldh, &u[k1 + k1 * u_dim1], ldu, &c_b7, &wv[wv_offset], ldwv, (ftnlen)1,
                           (ftnlen)1);
                    dlacpy_((char *)"ALL", &jlen, &nu, &wv[wv_offset], ldwv,
                            &h__[jrow + (incol + k1) * h_dim1], ldh, (ftnlen)3);
                }
                if (*wantz) {
                    i__3 = *ihiz;
                    i__4 = *nv;
                    for (jrow = *iloz; i__4 < 0 ? jrow >= i__3 : jrow <= i__3; jrow += i__4) {
                        i__5 = *nv, i__7 = *ihiz - jrow + 1;
                        jlen = min(i__5, i__7);
                        dgemm_((char *)"N", (char *)"N", &jlen, &nu, &nu, &c_b8, &z__[jrow + (incol + k1) * z_dim1],
                               ldz, &u[k1 + k1 * u_dim1], ldu, &c_b7, &wv[wv_offset], ldwv,
                               (ftnlen)1, (ftnlen)1);
                        dlacpy_((char *)"ALL", &jlen, &nu, &wv[wv_offset], ldwv,
                                &z__[jrow + (incol + k1) * z_dim1], ldz, (ftnlen)3);
                    }
                }
            } else {
                i2 = (kdu + 1) / 2;
                i4 = kdu;
                j2 = i4 - i2;
                j4 = kdu;
                kzs = j4 - j2 - (ns + 1);
                knz = ns + 1;
                i__4 = jbot;
                i__3 = *nh;
                for (jcol = min(ndcol, *kbot) + 1; i__3 < 0 ? jcol >= i__4 : jcol <= i__4;
                     jcol += i__3) {
                    i__5 = *nh, i__7 = jbot - jcol + 1;
                    jlen = min(i__5, i__7);
                    dlacpy_((char *)"ALL", &knz, &jlen, &h__[incol + 1 + j2 + jcol * h_dim1], ldh,
                            &wh[kzs + 1 + wh_dim1], ldwh, (ftnlen)3);
                    dlaset_((char *)"ALL", &kzs, &jlen, &c_b7, &c_b7, &wh[wh_offset], ldwh, (ftnlen)3);
                    dtrmm_((char *)"L", (char *)"U", (char *)"C", (char *)"N", &knz, &jlen, &c_b8, &u[j2 + 1 + (kzs + 1) * u_dim1],
                           ldu, &wh[kzs + 1 + wh_dim1], ldwh, (ftnlen)1, (ftnlen)1, (ftnlen)1,
                           (ftnlen)1);
                    dgemm_((char *)"C", (char *)"N", &i2, &jlen, &j2, &c_b8, &u[u_offset], ldu,
                           &h__[incol + 1 + jcol * h_dim1], ldh, &c_b8, &wh[wh_offset], ldwh,
                           (ftnlen)1, (ftnlen)1);
                    dlacpy_((char *)"ALL", &j2, &jlen, &h__[incol + 1 + jcol * h_dim1], ldh,
                            &wh[i2 + 1 + wh_dim1], ldwh, (ftnlen)3);
                    dtrmm_((char *)"L", (char *)"L", (char *)"C", (char *)"N", &j2, &jlen, &c_b8, &u[(i2 + 1) * u_dim1 + 1], ldu,
                           &wh[i2 + 1 + wh_dim1], ldwh, (ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
                    i__5 = i4 - i2;
                    i__7 = j4 - j2;
                    dgemm_((char *)"C", (char *)"N", &i__5, &jlen, &i__7, &c_b8, &u[j2 + 1 + (i2 + 1) * u_dim1],
                           ldu, &h__[incol + 1 + j2 + jcol * h_dim1], ldh, &c_b8,
                           &wh[i2 + 1 + wh_dim1], ldwh, (ftnlen)1, (ftnlen)1);
                    dlacpy_((char *)"ALL", &kdu, &jlen, &wh[wh_offset], ldwh,
                            &h__[incol + 1 + jcol * h_dim1], ldh, (ftnlen)3);
                }
                i__3 = max(incol, *ktop) - 1;
                i__4 = *nv;
                for (jrow = jtop; i__4 < 0 ? jrow >= i__3 : jrow <= i__3; jrow += i__4) {
                    i__5 = *nv, i__7 = max(incol, *ktop) - jrow;
                    jlen = min(i__5, i__7);
                    dlacpy_((char *)"ALL", &jlen, &knz, &h__[jrow + (incol + 1 + j2) * h_dim1], ldh,
                            &wv[(kzs + 1) * wv_dim1 + 1], ldwv, (ftnlen)3);
                    dlaset_((char *)"ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[wv_offset], ldwv, (ftnlen)3);
                    dtrmm_((char *)"R", (char *)"U", (char *)"N", (char *)"N", &jlen, &knz, &c_b8, &u[j2 + 1 + (kzs + 1) * u_dim1],
                           ldu, &wv[(kzs + 1) * wv_dim1 + 1], ldwv, (ftnlen)1, (ftnlen)1, (ftnlen)1,
                           (ftnlen)1);
                    dgemm_((char *)"N", (char *)"N", &jlen, &i2, &j2, &c_b8, &h__[jrow + (incol + 1) * h_dim1], ldh,
                           &u[u_offset], ldu, &c_b8, &wv[wv_offset], ldwv, (ftnlen)1, (ftnlen)1);
                    dlacpy_((char *)"ALL", &jlen, &j2, &h__[jrow + (incol + 1) * h_dim1], ldh,
                            &wv[(i2 + 1) * wv_dim1 + 1], ldwv, (ftnlen)3);
                    i__5 = i4 - i2;
                    dtrmm_((char *)"R", (char *)"L", (char *)"N", (char *)"N", &jlen, &i__5, &c_b8, &u[(i2 + 1) * u_dim1 + 1], ldu,
                           &wv[(i2 + 1) * wv_dim1 + 1], ldwv, (ftnlen)1, (ftnlen)1, (ftnlen)1,
                           (ftnlen)1);
                    i__5 = i4 - i2;
                    i__7 = j4 - j2;
                    dgemm_((char *)"N", (char *)"N", &jlen, &i__5, &i__7, &c_b8,
                           &h__[jrow + (incol + 1 + j2) * h_dim1], ldh,
                           &u[j2 + 1 + (i2 + 1) * u_dim1], ldu, &c_b8, &wv[(i2 + 1) * wv_dim1 + 1],
                           ldwv, (ftnlen)1, (ftnlen)1);
                    dlacpy_((char *)"ALL", &jlen, &kdu, &wv[wv_offset], ldwv,
                            &h__[jrow + (incol + 1) * h_dim1], ldh, (ftnlen)3);
                }
                if (*wantz) {
                    i__4 = *ihiz;
                    i__3 = *nv;
                    for (jrow = *iloz; i__3 < 0 ? jrow >= i__4 : jrow <= i__4; jrow += i__3) {
                        i__5 = *nv, i__7 = *ihiz - jrow + 1;
                        jlen = min(i__5, i__7);
                        dlacpy_((char *)"ALL", &jlen, &knz, &z__[jrow + (incol + 1 + j2) * z_dim1], ldz,
                                &wv[(kzs + 1) * wv_dim1 + 1], ldwv, (ftnlen)3);
                        dlaset_((char *)"ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[wv_offset], ldwv, (ftnlen)3);
                        dtrmm_((char *)"R", (char *)"U", (char *)"N", (char *)"N", &jlen, &knz, &c_b8,
                               &u[j2 + 1 + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) * wv_dim1 + 1],
                               ldwv, (ftnlen)1, (ftnlen)1, (ftnlen)1, (ftnlen)1);
                        dgemm_((char *)"N", (char *)"N", &jlen, &i2, &j2, &c_b8, &z__[jrow + (incol + 1) * z_dim1],
                               ldz, &u[u_offset], ldu, &c_b8, &wv[wv_offset], ldwv, (ftnlen)1,
                               (ftnlen)1);
                        dlacpy_((char *)"ALL", &jlen, &j2, &z__[jrow + (incol + 1) * z_dim1], ldz,
                                &wv[(i2 + 1) * wv_dim1 + 1], ldwv, (ftnlen)3);
                        i__5 = i4 - i2;
                        dtrmm_((char *)"R", (char *)"L", (char *)"N", (char *)"N", &jlen, &i__5, &c_b8, &u[(i2 + 1) * u_dim1 + 1],
                               ldu, &wv[(i2 + 1) * wv_dim1 + 1], ldwv, (ftnlen)1, (ftnlen)1,
                               (ftnlen)1, (ftnlen)1);
                        i__5 = i4 - i2;
                        i__7 = j4 - j2;
                        dgemm_((char *)"N", (char *)"N", &jlen, &i__5, &i__7, &c_b8,
                               &z__[jrow + (incol + 1 + j2) * z_dim1], ldz,
                               &u[j2 + 1 + (i2 + 1) * u_dim1], ldu, &c_b8,
                               &wv[(i2 + 1) * wv_dim1 + 1], ldwv, (ftnlen)1, (ftnlen)1);
                        dlacpy_((char *)"ALL", &jlen, &kdu, &wv[wv_offset], ldwv,
                                &z__[jrow + (incol + 1) * z_dim1], ldz, (ftnlen)3);
                    }
                }
            }
        }
    }
    return 0;
}
#ifdef __cplusplus
}
#endif
