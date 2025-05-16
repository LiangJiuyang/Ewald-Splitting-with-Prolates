#ifdef __cplusplus
extern "C" {
#endif
#include "lmp_f2c.h"
static doublecomplex c_b1 = {1., 0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static doublereal c_b14 = -1.;
static doublereal c_b15 = 1.;
int zpotrf_(char *uplo, integer *n, doublecomplex *a, integer *lda, integer *info, ftnlen uplo_len)
{
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;
    integer j, jb, nb;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    extern int zgemm_(char *, char *, integer *, integer *, integer *, doublecomplex *,
                      doublecomplex *, integer *, doublecomplex *, integer *, doublecomplex *,
                      doublecomplex *, integer *, ftnlen, ftnlen),
        zherk_(char *, char *, integer *, integer *, doublereal *, doublecomplex *, integer *,
               doublereal *, doublecomplex *, integer *, ftnlen, ftnlen);
    logical upper;
    extern int ztrsm_(char *, char *, char *, char *, integer *, integer *, doublecomplex *,
                      doublecomplex *, integer *, doublecomplex *, integer *, ftnlen, ftnlen,
                      ftnlen, ftnlen),
        xerbla_(char *, integer *, ftnlen);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, integer *, integer *,
                           ftnlen, ftnlen);
    extern int zpotrf2_(char *, integer *, doublecomplex *, integer *, integer *, ftnlen);
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    *info = 0;
    upper = lsame_(uplo, (char *)"U", (ftnlen)1, (ftnlen)1);
    if (!upper && !lsame_(uplo, (char *)"L", (ftnlen)1, (ftnlen)1)) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*lda < max(1, *n)) {
        *info = -4;
    }
    if (*info != 0) {
        i__1 = -(*info);
        xerbla_((char *)"ZPOTRF", &i__1, (ftnlen)6);
        return 0;
    }
    if (*n == 0) {
        return 0;
    }
    nb = ilaenv_(&c__1, (char *)"ZPOTRF", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
    if (nb <= 1 || nb >= *n) {
        zpotrf2_(uplo, n, &a[a_offset], lda, info, (ftnlen)1);
    } else {
        if (upper) {
            i__1 = *n;
            i__2 = nb;
            for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
                i__3 = nb, i__4 = *n - j + 1;
                jb = min(i__3, i__4);
                i__3 = j - 1;
                zherk_((char *)"Upper", (char *)"Conjugate transpose", &jb, &i__3, &c_b14, &a[j * a_dim1 + 1], lda,
                       &c_b15, &a[j + j * a_dim1], lda, (ftnlen)5, (ftnlen)19);
                zpotrf2_((char *)"Upper", &jb, &a[j + j * a_dim1], lda, info, (ftnlen)5);
                if (*info != 0) {
                    goto L30;
                }
                if (j + jb <= *n) {
                    i__3 = *n - j - jb + 1;
                    i__4 = j - 1;
                    z__1.r = -1., z__1.i = -0.;
                    zgemm_((char *)"Conjugate transpose", (char *)"No transpose", &jb, &i__3, &i__4, &z__1,
                           &a[j * a_dim1 + 1], lda, &a[(j + jb) * a_dim1 + 1], lda, &c_b1,
                           &a[j + (j + jb) * a_dim1], lda, (ftnlen)19, (ftnlen)12);
                    i__3 = *n - j - jb + 1;
                    ztrsm_((char *)"Left", (char *)"Upper", (char *)"Conjugate transpose", (char *)"Non-unit", &jb, &i__3, &c_b1,
                           &a[j + j * a_dim1], lda, &a[j + (j + jb) * a_dim1], lda, (ftnlen)4,
                           (ftnlen)5, (ftnlen)19, (ftnlen)8);
                }
            }
        } else {
            i__2 = *n;
            i__1 = nb;
            for (j = 1; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
                i__3 = nb, i__4 = *n - j + 1;
                jb = min(i__3, i__4);
                i__3 = j - 1;
                zherk_((char *)"Lower", (char *)"No transpose", &jb, &i__3, &c_b14, &a[j + a_dim1], lda, &c_b15,
                       &a[j + j * a_dim1], lda, (ftnlen)5, (ftnlen)12);
                zpotrf2_((char *)"Lower", &jb, &a[j + j * a_dim1], lda, info, (ftnlen)5);
                if (*info != 0) {
                    goto L30;
                }
                if (j + jb <= *n) {
                    i__3 = *n - j - jb + 1;
                    i__4 = j - 1;
                    z__1.r = -1., z__1.i = -0.;
                    zgemm_((char *)"No transpose", (char *)"Conjugate transpose", &i__3, &jb, &i__4, &z__1,
                           &a[j + jb + a_dim1], lda, &a[j + a_dim1], lda, &c_b1,
                           &a[j + jb + j * a_dim1], lda, (ftnlen)12, (ftnlen)19);
                    i__3 = *n - j - jb + 1;
                    ztrsm_((char *)"Right", (char *)"Lower", (char *)"Conjugate transpose", (char *)"Non-unit", &i__3, &jb, &c_b1,
                           &a[j + j * a_dim1], lda, &a[j + jb + j * a_dim1], lda, (ftnlen)5,
                           (ftnlen)5, (ftnlen)19, (ftnlen)8);
                }
            }
        }
    }
    goto L40;
L30:
    *info = *info + j - 1;
L40:
    return 0;
}
#ifdef __cplusplus
}
#endif
