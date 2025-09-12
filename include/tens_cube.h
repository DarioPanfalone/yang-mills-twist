#ifndef TENS_CUBE_H
#define TENS_CUBE_H

#include<complex.h>
#include<stdio.h>
#include<stdlib.h>

#include"macro.h"

typedef struct TensCube {
   double complex comp[NCOLOR][NCOLOR][NCOLOR][NCOLOR][NCOLOR][NCOLOR] __attribute__((aligned(DOUBLE_ALIGN)));
} TensCube;

// A = 0
inline void zero_TensCube(TensCube * restrict A) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                A->comp[i1][j1][i2][j2][i3][j3] = 0.0 + 0.0*I;
            }
            }
        }
        }
    }
    }
}

// A = 1
inline void one_TensCube(TensCube * restrict A) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                A->comp[i1][j1][i2][j2][i3][j3] = 0.0 + 0.0*I;
            }
            }
        }
        }
    }
    }

    for (int i1 = 0; i1 < NCOLOR; i1++) {
        for (int i2 = 0; i2 < NCOLOR; i2++) {
            for (int i3 = 0; i3 < NCOLOR; i3++) {
                A->comp[i1][i1][i2][i2][i3][i3] = 1.0 + 0.0*I;
            }
        }
    }
}

// A = B
inline void equal_TensCube(TensCube * restrict A, TensCube const * const restrict B) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                A->comp[i1][j1][i2][j2][i3][j3] = B->comp[i1][j1][i2][j2][i3][j3];
            }
            }
        }
        }
    }
    }
}

// A = r * A
inline void times_equal_real_TensCube(TensCube * restrict A, double r) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                A->comp[i1][j1][i2][j2][i3][j3] *= r; 
            }
            }
        }
        }
    }
    }
}

// A = A + B
inline void plus_equal_TensCube(TensCube * restrict A, TensCube const * const restrict B) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                A->comp[i1][j1][i2][j2][i3][j3] += B->comp[i1][j1][i2][j2][i3][j3];
            }
            }
        }
        }
    }
    }
}

// A = B * C
inline void times_TensCube(TensCube * restrict A, TensCube const * const restrict B, TensCube const *const restrict C) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                
                complex sum = 0. + 0.*I;

                for (int k1 = 0; k1 < NCOLOR; k1++) {
                    for (int k2 = 0; k2 < NCOLOR; k2++) {
                        for (int k3 = 0; k3 < NCOLOR; k3++) {
                            sum += B->comp[i1][k1][i2][k2][i3][k3] * C->comp[k1][j1][k2][j2][k3][j3];
                        }
                    }
                }
                A->comp[i1][j1][i2][j2][i3][j3] = sum;
            }
            }
        }
        }
    }
    }
}

// A = A * B
inline void times_equal_TensCube(TensCube * restrict A, TensCube const * const restrict B) {
    TensCube tmp __attribute__((aligned(DOUBLE_ALIGN)));

    equal_TensCube(&tmp, A);
    times_TensCube(A, &tmp, B);
}

// Re{tr[A]}
inline double retr_TensCube(TensCube const * const restrict A) {
    double retr = 0;

    for (int i1 = 0; i1 < NCOLOR; i1++) {
        for (int i2 = 0; i2 < NCOLOR; i2++) {
            for (int i3 = 0; i3 < NCOLOR; i3++) {
                retr += creal(A->comp[i1][i1][i2][i2][i3][i3]);
            }
        }
    }

    return retr / (NCOLOR * NCOLOR * NCOLOR);
}

// Im{tr[A]}
inline double imtr_TensCube(TensCube const * const restrict A) {
    double imtr = 0;

    for (int i1 = 0; i1 < NCOLOR; i1++) {
        for (int i2 = 0; i2 < NCOLOR; i2++) {
            for (int i3 = 0; i3 < NCOLOR; i3++) {
                imtr += cimag(A->comp[i1][i1][i2][i2][i3][i3]);
            }
        }
    }

    return imtr / (NCOLOR * NCOLOR * NCOLOR);
}

inline void TensCube_init(TensCube * TC, GAUGE_GROUP const * const A, GAUGE_GROUP const * const B, GAUGE_GROUP const * const C) {
    for (int i1 = 0; i1 < NCOLOR; i1++) {for (int j1 = 0; j1 < NCOLOR; j1++){
        for (int i2 = 0; i2 < NCOLOR; i2++) {for (int j2 = 0; j2 < NCOLOR; j2++){
            for (int i3 = 0; i3 < NCOLOR; i3++) {for (int j3 = 0; j3 < NCOLOR; j3++){
                TC->comp[i1][j1][i2][j2][i3][j3] = A->comp[m(i1, j1)] * B->comp[m(i2, j2)] * C->comp[m(i3, j3)];
            }
            }
        }
        }
    }
    }
}

#endif