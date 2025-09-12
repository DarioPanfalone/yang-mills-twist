#ifndef TUBE_MULTILEVEL_H
#define TUBE_MULTILEVEL_H

#include "macro.h"

#include<complex.h>
#include<openssl/md5.h>
#include<stdio.h>

#include"gauge_conf.h"
#include"gparam.h"
#include"geometry.h"
#include"su2.h"
#include"sun.h"
#include "tens_prod.h"

typedef struct TubeStuff {
    GAUGE_GROUP **local_poly; // [slab][rsp]
    double __complex__ *local_plaq;  // [rsp]

    TensProd **slab_polycorr; // [slab][rsp] in [slab_num][spave_vol]
    TensProd **slab_polyplaq; // [dist][rsp] in [transv_dist_max][space_vol]
} TubeStuff;

void init_TubeStuff(GParam const * const param,
                    Geometry const * const geo,
                    TubeStuff * tube_ptrs);

void free_TubeStuff(GParam const * const param,
                    Geometry const * const geo,
                    TubeStuff * tube_ptrs);

void perform_measures_onelevel_tube_vary(Gauge_Conf *GC,
                                Geometry const * const geo,
                                GParam const * const param,
                                TubeStuff * tube_ptrs,
                                FILE *datafilep);

#endif