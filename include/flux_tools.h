#ifndef FLUX_TOOLS_INCLUDED
#define FLUX_TOOLS_INCLUDED

#include "macro.h"

#include<complex.h>
#include<openssl/md5.h>
#include<stdio.h>

#include"function_pointers.h"
#include"gauge_conf.h"
#include"gparam.h"
#include"geometry.h"
#include"su2.h"
#include"sun.h"

typedef struct sources_cfg{
    int positive;
    int negative;

    int *position[STDIM];
} sources_cfg;

sources_cfg* new_sources_cfg(int positive, int negative);
void free_sources_cfg(sources_cfg* ptr);

void conf_smear_2n(Gauge_Conf * GC,
                   Gauge_Conf * aux_GC,
                   Geometry const * const geo,
                   double smearing_rho,
                   int smeraring_step_half);

void fluxtube_sweep(Gauge_Conf const * const GC,
                   Geometry const * const geo,
                   GParam const * const param,
                   FILE* flux_filep);

void generic_fundsources_sweep(Gauge_Conf const * const GC,
                               Geometry const * const geo,
                               GParam const * const param,
                               sources_cfg const * const sources,
                               FILE* flux_filep);

#endif