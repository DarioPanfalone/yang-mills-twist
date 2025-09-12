#ifndef BARION_H
#define BARION_H

#include"../include/gauge_conf.h"
#include"../include/geometry.h"
#include"../include/gparam.h"
#include"../include/tens_cube.h"

typedef struct Barion_Aux{
    double *poly_re, *poly_im;

    double* barion_re, *barion_im; // for barion at multiple temperatures

    TensCube ***poly_slab; // for multilevel, poly_slab[level][slab][rsp]
}Barion_Aux;

void init_barion_aux(Barion_Aux * my_ptr,
                     Geometry const * const geo,
                     GParam const * const param);

void free_barion_aux(Barion_Aux * my_ptr);

void init_barion_aux_for_ml(Barion_Aux * my_ptr,
                            Geometry const * const geo,
                            GParam const * const param);

void free_barion_aux_for_ml(Barion_Aux * my_ptr,
                            Geometry const * const geo,
                            GParam const * const param);

void perform_measures_barion(Gauge_Conf const * const GC,
                             Geometry const * const geo,
                             GParam const * const param,
                             FILE * datafilep,
                             Barion_Aux * my_ptr);

void perform_measures_barion_multihit(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      GParam const * const param,
                                      FILE * datafilep,
                                      Barion_Aux * my_ptr);

void perform_measures_barion_onelevel(Gauge_Conf * GC,
                                      Geometry const * const geo,
                                      GParam const * const param,
                                      FILE * datafilep,
                                      Barion_Aux * my_ptr);

void perform_measures_barion_onelevel_multiT(Gauge_Conf * GC,
                                             Geometry const * const geo,
                                             GParam const * const param,
                                             FILE * datafilep,
                                             Barion_Aux * my_ptr);

void print_parameters_barion(GParam const * const param,
                             time_t start,
                             time_t end);

#endif