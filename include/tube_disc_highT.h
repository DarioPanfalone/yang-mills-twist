#include <stdio.h>

#include"../include/gauge_conf.h"
#include"../include/geometry.h"
#include"../include/gparam.h"

typedef struct{
    double *aux_poly_r, *aux_poly_i;
    double *plaq_column_r, *plaq_column_i;
    double *three_pts_r, *three_pts_i;
    double *two_pts_r, *two_pts_i;
    double *plaq_corr_r, *plaq_corr_i;
    
    FILE* tubefilep;
    FILE* polyfilep;
    FILE* plaqfilep;
    FILE* padjfilep;
} Tube_disc_ptrs;

void perform_measures_tube_disc_highT(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      GParam const * const param,
                                      Tube_disc_ptrs const * const my_ptrs);

void perfor_measures_tube_disc_givenR(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      int poly_dist,
                                      Tube_disc_ptrs const * const my_ptrs);

void perform_measures_two_and_three_pts(long update_index,
                                        Geometry const * const geo,
                                        GParam const * const param,
                                        Tube_disc_ptrs * my_ptrs);

void perform_measures_two_pts(long update_index,
                              Geometry const * const geo,
                              GParam const * const param,
                              Tube_disc_ptrs * my_ptrs);

void perform_measures_plaqcor(long update_index,
                              Geometry const * const geo,
                              Tube_disc_ptrs const * const my_ptrs);

void perform_measures_two_pts_withAdj(long update_index,
                                      Geometry const *const geo,
                                      GParam const *const param,
                                      Tube_disc_ptrs *my_ptrs);

void compute_polyakov_tube_disc_highT(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      Tube_disc_ptrs * my_ptrs);

void compute_plaquette_columns(Gauge_Conf const * const GC,
                               Geometry const * const geo,
                               GParam const * const param,
                               Tube_disc_ptrs * my_ptrs);

void init_tube_disc_stuff(Geometry const * const geo,
                          GParam const * const param,
                          Tube_disc_ptrs * my_ptrs);

void init_with_adj(Geometry const * const geo,
                   GParam const * const param,
                   Tube_disc_ptrs *my_ptrs);

void init_tube_file(FILE **dataf, GParam const * const param);
void init_plaq_file(FILE **dataf, GParam const * const param);
void init_poly_file(FILE **dataf, GParam const * const param);
void init_padj_file(FILE **dataf, GParam const *const param);

void clean_tube_disc_stuff(Tube_disc_ptrs * my_ptrs);
void clean_with_adj(Tube_disc_ptrs *my_ptrs);