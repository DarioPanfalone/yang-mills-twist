#include"../include/barion.h"

#include"../include/endianness.h"
#include"../include/function_pointers.h"
#include"../include/gauge_conf.h"
#include"../include/geometry.h"
#include"../include/gparam.h"
#include"../include/tens_cube.h"

void init_barion_aux(Barion_Aux * my_ptr,
                     Geometry const * const geo,
                     GParam const * const param) {
    int err = 0;
    err = posix_memalign((void**) &(my_ptr->poly_re), DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err == 0) {
        err = posix_memalign((void**) &(my_ptr->poly_im), DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    }
    if (err != 0) {
        fprintf(stderr, "Unable to allocate memeory for polyakov loops!");
        exit(EXIT_FAILURE);
    }

    err = param->d_coolrepeat;
}

void free_barion_aux(Barion_Aux * my_ptr){
    free(my_ptr->poly_re);
    free(my_ptr->poly_im);
}

void init_barion_aux_for_ml(Barion_Aux * my_ptr,
                            Geometry const * const geo,
                            GParam const * const param) {
    int err = 0;
    err = posix_memalign((void **) &(my_ptr->poly_slab), DOUBLE_ALIGN, (size_t) NLEVELS * sizeof(TensCube**));
    for (int i = 0; i < NLEVELS; i++) {
        int slab_num = geo->d_size[0] / param->d_ml_step[i];
        if (err == 0) err = posix_memalign((void**) &(my_ptr->poly_slab[i]), DOUBLE_ALIGN, (size_t) slab_num * sizeof(TensCube*));
        else break;
        for (int j = 0; j < slab_num; j++) {
            if (err == 0) err = posix_memalign((void **) &(my_ptr->poly_slab[i][j]), DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(TensCube));
            else break;
        }
    }

    int slab_num = geo->d_size[0] / param->d_ml_step[NLEVELS - 1];
    if (err == 0) err = posix_memalign((void**) &(my_ptr->poly_im), DOUBLE_ALIGN, (size_t) slab_num * sizeof(double));
    if (err == 0) err = posix_memalign((void**) &(my_ptr->poly_re), DOUBLE_ALIGN, (size_t) slab_num * sizeof(double));
    if (err == 0) err = posix_memalign((void**) &(my_ptr->barion_im), DOUBLE_ALIGN, (size_t) slab_num * sizeof(double));
    if (err == 0) err = posix_memalign((void**) &(my_ptr->barion_re), DOUBLE_ALIGN, (size_t) slab_num * sizeof(double));

    if (err != 0) {
        fprintf(stderr, "Unable to allocate memeory for tensor cubes!");
        exit(EXIT_FAILURE);
    }
}

void free_barion_aux_for_ml(Barion_Aux * my_ptr,
                            Geometry const * const geo,
                            GParam const * const param)
{
    for (int i = 0; i < NLEVELS; i++) {
        int slab_num = geo->d_size[0] / param->d_ml_step[i];
        for (int j = 0; j < slab_num; j++) free(my_ptr->poly_slab[i][j]);
        free(my_ptr->poly_slab[i]);
    }
    free(my_ptr->poly_slab);

    free(my_ptr->poly_re);
    free(my_ptr->poly_im);
    free(my_ptr->barion_re);
    free(my_ptr->barion_im);

}

void perform_measures_barion(Gauge_Conf const * const GC,
                             Geometry const * const geo,
                             GParam const * const param,
                             FILE * datafilep,
                             Barion_Aux * my_ptr){

    double mean_poly_re = 0, mean_poly_im = 0;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) reduction(+:mean_poly_re) reduction(+:mean_poly_im)
    #endif
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        GAUGE_GROUP mtrx;
        one(&mtrx);

        long r = sisp_and_t_to_si(geo, rsp, 0);
        for (int t = 0; t < geo->d_size[0]; t++) {
            times_equal(&mtrx, &(GC->lattice[r][0]));
            r = nnp(geo, r, 0);
        }

        my_ptr->poly_re[rsp] = retr(&mtrx);
        my_ptr->poly_im[rsp] = imtr(&mtrx);

        mean_poly_re += my_ptr->poly_re[rsp];
        mean_poly_im += my_ptr->poly_im[rsp];
    }
    mean_poly_re *= geo->d_inv_space_vol;
    mean_poly_im *= geo->d_inv_space_vol;

    double plaqs, plaqt;
    plaquette(GC, geo, &plaqs, &plaqt);

    double barion_re = 0, barion_im = 0;
    double poly_corr_re = 0, poly_corr_im = 0;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) reduction(+:barion_re) reduction(+:barion_im) reduction(+:poly_corr_re) reduction(+:poly_corr_im)
    #endif
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        long r = sisp_and_t_to_si(geo, rsp, 0);
        long r_vertex = r;
        int half_way = param->d_triangle_base / 2;
        for (int i = 0; i <  half_way; i++) {
            r_vertex = nnp(geo, r_vertex, 1);
        }
        long r_other = r_vertex;
        for (int i = half_way; i < param->d_triangle_base; i++) {
            r_other = nnp(geo, r_other, 1);
        }
        for (int i = 0; i < param->d_triangle_hight; i++) {
            r_vertex = nnp(geo, r_vertex, 2);
        }

        long rsp_other, rsp_vertex;
        int t_tmp;
        si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r_other);
        si_to_sisp_and_t(&rsp_vertex, &t_tmp, geo, r_vertex);

        double base_re = my_ptr->poly_re[rsp] * my_ptr->poly_re[rsp_other] -
                         my_ptr->poly_im[rsp] * my_ptr->poly_im[rsp_other];
        double base_im = my_ptr->poly_im[rsp] * my_ptr->poly_re[rsp_other] +
                         my_ptr->poly_re[rsp] * my_ptr->poly_im[rsp_other];

        poly_corr_re += my_ptr->poly_re[rsp] * my_ptr->poly_re[rsp_other] +
                        my_ptr->poly_im[rsp] * my_ptr->poly_im[rsp_other];
        poly_corr_im += my_ptr->poly_im[rsp] * my_ptr->poly_re[rsp_other] -
                        my_ptr->poly_re[rsp] * my_ptr->poly_im[rsp_other];
        
        barion_re += base_re * my_ptr->poly_re[rsp_vertex] -
                     base_im * my_ptr->poly_im[rsp_vertex];
        barion_im += base_im * my_ptr->poly_re[rsp_vertex] +
                     base_re * my_ptr->poly_im[rsp_vertex];
    }

    poly_corr_re *= geo->d_inv_space_vol;
    poly_corr_im *= geo->d_inv_space_vol;

    barion_re *= geo->d_inv_space_vol;
    barion_im *= geo->d_inv_space_vol;

    fprintf(datafilep, "%ld %.12f %.12f %.12f %.12f ", GC->update_index, plaqs, plaqt, mean_poly_re, mean_poly_im); // local obs
    fprintf(datafilep, "%.12f %.12f %.12f %.12f \n", poly_corr_re, poly_corr_im, barion_re, barion_im);
    fflush(datafilep);
}

void perform_measures_barion_multihit(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      GParam const * const param,
                                      FILE * datafilep,
                                      Barion_Aux * my_ptr){

    double mean_poly_re = 0, mean_poly_im = 0;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) reduction(+:mean_poly_re) reduction(+:mean_poly_im)
    #endif
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        GAUGE_GROUP mtrx;
        one(&mtrx);

        long r = sisp_and_t_to_si(geo, rsp, 0);
        for (int t = 0; t < geo->d_size[0]; t++) { // POSSIBLE BUG / SMTH I DON'T UNDERSTAND...
            GAUGE_GROUP sum, staple, hitting_mtrx;

            equal(&sum, &(GC->lattice[r][0]));
            
            calcstaples_wilson(GC, geo, r, 0, &staple);
            times_equal_real(&staple, param->d_beta);
            
            equal(&hitting_mtrx, &(GC->lattice[r][0]));
            
            for (int i = 1; i < param->d_multihit; i++) {
                single_heatbath(&hitting_mtrx, &staple);
                plus_equal(&sum, &hitting_mtrx);
                unitarize(&hitting_mtrx);
            }
            times_equal_real(&sum, 1. / param->d_multihit);
            times_equal(&mtrx, &sum);
            r = nnp(geo, r, 0);
        }

        my_ptr->poly_re[rsp] = retr(&mtrx);
        my_ptr->poly_im[rsp] = imtr(&mtrx);

        mean_poly_re += my_ptr->poly_re[rsp];
        mean_poly_im += my_ptr->poly_im[rsp];
    }
    mean_poly_re *= geo->d_inv_space_vol;
    mean_poly_im *= geo->d_inv_space_vol;

    double plaqs, plaqt;
    plaquette(GC, geo, &plaqs, &plaqt);

    double barion_re = 0, barion_im = 0;
    double poly_corr_re = 0, poly_corr_im = 0;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) reduction(+:barion_re) reduction(+:barion_im) reduction(+:poly_corr_re) reduction(+:poly_corr_im)
    #endif
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        long r = sisp_and_t_to_si(geo, rsp, 0);
        long r_vertex = r;
        int half_way = param->d_triangle_base / 2;
        for (int i = 0; i <  half_way; i++) {
            r_vertex = nnp(geo, r_vertex, 1);
        }
        long r_other = r_vertex;
        for (int i = half_way; i < param->d_triangle_base; i++) {
            r_other = nnp(geo, r_other, 1);
        }
        for (int i = 0; i < param->d_triangle_hight; i++) {
            r_vertex = nnp(geo, r_vertex, 2);
        }

        long rsp_other, rsp_vertex;
        int t_tmp;
        si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r_other);
        si_to_sisp_and_t(&rsp_vertex, &t_tmp, geo, r_vertex);

        double base_re = my_ptr->poly_re[rsp] * my_ptr->poly_re[rsp_other] -
                         my_ptr->poly_im[rsp] * my_ptr->poly_im[rsp_other];
        double base_im = my_ptr->poly_im[rsp] * my_ptr->poly_re[rsp_other] +
                         my_ptr->poly_re[rsp] * my_ptr->poly_im[rsp_other];

        poly_corr_re += my_ptr->poly_re[rsp] * my_ptr->poly_re[rsp_other] +
                        my_ptr->poly_im[rsp] * my_ptr->poly_im[rsp_other];
        poly_corr_im += my_ptr->poly_im[rsp] * my_ptr->poly_re[rsp_other] -
                        my_ptr->poly_re[rsp] * my_ptr->poly_im[rsp_other];
        
        barion_re += base_re * my_ptr->poly_re[rsp_vertex] -
                     base_im * my_ptr->poly_im[rsp_vertex];
        barion_im += base_im * my_ptr->poly_re[rsp_vertex] +
                     base_re * my_ptr->poly_im[rsp_vertex];
    }

    poly_corr_re *= geo->d_inv_space_vol;
    poly_corr_im *= geo->d_inv_space_vol;

    barion_re *= geo->d_inv_space_vol;
    barion_im *= geo->d_inv_space_vol;

    fprintf(datafilep, "%ld %.12f %.12f %.12f %.12f ", GC->update_index, plaqs, plaqt, mean_poly_re, mean_poly_im); // local obs
    fprintf(datafilep, "%.12f %.12f %.12f %.12f \n", poly_corr_re, poly_corr_im, barion_re, barion_im);
    fflush(datafilep);
}

void perform_measures_barion_onelevel(Gauge_Conf * GC,
                                      Geometry const * const geo,
                                      GParam const * const param,
                                      FILE * datafilep,
                                      Barion_Aux * my_ptr)
{
    const int slab_num = geo->d_size[0] / param->d_ml_step[NLEVELS - 1];
    const int half_way = param->d_triangle_base / 2;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) private(raux)
    #endif
    for (long raux=0; raux<geo->d_space_vol*slab_num; raux++) { // inizialize tensor cubes
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        zero_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]));
        zero_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]));
    }

    // slab update
    for (int updt = 0; updt < param->d_ml_upd[NLEVELS - 1]; updt++) { // d_ml_upd[0] or d_ml_upd[NLEVEL - 1] ?
        update_for_multilevel(GC, geo, param, NLEVELS - 1);
        compute_local_poly(GC, geo, param);

        #ifdef OPENMP_MODE
        #pragma omp parallel for num_threads(NTHREADS) private(raux)
        #endif
        for (long raux=0; raux<geo->d_space_vol*slab_num; raux++) {
            int slab = (int) (raux % slab_num);
            long rsp = raux / slab_num;

            long rsp_other, rsp_vertex;
            long r_base, r_other, r_vertex;

            int tmp_time = 0;
            r_base = sisp_and_t_to_si(geo, rsp, 0);

            r_vertex = r_base;
            for (int i = 0; i < half_way; i++) {
                r_vertex = nnp(geo, r_vertex, 1);
            }

            r_other = r_vertex;
            for (int i = half_way; i < param->d_triangle_base; i++) {
                r_other = nnp(geo, r_other, 1);
            }

            for (int i = 0; i < param->d_triangle_hight; i++) {
                r_vertex = nnp(geo, r_vertex, 2);
            }

            si_to_sisp_and_t(&rsp_other, &tmp_time, geo, r_other);
            si_to_sisp_and_t(&rsp_vertex, &tmp_time, geo, r_vertex);

            TensProd TP;
            TensProd_init(&TP,
                          &(GC->loc_poly[slab][rsp]),
                          &(GC->loc_poly[slab][rsp_other]));
            plus_equal_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]), &TP);

            TensCube TC;
            TensCube_init(&TC,
                          &(GC->loc_poly[slab][rsp]),
                          &(GC->loc_poly[slab][rsp_other]),
                          &(GC->loc_poly[slab][rsp_vertex]));

            plus_equal_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]), &TC);
        }
    } // end of slub update

    double normalization = 1. / param->d_ml_upd[NLEVELS - 1];
    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) private(raux)
    #endif
    for(long raux = 0; raux < geo->d_space_vol * slab_num; raux++){
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        times_equal_real_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]), normalization);
        times_equal_real_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]), normalization);
    }

    double barion_re = 0, barion_im = 0, polycorr_re = 0, polycorr_im = 0;
    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) private(raux) reduction(+:barion_re) reduction(+:barion_im)
    #endif
    for (long rsp=0; rsp < geo->d_space_vol; rsp++) { // accumulation of slabs
        
        for (int slab = 1; slab < slab_num; slab++) {
            times_equal_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]), 
                                    &(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]));

            times_equal_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]), 
                                 &(GC->ml_polycorr[NLEVELS - 1][slab][rsp]));
        }

        barion_re += retr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));
        barion_im += imtr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));

        polycorr_re += retr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));
        polycorr_im += imtr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));

    }

    normalization = 1. / ((double) geo->d_space_vol);
    barion_im *= normalization;
    barion_re *= normalization;
    polycorr_im *= normalization;
    polycorr_re *= normalization;

    fprintf(datafilep, "%ld %.12f %.12f %.12f %.12f \n", GC->update_index, polycorr_re, polycorr_im, barion_re, barion_im);
    fflush(datafilep);
}

void perform_measures_barion_onelevel_multiT(Gauge_Conf * GC,
                                             Geometry const * const geo,
                                             GParam const * const param,
                                             FILE * datafilep,
                                             Barion_Aux * my_ptr)
{
    const int slab_num = geo->d_size[0] / param->d_ml_step[NLEVELS - 1];
    const int half_way = param->d_triangle_base / 2;

    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) private(raux)
    #endif
    for (long raux=0; raux<geo->d_space_vol*slab_num; raux++) { // inizialize tensor cubes
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        zero_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]));
        zero_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]));
    }

    // slab update
    for (int updt = 0; updt < param->d_ml_upd[NLEVELS - 1]; updt++) { // d_ml_upd[0] or d_ml_upd[NLEVEL - 1] ?
        update_for_multilevel(GC, geo, param, NLEVELS - 1);
        compute_local_poly(GC, geo, param);

        #ifdef OPENMP_MODE
        #pragma omp parallel for num_threads(NTHREADS) private(raux)
        #endif
        for (long raux=0; raux<geo->d_space_vol*slab_num; raux++) {
            int slab = (int) (raux % slab_num);
            long rsp = raux / slab_num;

            long rsp_other, rsp_vertex;
            long r_base, r_other, r_vertex;

            int tmp_time = 0;
            r_base = sisp_and_t_to_si(geo, rsp, 0);

            r_vertex = r_base;
            for (int i = 0; i < half_way; i++) {
                r_vertex = nnp(geo, r_vertex, 1);
            }

            r_other = r_vertex;
            for (int i = half_way; i < param->d_triangle_base; i++) {
                r_other = nnp(geo, r_other, 1);
            }

            for (int i = 0; i < param->d_triangle_hight; i++) {
                r_vertex = nnp(geo, r_vertex, 2);
            }

            si_to_sisp_and_t(&rsp_other, &tmp_time, geo, r_other);
            si_to_sisp_and_t(&rsp_vertex, &tmp_time, geo, r_vertex);

            TensProd TP;
            TensProd_init(&TP,
                          &(GC->loc_poly[slab][rsp]),
                          &(GC->loc_poly[slab][rsp_other]));
            plus_equal_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]), &TP);

            TensCube TC;
            TensCube_init(&TC,
                          &(GC->loc_poly[slab][rsp]),
                          &(GC->loc_poly[slab][rsp_other]),
                          &(GC->loc_poly[slab][rsp_vertex]));

            plus_equal_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]), &TC);
        }
    } // end of slub update

    double normalization = 1. / param->d_ml_upd[NLEVELS - 1];
    #ifdef OPENMP_MODE
    #pragma omp parallel for num_threads(NTHREADS) private(raux)
    #endif
    for(long raux = 0; raux < geo->d_space_vol * slab_num; raux++){
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        times_equal_real_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]), normalization);
        times_equal_real_TensProd(&(GC->ml_polycorr[NLEVELS - 1][slab][rsp]), normalization);
    }

    for (int slab = 0; slab < slab_num; slab++) {
        my_ptr->barion_re[slab] = 0;
        my_ptr->barion_im[slab] = 0;
        my_ptr->poly_re[slab] = 0;
        my_ptr->poly_im[slab] = 0;
    }
    for (long rsp=0; rsp < geo->d_space_vol; rsp++) { // accumulation of slabs
        

        my_ptr->barion_re[0] += retr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));
        my_ptr->barion_im[0] += imtr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));

        my_ptr->poly_re[0] += retr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));
        my_ptr->poly_im[0] += imtr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));

        for (int slab = 1; slab < slab_num; slab++) {
            times_equal_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]), 
                                    &(my_ptr->poly_slab[NLEVELS - 1][slab][rsp]));

            times_equal_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]), 
                                 &(GC->ml_polycorr[NLEVELS - 1][slab][rsp]));

            my_ptr->barion_re[slab] += retr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));
            my_ptr->barion_im[slab] += imtr_TensCube(&(my_ptr->poly_slab[NLEVELS - 1][0][rsp]));

            my_ptr->poly_re[slab] += retr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));
            my_ptr->poly_im[slab] += imtr_TensProd(&(GC->ml_polycorr[NLEVELS - 1][0][rsp]));
        }
    }

    normalization = 1. / ((double) geo->d_space_vol);
    for (int slab = 0; slab < slab_num; slab++) {
        my_ptr->barion_re[slab] *= normalization;
        my_ptr->barion_im[slab] *= normalization;
        my_ptr->poly_re[slab] *= normalization;
        my_ptr->poly_im[slab] *= normalization;
    }

    fprintf(datafilep, "%ld", GC->update_index);
    for (int slab = 0; slab < slab_num; slab++) {
        fprintf(datafilep, " %.12f %.12f", my_ptr->poly_re[slab], my_ptr->poly_im[slab]);
        fprintf(datafilep, " %.12f %.12f", my_ptr->barion_re[slab], my_ptr->barion_re[slab]);
    }
    fprintf(datafilep, "\n");
    fflush(datafilep);
}

void print_parameters_barion(GParam const * const param,
                             time_t start,
                             time_t end){
    FILE *fp;
    int i;
    double diff_sec;

    fp=fopen(param->d_log_file, "w");
    fprintf(fp, "+---------------------------------------------+\n");
    fprintf(fp, "| Simulation details for su3_barion           |\n");
    fprintf(fp, "+---------------------------------------------+\n\n");

    #ifdef OPENMP_MODE
     fprintf(fp, "using OpenMP with %d threads\n\n", NTHREADS);
    #endif

    fprintf(fp, "number of colors: %d\n", NCOLOR);
    fprintf(fp, "spacetime dimensionality: %d\n\n", STDIM);

    fprintf(fp, "lattice: %d", param->d_sizeg[0]);
    for(i=1; i<STDIM; i++)
       {
       fprintf(fp, "x%d", param->d_sizeg[i]);
       }
    fprintf(fp, "\n\n");

    fprintf(fp, "beta: %.10lf\n", param->d_beta);
    #ifdef THETA_MODE
      fprintf(fp, "theta: %.10lf\n", param->d_theta);
    #endif
    fprintf(fp, "\n");

    fprintf(fp, "sample:    %d\n", param->d_sample);
    fprintf(fp, "thermal:   %d\n", param->d_thermal);
    fprintf(fp, "overrelax: %d\n", param->d_overrelax);
    fprintf(fp, "measevery: %d\n", param->d_measevery);
    fprintf(fp, "\n");

    fprintf(fp, "start:                   %d\n", param->d_start);
    fprintf(fp, "saveconf_back_every:     %d\n", param->d_saveconf_back_every);
    fprintf(fp, "\n");

    fprintf(fp, "triangle base:           %d\n", param->d_triangle_base);
    fprintf(fp, "triangle hight:          %d\n", param->d_triangle_hight);
    fprintf(fp, "multihit:                %d\n", param->d_multihit);
#ifdef BARION_MULTILEVEL
    fprintf(fp, "multilevel update:       %d\n", param->d_ml_upd);
    fprintf(fp, "multilevel slice hight:  %d\n", param->d_ml_step);
#endif
    fprintf(fp, "\n");

    fprintf(fp, "randseed: %u\n", param->d_randseed);
    fprintf(fp, "\n");

    diff_sec = difftime(end, start);
    fprintf(fp, "Simulation time: %.3lf seconds\n", diff_sec );
    fprintf(fp, "\n");

    if(endian()==0)
      {
      fprintf(fp, "Little endian machine\n\n");
      }
    else
      {
      fprintf(fp, "Big endian machine\n\n");
      }

    fclose(fp);
}