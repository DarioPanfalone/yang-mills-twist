#include"../include/tube_multilevel.h"
#include "../include/function_pointers.h"

void compute_locals_slab(Gauge_Conf *GC,
                         Geometry const * const geo,
                         GParam const * const param,
                         TubeStuff * tube_ptrs)
{
    #ifdef THETA_MODE
    // clovers are eventually needed by the multihit
    compute_clovers(GC, geo, 0);
    #endif

    const int slab_wdt = param->d_ml_step[NLEVELS-1];
    const int slab_num = geo->d_size[0] / slab_wdt;

    for (long raux = 0; raux < geo->d_space_vol * slab_num; raux++) {
        long rsp = raux / slab_num;
        int slab = (int) (raux % slab_num);

        GAUGE_GROUP matrix; one(&matrix);
        one(&(tube_ptrs->local_poly[slab][rsp]));
        for (int t = 0; t < slab_wdt; t++) {
            int time = slab * slab_wdt + t;
            long r = sisp_and_t_to_si(geo, rsp, time);
            multihit(GC, geo, param, r, 0, param->d_multihit, &matrix);
            times_equal(&(tube_ptrs->local_poly[slab][rsp]), &matrix);
        }
    }

    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        long r = sisp_and_t_to_si(geo, rsp, 1); // time = 1
        tube_ptrs->local_plaq[rsp] = plaquettep_complex(GC, geo, r, param->d_plaq_dir[0], param->d_plaq_dir[1]);
    }
}

void perform_measures_onelevel_tube_vary(Gauge_Conf *GC,
                                Geometry const * const geo,
                                GParam const * const param,
                                TubeStuff * tube_ptrs,
                                FILE *datafilep)
{
    int slab_num = geo->d_size[0]/param->d_ml_step[NLEVELS - 1];
    int dist_num = 2 * param->d_trasv_dist + 1;

    for (long raux = 0; raux < slab_num * geo->d_space_vol; raux++) { // setup poly corr
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        zero_TensProd(&(tube_ptrs->slab_polycorr[slab][rsp]));
    }

    for (long raux = 0; raux < dist_num * geo->d_space_vol; raux++) { // setup poly plaq
        int dist_idx = (int) (raux % dist_num);
        // int dist_sgn = dist_idx - param->d_trasv_dist;
        long rsp = raux / dist_num;

        zero_TensProd(&(tube_ptrs->slab_polyplaq[dist_idx][rsp]));
    }

    for (int upd = 0; upd < param->d_ml_upd[NLEVELS - 1]; upd++) { // update cycle
        update_for_multilevel(GC, geo, param, NLEVELS - 1);

        compute_locals_slab(GC, geo, param, tube_ptrs);

        for (int raux = 0; raux < (slab_num - 1) * geo->d_space_vol; raux++) { // all but first slab
            long rsp = raux / (slab_num - 1);
            int slab = (raux % (slab_num - 1)) + 1;

            long r = sisp_and_t_to_si(geo, rsp, 0);
            for (int i = 0; i < param->d_dist_poly; i++) { // go dist step in direction 1
                r = nnp(geo, r, 1);
            }

            int tmp_time = 0;
            long rsp_other;
            si_to_sisp_and_t(&rsp_other, &tmp_time, geo, r);

            TensProd TP;
            TensProd_init(&TP, &(tube_ptrs->local_poly[slab][rsp]), &(tube_ptrs->local_poly[slab][rsp_other]));
            plus_equal_TensProd(&(tube_ptrs->slab_polycorr[slab][rsp]), &TP);
        }

        const int half_way = param->d_dist_poly / 2;
        for (long rsp = 0; rsp < geo->d_space_vol; rsp++) { // first slab
            int slab = 0;
            long r = sisp_and_t_to_si(geo, rsp, 0);
            for (int i = 0; i < half_way; i++) { // go dist step in direction 1
                r = nnp(geo, r, 1);
            }
            long r_halfway = r;
            for (int i = half_way; i < param->d_dist_poly; i++) {
                r = nnp(geo, r, 1);
            }

            int tmp_time = 0;
            long rsp_other;
            si_to_sisp_and_t(&rsp_other, &tmp_time, geo, r);

            TensProd TP;
            TensProd_init(&TP, &(tube_ptrs->local_poly[slab][rsp]), &(tube_ptrs->local_poly[slab][rsp_other]));

            plus_equal_TensProd(&(tube_ptrs->slab_polycorr[slab][rsp]), &TP);

            for (int dist_idx = 0; dist_idx < dist_num; dist_idx++) {
                int dist = dist_idx - param->d_trasv_dist;

                long r_vertex = r_halfway;
                if (dist >=  0) {
                    for (int i = 0; i < dist; i++) r_vertex = nnp(geo, r_vertex, 2);
                }
                else {
                    dist = -dist;
                    for (int i = 0; i < dist; i++) r_vertex = nnm(geo, r_vertex, 2);
                }

                long rsp_vertex;
                si_to_sisp_and_t(&rsp_vertex, &tmp_time, geo, r_vertex);

                TensProd plaqTP; equal_TensProd(&plaqTP, &TP);
                times_equal_complex_TensProd(&plaqTP, tube_ptrs->local_plaq[rsp_vertex]);
                plus_equal_TensProd(&(tube_ptrs->slab_polyplaq[dist_idx][rsp]), &plaqTP);
            }
        }
    } //end of update cycle

    double normalization = 1. / param->d_ml_upd[NLEVELS - 1];
    for (long raux = 0; raux < slab_num * geo->d_space_vol; raux++) { // normalize poly corr
        int slab = (int) (raux % slab_num);
        long rsp = raux / slab_num;

        times_equal_real_TensProd(&(tube_ptrs->slab_polycorr[slab][rsp]), normalization);
    }

    for (long raux = 0; raux < dist_num * geo->d_space_vol; raux++) { // normalize poly plaq
        int dist_idx = (int) (raux % dist_num);
        long rsp = raux / dist_num;

        times_equal_real_TensProd(&(tube_ptrs->slab_polyplaq[dist_idx][rsp]), normalization);
    }

    for (int slab = 1; slab < slab_num; slab++) { // slab accumulation
        for (long rsp = 0; rsp <  geo->d_space_vol; rsp++) {
            TensProd* which_TP = &(tube_ptrs->slab_polycorr[slab][rsp]);
            times_equal_TensProd(&(tube_ptrs->slab_polycorr[0][rsp]), which_TP);
            for (int dist = 0; dist < dist_num; dist++) {
                times_equal_TensProd(&(tube_ptrs->slab_polyplaq[dist][rsp]), which_TP);
            }
        }
    }

    double polycorr_re = 0., polycorr_im = 0.;
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) { // average poly corr
        polycorr_re += retr_TensProd(&(tube_ptrs->slab_polycorr[0][rsp]));
        polycorr_im += imtr_TensProd(&(tube_ptrs->slab_polycorr[0][rsp]));
    }
    polycorr_re *= geo->d_inv_space_vol;
    polycorr_im *= geo->d_inv_space_vol;

    fprintf(datafilep, "%ld ", GC->update_index);
    fprintf(datafilep, "%.12g %.12g ", polycorr_re, polycorr_im);

    for (int dist = 0; dist < dist_num; dist++) { //average poly plaq
        double polyplaq_re = 0., polyplaq_im = 0.;
        for (long rsp = 0; rsp < geo->d_space_vol; rsp++) { // average poly corr
            polyplaq_re += retr_TensProd(&(tube_ptrs->slab_polyplaq[dist][rsp]));
            polyplaq_im += imtr_TensProd(&(tube_ptrs->slab_polyplaq[dist][rsp]));
        }
        polyplaq_re *= geo->d_inv_space_vol;
        polyplaq_im *= geo->d_inv_space_vol;

        fprintf(datafilep, "%.12g %.12g ", polyplaq_re, polyplaq_im);
    }

    fprintf(datafilep, "\n"); fflush(datafilep);
}

void init_TubeStuff(GParam const * const param,
                    Geometry const * const geo,
                    TubeStuff * tube_ptrs)
{
    int slab_num = geo->d_size[0] / param->d_ml_step[NLEVELS - 1];
    int dist_num = 2 * param->d_trasv_dist + 1;
    int err = 0;

    // GAUGE_GROUP local_poly[slab][rsp]
    err = posix_memalign((void **) &(tube_ptrs->local_poly), (size_t) DOUBLE_ALIGN, (size_t) slab_num * sizeof(GAUGE_GROUP *));
    if (err != 0) {
        fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    for (int slab = 0; slab < slab_num; slab++) {
        err = posix_memalign((void **) &(tube_ptrs->local_poly[slab]), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(GAUGE_GROUP));
        if (err != 0) {
            fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // double __complex__ *local_plaq
    err=posix_memalign((void**) &(tube_ptrs->local_plaq), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol *sizeof(double complex));
    if(err!=0) {
        fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // TensProd slab_polycorr[slab][rsp]
    err = posix_memalign((void **) &(tube_ptrs->slab_polycorr), (size_t) DOUBLE_ALIGN, (size_t) slab_num * sizeof(TensProd*));
    if(err!=0) {
        fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    for (int slab = 0; slab < slab_num; slab++) {
        err =posix_memalign((void **) &(tube_ptrs->slab_polycorr[slab]), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(TensProd));
        if (err != 0) {
            fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // TensProd slab_polyplaq[dist][rsp]
    err = posix_memalign((void **) &(tube_ptrs->slab_polyplaq), (size_t) DOUBLE_ALIGN, (size_t) dist_num * sizeof(TensProd*));
    if(err!=0) {
        fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    for (int dist = 0; dist < dist_num; dist++) {
        err = posix_memalign((void **) &(tube_ptrs->slab_polyplaq[dist]), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(TensProd));
        if (err != 0) {
            fprintf(stderr, "Problems allocating TubeStuff (%s, %d)\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }
}

void free_TubeStuff(GParam const * const param,
                    Geometry const * const geo,
                    TubeStuff * tube_ptrs)
{
    int slab_num = geo->d_size[0] / param->d_ml_step[NLEVELS - 1];
    int dist_num = 2 * param->d_trasv_dist + 1;

    for (int dist = 0; dist < dist_num; dist++) free(tube_ptrs->slab_polyplaq[dist]);
    free(tube_ptrs->slab_polyplaq);

    for (int slab = 0; slab < slab_num; slab++) free(tube_ptrs->slab_polycorr[slab]);
    free(tube_ptrs->slab_polycorr);

    free(tube_ptrs->local_plaq);

    for (int slab = 0; slab < slab_num; slab++) free(tube_ptrs->local_poly[slab]);
    free(tube_ptrs->local_poly);
}
