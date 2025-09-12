#include "../include/flux_tools.h"

sources_cfg* new_sources_cfg(int positive, int negative)
{
    sources_cfg* ret = NULL;
    ret = malloc(sizeof(sources_cfg));

    int err = 0;
    if (ret == NULL) err = 1;
    int total = positive + negative;
    
    if (err == 0) ret->negative = negative;
    if (err == 0) ret->positive = positive;
    
    for (int mu = 0; mu < STDIM; mu++) if (err == 0)
        err = posix_memalign((void**) &(ret->position[mu]), INT_ALIGN, (size_t) total * sizeof(int));
    
    if (err != 0){
        fprintf(stderr, "Error while allocating source configuration (%s %d %d)", __FILE__,  __LINE__, err);
        exit(EXIT_FAILURE);
    }

    return ret;
}

void free_sources_cfg(sources_cfg* ptr)
{
    for (int mu = 0; mu < STDIM; mu++) free(ptr->position[mu]);
    free(ptr);
}

void smearing_step(Gauge_Conf const *const GC,
                   Geometry const *const geo,
                   double smearing_rho,
                   Gauge_Conf *new_GC)
{
    for (int mu = 0; mu < STDIM; mu++) {
        long r;
        #ifdef OPENMP_MODE
        #pragma omp parallel for num_threads(NTHREADS) private(r)
        #endif 
        for(r=0; r<(geo->d_volume)/2; r++) {
            GAUGE_GROUP staple;
            GAUGE_GROUP link = GC->lattice[r][mu];
            
            calcstaples_wilson(GC, geo, r, mu, &staple);
            times_equal_real(&staple, smearing_rho);
            plus_equal_dag(&link, &staple);

            unitarize(&link);

            new_GC->lattice[r][mu] = link;
        }
        #ifdef OPENMP_MODE
        #pragma omp parallel for num_threads(NTHREADS) private(r)
        #endif 
        for(r=(geo->d_volume)/2; r<(geo->d_volume); r++) {
            GAUGE_GROUP staple;
            GAUGE_GROUP link = GC->lattice[r][mu];
            
            calcstaples_wilson(GC, geo, r, mu, &staple);
            times_equal_real(&staple, smearing_rho);
            plus_equal_dag(&link, &staple);

            unitarize(&link);

            new_GC->lattice[r][mu] = link;
        }
    }
}

void conf_smear_2n(Gauge_Conf * GC,
                   Gauge_Conf * aux_GC,
                   Geometry const * const geo,
                   double smearing_rho,
                   int smeraring_step_half)
{
    for (int i = 0; i < smeraring_step_half; i++){
        smearing_step(GC, geo, smearing_rho, aux_GC);
        smearing_step(aux_GC, geo, smearing_rho, GC);
    }
}

void compute_polyakov_point(Gauge_Conf const * const GC,
                            Geometry const * const geo,
                            long rsp,
                            GAUGE_GROUP * loop)
{
    one(loop);
    int time = 0;
    long r = sisp_and_t_to_si(geo, rsp, time);
    for (time = 0; time < geo->d_size[0]; time++) {
        r = nnp(geo, r, 0);
        times_equal(loop, &(GC->lattice[r][0]));
    }
}

void polyfund_sources_complex_trace(sources_cfg const * const ptr,
                                    Gauge_Conf const * const GC,
                                    Geometry const * const geo,
                                    long rsp_center, 
                                    double *real_part, double *imag_part)
{
    *real_part = 1; *imag_part = 0;
    GAUGE_GROUP loop;

    int cart_center[STDIM];
    int t = 0;
    long r = sisp_and_t_to_si(geo, rsp_center, t);
    si_to_cart(cart_center, r, geo);

    int total = ptr->positive + ptr->negative;
    for (int i = 0; i < total; i++) {
        int cart[STDIM];
        for (int mu = 0; mu < STDIM; mu++) cart[mu] =
            (ptr->position[mu][i] + cart_center[mu]) % geo->d_size[mu];    
        r = cart_to_si(cart, geo);
        long rsp;
        si_to_sisp_and_t(&rsp, &t, geo, r);

        compute_polyakov_point(GC, geo, rsp, &loop);
        double real_poly = retr(&loop);
        double imag_poly = imtr(&loop);

        double new_real_part, new_imag_part;

        if (i < ptr->positive) {
            new_real_part = *real_part * real_poly - *imag_part * imag_poly;
            new_imag_part = *real_part * imag_poly + *imag_part * imag_poly;    
        }
        else {
            new_real_part = *real_part * real_poly + *imag_part * imag_poly;
            new_imag_part = *real_part * imag_poly - *imag_part * imag_poly;
        }

        *real_part = new_real_part;
        *imag_part = new_imag_part;
    }
}

void compute_plaqcol_point(Gauge_Conf const * const GC,
                          Geometry const * const geo,
                          long rsp,
                          int mu, int nu,
                          double *plaq_re, double *plaq_im)
{
    *plaq_re = 0.;
    *plaq_im = 0.;

    for (int t = 0; t < geo->d_size[0]; t++) {
        long r = sisp_and_t_to_si(geo, rsp, t);
        GAUGE_GROUP plaq; plaquettep_matrix(GC, geo, r, mu, nu, &plaq);
        *plaq_re += retr(&plaq);
        *plaq_im += imtr(&plaq);
    }

    double norm = 1. / (double) geo->d_size[0];
    *plaq_re *= norm;
    *plaq_im *= norm;
}

long rsp_difference(long rsp1, long rsp2, Geometry const * const geo)
{
    int cart1[STDIM], cart2[STDIM];
    int time = 0;

    long r = sisp_and_t_to_si(geo, rsp1, time);
    si_to_cart(cart1, r, geo);

    r = sisp_and_t_to_si(geo, rsp2, time);
    si_to_cart(cart2, r, geo);

    for (int mu = 0; mu < STDIM; mu++)
        cart1[mu] = (cart1[mu] - cart2[mu] + geo->d_size[mu]) % geo->d_size[mu];

    r = cart_to_si(cart1, geo);
    long ret; si_to_sisp_and_t(&ret, &time, geo, r);

    return ret;
}

void fluxtube_sweep(Gauge_Conf const * const GC,
                    Geometry const * const geo,
                    GParam const * const param,
                    FILE* flux_filep)
{
    //handy parameters
    int dist = param->d_dist_poly;
    int mu = param->d_plaq_dir[0], nu = param->d_plaq_dir[1];

    // allocate space_vol matrix
    int err = 0;
    double* flux;
    err = posix_memalign((void**) &flux, DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));

    if (err != 0) {
        fprintf(stderr, "Unable to allocate flux array (%s %d %d)", __FILE__, __LINE__, err);
        exit(EXIT_FAILURE);
    }

    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) flux[rsp] = 0;

    // loop on space sites PPdagger
    double poly_corr = 0;
    for (long rsp_poly = 0; rsp_poly < geo->d_space_vol; rsp_poly++) {
        // measure PPdagger
        GAUGE_GROUP poly;
        compute_polyakov_point(GC, geo, rsp_poly, &poly);
        double poly_re = retr(&poly);
        double poly_im = imtr(&poly);

        int time = 0;
        long r_poly = sisp_and_t_to_si(geo, rsp_poly, time);
        for (int x = 0; x < dist; x++) r_poly = nnp(geo, r_poly, 1); // moove dist sites along x
        long rsp_pdag; si_to_sisp_and_t(&rsp_pdag, &time, geo, r_poly);

        compute_polyakov_point(GC, geo, rsp_pdag, &poly);
        double other_re = retr(&poly);
        double other_im = imtr(&poly);
    
        double ppdag_re = poly_re * other_re + poly_im * other_im;
        double ppdag_im = poly_im * other_re - poly_re * other_im;
        poly_corr += ppdag_re;
        
        // inner loop on spcae sites plaquette -> TODO parallelization
        for (long rsp_plaq = 0; rsp_plaq < geo->d_space_vol; rsp_plaq++) {
            double plaq_re, plaq_im;
            // measure time averaged plaquette
            compute_plaqcol_point(GC, geo, rsp_plaq, mu, nu, &plaq_re, &plaq_im);

            double flux_pt = ppdag_re * plaq_re - ppdag_im * plaq_im;
            long delta_rsp = rsp_difference(rsp_plaq, rsp_poly, geo);

            flux[delta_rsp] += flux_pt;
        }
    }

    double norm = 1. / (double) geo->d_space_vol; // 1. / poly_corr; 
    poly_corr *= norm;
    fprintf(flux_filep, "\n");
    fprintf(flux_filep, "%.12g \n", poly_corr);
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        flux[rsp] *= norm;

        int time = 0;
        long r = sisp_and_t_to_si(geo, rsp, time);
        int cart[STDIM]; si_to_cart(cart, r, geo);

        for (int mu = 0; mu < STDIM; mu++) fprintf(flux_filep, "%d ", cart[mu]);
        fprintf(flux_filep, "%.12g \n", flux[rsp]);
    }

    free(flux);
}

void generic_fundsources_sweep(Gauge_Conf const * const GC,
                               Geometry const * const geo,
                               GParam const * const param,
                               sources_cfg const * const sources,
                               FILE* flux_filep)
{
    int mu = param->d_plaq_dir[0], nu = param->d_plaq_dir[1];

    // allocate flux matrix
    int err = 0;
    double* flux;
    err = posix_memalign((void**) &flux, DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));

    // allocate plaquette matrix
    double* plaq_re;
    if (err == 0) err = posix_memalign((void**) &plaq_re, DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));

    double* plaq_im;
    if (err == 0) err = posix_memalign((void**) &plaq_im, DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));

    if (err != 0) {
        fprintf(stderr, "Error while preparing for a sweep (%s %d %d)", __FILE__, __LINE__, err);
        exit(EXIT_FAILURE);
    }

    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        compute_plaqcol_point(GC, geo, rsp, mu, nu, &(plaq_re[rsp]), &(plaq_im[rsp]));
    }

    double source_re, source_im;
    double source_tot = 0;
    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        polyfund_sources_complex_trace(sources, GC, geo, rsp, &source_re, &source_im);
        source_tot += source_re;

        for (long rsp_plaq = 0; rsp_plaq < geo->d_space_vol; rsp_plaq++) {
            double flux_pt = source_re * plaq_re[rsp_plaq] - source_im * plaq_im[rsp_plaq];
            long delta_rsp = rsp_difference(rsp_plaq, rsp, geo);

            flux[delta_rsp] += flux_pt;
        }
    }

    free(plaq_re);
    free(plaq_im);
    
    source_tot *= geo->d_inv_space_vol;
    fprintf(flux_filep, "\n");
    fprintf(flux_filep, "%.12g \n", source_tot);

    for (long rsp = 0; rsp < geo->d_space_vol; rsp++) {
        flux[rsp] *= geo->d_inv_space_vol;

        int t = 0;
        long r = sisp_and_t_to_si(geo, rsp, t);
        int cart[STDIM]; si_to_cart(cart, r, geo);

        for (int mu = 0; mu < STDIM; mu++) fprintf(flux_filep, "%d ", cart[mu]);
        fprintf(flux_filep, "%.12g \n", flux[rsp]);
    }

    free(flux);
}