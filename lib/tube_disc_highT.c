#ifndef LIB_FOR_TUBE_DISC_C
#define LIB_FOR_TUBE_DISC_C

#include"../include/macro.h"

#include<complex.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#include"../include/gparam.h"
#include"../include/function_pointers.h"
#include"../include/geometry.h"
#include"../include/gauge_conf.h"
#include "../include/tube_disc_highT.h"

void print_cartesian(Geometry const * const geo,
                     long r)
{
   int cart[STDIM];
   si_to_cart(cart, r, geo);

   for (int i = 0; i < STDIM; i++) {
      printf("%d \t", cart[i]);
   }
}

void init_tube_file(FILE **dataf, GParam const * const param)
{
  if(param->d_start==2)
    {
    *dataf=fopen(param->d_tube_file, "r");
    if(*dataf!=NULL) // file exists
      {
      fclose(*dataf);
      *dataf=fopen(param->d_tube_file, "a");
      fflush(*dataf);
      return;
      }
    }
  *dataf=fopen(param->d_tube_file, "w");
  fprintf(*dataf, "%d %d %d\n", param->d_poly_dist_min, param->d_poly_dist_max, param->d_poly_dist_step);
  fflush(*dataf);
}

void init_plaq_file(FILE **dataf, GParam const *const param)
{
   if(param->d_start==2)
    {
    *dataf=fopen(param->d_plaq_file, "r");
    if(*dataf!=NULL) // file exists
      {
      fclose(*dataf);
      *dataf=fopen(param->d_plaq_file, "a");
      fflush(*dataf);
      return;
      }
    }
  *dataf=fopen(param->d_plaq_file, "w");
  fprintf(*dataf, "%d ", STDIM);
  for(int i=0; i<STDIM; i++)
      {
      fprintf(*dataf, "%d ", param->d_sizeg[i]);
      }
  fprintf(*dataf, "\n");
  fflush(*dataf);
}

void init_poly_file(FILE **dataf, GParam const *const param)
{
   if(param->d_start==2)
    {
    *dataf=fopen(param->d_poly_file, "r");
    if(*dataf!=NULL) // file exists
      {
      fclose(*dataf);
      *dataf=fopen(param->d_poly_file, "a");
      fflush(*dataf);
      return;
      }
    }
  *dataf=fopen(param->d_poly_file, "w");
  fprintf(*dataf, "%d ", STDIM);
  for(int i=0; i<STDIM; i++)
      {
      fprintf(*dataf, "%d ", param->d_sizeg[i]);
      }
  fprintf(*dataf, "\n");
  fflush(*dataf);
}

void init_padj_file(FILE **dataf, GParam const *const param)
{
   if(param->d_start==2)
    {
    *dataf=fopen(param->d_padj_file, "r");
    if(*dataf!=NULL) // file exists
      {
      fclose(*dataf);
      *dataf=fopen(param->d_padj_file, "a");
      fflush(*dataf);
      return;
      }
    }
  *dataf=fopen(param->d_padj_file, "w");
  fprintf(*dataf, "%d ", STDIM);
  for(int i=0; i<STDIM; i++)
      {
      fprintf(*dataf, "%d ", param->d_sizeg[i]);
      }
  fprintf(*dataf, "\n");
  fflush(*dataf);
}

void compute_polyakov_tube_disc_highT(Gauge_Conf const *const GC,
                                      Geometry const *const geo,
                                      Tube_disc_ptrs *my_ptrs)
{
   long rsp;
   // compute all the polyakov
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r;
      GAUGE_GROUP matrix;

      r=sisp_and_t_to_si(geo, rsp, 0);

      one(&matrix);
      for(int t=0; t<geo->d_size[0]; t++)
         {
         times_equal(&matrix, &(GC->lattice[r][0]));
         r=nnp(geo, r, 0);
         }
      my_ptrs->aux_poly_r[rsp] = retr(&matrix);
      my_ptrs->aux_poly_i[rsp] = imtr(&matrix);
   }
}

void compute_polyakov_tube_mom_highT(Gauge_Conf const *const GC,
                                      Geometry const *const geo,
                                      Tube_disc_ptrs *my_ptrs,
                                      int n )
{
   long rsp;
   for (int y=0;y<geo->d_size[2];y++){
        my_ptrs -> aux_mom_poly_r[y] = 0;
        my_ptrs -> aux_mom_poly_i[y] = 0;
   }
   // compute all the polyakov
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r;
      GAUGE_GROUP matrix;

      r=sisp_and_t_to_si(geo, rsp, 0);

      one(&matrix);
      
      for(int t=0; t<geo->d_size[0]; t++)
         {
         times_equal(&matrix, &(GC->lattice[r][0]));
         r=nnp(geo, r, 0);
         }
      int cart[STDIM];
      si_to_cart(cart,r,geo); 
      int y=cart[2];
      int x=cart[1];
      
      double k = ((2.0 * PI * n * geo -> d_inv_space_vol)/(NCOLOR));
      //double phase = exp(- I * k * x);
      
      my_ptrs -> aux_mom_poly_r[y] += cos(k * x)*retr(&matrix) + sin(k * x)*imtr(&matrix);
      my_ptrs -> aux_mom_poly_i[y] += cos(k * x)*imtr(&matrix) - sin(k * x)*retr(&matrix);

      //my_ptrs->aux_poly_r[rsp] = retr(&matrix);
      //my_ptrs->aux_poly_i[rsp] = imtr(&matrix);
   }
      for (int y=0; y<geo->d_size[2]; y++) {
       my_ptrs->aux_mom_poly_r[y] *= geo->d_inv_space_vol;
       my_ptrs->aux_mom_poly_i[y] *= geo->d_inv_space_vol;
   }
}

void compute_plaquette_columns(Gauge_Conf const * const GC,
                               Geometry const * const geo,
                               GParam const * const param,
                               Tube_disc_ptrs * my_ptrs)
{
   long rsp;
   // compute all the plaquette columns
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r;
      GAUGE_GROUP matrix;

      r=sisp_and_t_to_si(geo, rsp, 0);

      my_ptrs->plaq_column_r[rsp] = 0;
      my_ptrs->plaq_column_i[rsp] = 0;

      for (int t=0; t<geo->d_size[0]; t++) {
         plaquettep_matrix(GC, geo, r, param->d_plaq_dir[1], param->d_plaq_dir[0], &matrix);
         my_ptrs->plaq_column_r[rsp] += retr(&matrix);
         my_ptrs->plaq_column_i[rsp] += imtr(&matrix);
         r = nnp(geo, r, 0);
      }
      my_ptrs->plaq_column_r[rsp] /= geo->d_size[0];
      my_ptrs->plaq_column_i[rsp] /= geo->d_size[0];
   }
}

void faster_measures_tube_disc_givenR(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      int poly_dist,
                                      Tube_disc_ptrs const * const my_ptrs)
{
   const long space_len = geo->d_size[2];
   double two_pts_r = 0, two_pts_i = 0;
   double *three_pts_r = my_ptrs->three_pts_r, *three_pts_i = my_ptrs->three_pts_i;
   for (int y = 0; y < space_len; y++) {
      three_pts_i[y] = 0;
      three_pts_r[y] = 0;
   }

   long rsp;   // now the plaquete position
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+:three_pts_r[:space_len]) reduction(+:three_pts_i[:space_len]) reduction(+:two_pts_r) reduction(+:two_pts_i)
   #endif
   for (rsp = 0; rsp < geo->d_space_vol; rsp++) {
      // plaquette
      double plaq_r = my_ptrs->plaq_column_r[rsp];
      double plaq_i = my_ptrs->plaq_column_i[rsp];

      // Polyakov
      long rsp_poly1, rsp_poly2;
      long r_poly1, r_poly2;
      int t_tmp;
      r_poly1 = r_poly2 = sisp_and_t_to_si(geo, rsp, 0);
      for (int i = 0; i < poly_dist; i++) {
         r_poly1 = nnm(geo, r_poly1, 1);
         r_poly2 = nnp(geo, r_poly2, 1);
      }

      si_to_sisp_and_t(&rsp_poly1, &t_tmp, geo, r_poly1);
      si_to_sisp_and_t(&rsp_poly2, &t_tmp, geo, r_poly2);

      double poly_prod_r = my_ptrs->aux_poly_r[rsp_poly1] * my_ptrs->aux_poly_r[rsp_poly2] +
                           my_ptrs->aux_poly_i[rsp_poly1] * my_ptrs->aux_poly_i[rsp_poly2];
      double poly_prod_i = my_ptrs->aux_poly_i[rsp_poly1] * my_ptrs->aux_poly_r[rsp_poly2] -
                           my_ptrs->aux_poly_r[rsp_poly1] * my_ptrs->aux_poly_i[rsp_poly2];
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;

      three_pts_r[0] += poly_prod_r * plaq_r - poly_prod_i * plaq_i;
      three_pts_i[0] += poly_prod_i * plaq_r + poly_prod_r * plaq_i;

      for (int y = 1; y < space_len; y++) {
         r_poly1 = nnp(geo, r_poly1, 2);
         r_poly2 = nnp(geo, r_poly2, 2);
         
         si_to_sisp_and_t(&rsp_poly1, &t_tmp, geo, r_poly1);
         si_to_sisp_and_t(&rsp_poly2, &t_tmp, geo, r_poly2);

         double poly_prod_r = my_ptrs->aux_poly_r[rsp_poly1] * my_ptrs->aux_poly_r[rsp_poly2] +
                              my_ptrs->aux_poly_i[rsp_poly1] * my_ptrs->aux_poly_i[rsp_poly2];
         double poly_prod_i = my_ptrs->aux_poly_i[rsp_poly1] * my_ptrs->aux_poly_r[rsp_poly2] -
                              my_ptrs->aux_poly_r[rsp_poly1] * my_ptrs->aux_poly_i[rsp_poly2];
         
         three_pts_r[y] += poly_prod_r * plaq_r - poly_prod_i * plaq_i;
         three_pts_i[y] += poly_prod_i * plaq_r + poly_prod_r * plaq_i;
      }
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;

   fprintf(my_ptrs->tubefilep, "%ld %d ", GC->update_index, poly_dist);
   fprintf(my_ptrs->tubefilep, "%.12g %.12g ", two_pts_r, two_pts_i);

   for (int y = 0; y < space_len; y++) {
      three_pts_r[y] *= geo->d_inv_space_vol;
      three_pts_i[y] *= geo->d_inv_space_vol;

      fprintf(my_ptrs->tubefilep, "%.12g %.12g ", three_pts_r[y], three_pts_i[y]);
   }
   fprintf(my_ptrs->tubefilep, "\n");
}

void perfor_measures_tube_disc_givenR(Gauge_Conf const * const GC,
                                      Geometry const * const geo,
                                      int poly_dist,
                                      Tube_disc_ptrs const * const my_ptrs)
{
   const long space_len = geo->d_size[2];
   double two_pts_r = 0, two_pts_i = 0;
   double *three_pts_r = my_ptrs->three_pts_r, *three_pts_i = my_ptrs->three_pts_i;
   for (int y = 0; y < space_len; y++) {
      three_pts_i[y] = 0;
      three_pts_r[y] = 0;
   }

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+:three_pts_r[:space_len]) reduction(+:three_pts_i[:space_len]) reduction(+:two_pts_r) reduction(+:two_pts_i)
   #endif
   for (rsp = 0; rsp < geo->d_space_vol; rsp++) {
      // computing positions
      long rsp_other, r, r_mid;
      int t_tmp;
      r = sisp_and_t_to_si(geo, rsp, 0);
      for(int i = 0; i < poly_dist; i++) {
         r = nnp(geo, r, 1);
      }
      r_mid = r;
      r = nnp(geo, r, 1);
      for(int i = 0; i < poly_dist; i++) {
         r = nnp(geo, r, 1);
      }
      si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);

      // computing two points function
      double poly_prod_r = my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_r[rsp_other] +
                           my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_i[rsp_other];
      double poly_prod_i = my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_r[rsp_other] -
                           my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_i[rsp_other];
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;
      
      // loop on the y position of the plaquette
      GAUGE_GROUP matrix;
      double plaq_r, plaq_i;
      for (int y = 0; y < space_len; y++) {
         r = r_mid;
         plaq_r = 0;
         plaq_i = 0;
         for(int t = 0; t < geo->d_size[0]; t++) { // averaging on t position of plaq
            plaquettep_matrix(GC, geo, r, 0, 1, &matrix);
            plaq_r += retr(&matrix);
            plaq_i += imtr(&matrix);
            r = nnp(geo, r, 0);
         }
         plaq_r /= geo->d_size[0];
         plaq_i /= geo->d_size[0];

         three_pts_r[y] += poly_prod_r * plaq_r - poly_prod_i * plaq_i;
         three_pts_i[y] += poly_prod_i * plaq_r + poly_prod_r * plaq_i;

         r_mid = nnp(geo, r_mid, 2); // next y position of plaq
      }
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;

   fprintf(my_ptrs->tubefilep, "%ld %d ", GC->update_index, poly_dist);
   fprintf(my_ptrs->tubefilep, "%.12g %.12g ", two_pts_r, two_pts_i);

   for (int y = 0; y < space_len; y++) {
    three_pts_r[y] *= geo->d_inv_space_vol;
    three_pts_i[y] *= geo->d_inv_space_vol;

    fprintf(my_ptrs->tubefilep, "%.12g %.12g ", three_pts_r[y], three_pts_i[y]);
   }
   fprintf(my_ptrs->tubefilep, "\n");
}

void polyakov_corr_givenR(Geometry const * const geo,
                          Tube_disc_ptrs const * const my_ptrs,
                          int d)
{
   double two_pts_r = 0;
   double two_pts_i = 0;

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+: two_pts_r) reduction(+: two_pts_i)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r, rsp_other;
      int t_tmp;

      r = sisp_and_t_to_si(geo, rsp, 0);
      for (int i = 0; i < d; i++) r = nnp(geo, r, 1);

      si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);
      double poly_prod_r = my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_r[rsp_other] +
                           my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_i[rsp_other];
      double poly_prod_i = my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_r[rsp_other] -
                           my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_i[rsp_other];
      
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;

   fprintf(my_ptrs->polyfilep, "%.12g %.12g ", two_pts_r, two_pts_i);
}

void polyakov_corr_givenR_withAdj(Geometry const * const geo,
                                  Tube_disc_ptrs const * const my_ptrs,
                                  int d)
{
   double two_pts_r = 0.;
   double two_pts_i = 0.;
   double two_pts_adj = 0.;

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+: two_pts_r) reduction(+: two_pts_i) reduction(+: two_pts_adj)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r, rsp_other;
      int t_tmp;

      r = sisp_and_t_to_si(geo, rsp, 0);
      for (int i = 0; i < d; i++) r = nnp(geo, r, 1);

      si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);

      double poly_re_here = my_ptrs->aux_poly_r[rsp];
      double poly_im_here = my_ptrs->aux_poly_i[rsp];
      double poly_adj_here = (NCOLOR * NCOLOR) * poly_re_here * poly_re_here + poly_im_here * poly_im_here - 1.;

      double poly_re_other = my_ptrs->aux_poly_r[rsp_other];
      double poly_im_other = my_ptrs->aux_poly_i[rsp_other];
      double poly_adj_other = (NCOLOR * NCOLOR) * poly_re_other + poly_im_other * poly_im_other - 1.;
      
      double poly_prod_r = poly_re_here * poly_re_other + poly_im_here * poly_im_other;
      double poly_prod_i = poly_im_here * poly_re_other - poly_re_here * poly_re_other;

      double poly_prod_adj = poly_adj_here * poly_adj_other;
      
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;

      two_pts_adj += poly_prod_adj;
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;

   two_pts_adj *= geo->d_inv_space_vol;

   fprintf(my_ptrs->polyfilep, "%.12g %.12g ", two_pts_r, two_pts_i);
   fprintf(my_ptrs->padjfilep, "%.12g ", two_pts_adj);
}

void polycor_and_threepts_givenR(Geometry const * const geo,
                                 Tube_disc_ptrs const * const my_ptrs,
                                 int d)
{
   const long space_len = geo->d_size[1];
   double two_pts_r = 0;
   double two_pts_i = 0;

   for (int y = 0; y < space_len; y++){
      my_ptrs->three_pts_i[y] = 0;
      my_ptrs->three_pts_r[y] = 0;
   }

   const int d_halves = (d + 1) / 2;
   /* This is an oddity
   if d is odd, I want to go back (d - 1) / 2 + 1 steps, thus d_halves is right
   if d is even, I want to go back d / 2 steps
   For positive even d, (d + 1) / 2 should be d / 2 */

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+: two_pts_r) reduction(+: two_pts_i) reduction(+: my_ptrs->three_pts_r[:space_len]) reduction(+: my_ptrs->three_pts_i[:space_len])
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r, rsp_other;
      int t_tmp;

      r = sisp_and_t_to_si(geo, rsp, 0);
      for (int i = 0; i < d; i++) r = nnp(geo, r, 1);

      si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);
      double poly_prod_r = my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_r[rsp_other] +
                           my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_i[rsp_other];
      double poly_prod_i = my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_r[rsp_other] -
                           my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_i[rsp_other];
      
      for (int i = 0; i < d_halves; i++) r = nnm(geo, r, 1);
      long rsp_plaq;
      
      for (int y = 0; y < space_len; y++) {
         si_to_sisp_and_t(&rsp_plaq, &t_tmp, geo, r);
         my_ptrs->three_pts_r[y] += my_ptrs->plaq_column_r[rsp_plaq] * poly_prod_r -
                                    my_ptrs->plaq_column_i[rsp_plaq] * poly_prod_i;
         my_ptrs->three_pts_i[y] += my_ptrs->plaq_column_r[rsp_plaq] * poly_prod_i +
                                    my_ptrs->plaq_column_i[rsp_plaq] * poly_prod_r;

         r = nnp(geo, r, 2);
      }
      
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;
   fprintf(my_ptrs->polyfilep, "%.12g %.12g ", two_pts_r, two_pts_i);

   for (int y = 0; y < space_len; y++) {
      my_ptrs->three_pts_r[y] *= geo->d_inv_space_vol;
      my_ptrs->three_pts_i[y] *= geo->d_inv_space_vol;
      fprintf(my_ptrs->tubefilep, "%.12g %.12g ", my_ptrs->three_pts_r[y], my_ptrs->three_pts_i[y]);
   }
   fprintf(my_ptrs->tubefilep, "\n");
}


void perform_measures_two_and_three_pts(long update_index,
                                        Geometry const * const geo,
                                        GParam const * const param,
                                        Tube_disc_ptrs * my_ptrs)
{
   const int space_len = geo->d_size[1];

   fprintf(my_ptrs->polyfilep, "%ld ", update_index);
   for (int d = 0; d < space_len; d++) {
      if (d <= param->d_poly_dist_max) if (d >= param->d_poly_dist_min) {
         if ((d - param->d_poly_dist_min) % param->d_poly_dist_step == 0) {
            fprintf(my_ptrs->tubefilep, "%ld %d ", update_index, d);
            polycor_and_threepts_givenR(geo, my_ptrs, d);
            continue;
         }
      }
      if (d <= param->d_dist_poly)
         polyakov_corr_givenR(geo, my_ptrs, d);
   }
   fprintf(my_ptrs->polyfilep, "\n");

   fflush(my_ptrs->polyfilep);
   fflush(my_ptrs->tubefilep);
}

void perform_measures_two_pts(long update_index, Geometry const *const geo, GParam const *const param, Tube_disc_ptrs *my_ptrs)
{
   fprintf(my_ptrs->polyfilep, "%ld ", update_index);
   for (int d = 0; d <= param->d_dist_poly; d++) {
      polyakov_corr_givenR(geo, my_ptrs, d);
   }
   fprintf(my_ptrs->polyfilep, "\n");

   fflush(my_ptrs->polyfilep);
}

void perform_measures_two_pts_withAdj(long update_index,
                                      Geometry const *const geo,
                                      GParam const *const param,
                                      Tube_disc_ptrs *my_ptrs)
{
   fprintf(my_ptrs->polyfilep, "%ld ", update_index);
   fprintf(my_ptrs->padjfilep, "%ld ", update_index);
   for (int d = 0; d <= param->d_dist_poly; d++) {
      polyakov_corr_givenR_withAdj(geo, my_ptrs, d);
   }
   fprintf(my_ptrs->polyfilep, "\n");
   fprintf(my_ptrs->padjfilep, "\n");

   fflush(my_ptrs->polyfilep);
   fflush(my_ptrs->padjfilep);
}

#if POLY_CORR_FLY
void perform_measures_polycor(Gauge_Conf const *const GC,
                                                Geometry const *const geo,
                                                GParam const *const param,
                                                Tube_disc_ptrs *my_ptrs)
{
   const long space_len = geo->d_size[1];
   
   for (int d = 0; d < space_len; d++) {
      my_ptrs->two_pts_r[d] = 0;
      my_ptrs->two_pts_i[d] = 0;
   }

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+: my_ptrs->two_pts_r[:space_len]) reduction(+: my_ptrs->two_pts_i[:space_len])
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r, rsp_other;
      int t_tmp;

      r = sisp_and_t_to_si(geo, rsp, 0);

      for (int d = 0; d < space_len; d++) {
         si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);
         my_ptrs->two_pts_r[d] += my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_r[rsp_other] +
                                  my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_i[rsp_other];
         my_ptrs->two_pts_i[d] += my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_r[rsp_other] -
                                  my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_i[rsp_other];
      }

      r = nnp(geo, r, 1);
   }

   fprintf(my_ptrs->polyfilep, "%ld ", GC->update_index);
   for (int d = 0; d < space_len; d++) {
      my_ptrs->two_pts_r[d] *= geo->d_inv_space_vol;
      my_ptrs->two_pts_i[d] *= geo->d_inv_space_vol;

      fprintf(my_ptrs->polyfilep, "%.12g %.12g ", my_ptrs->two_pts_r[d], my_ptrs->two_pts_i[d]);
   }
   fprintf(my_ptrs->polyfilep, "\n");
   fflush(my_ptrs->polyfilep);
}
#endif

void perform_measures_plaqcor(long update_index,
                              Geometry const * const geo,
                              Tube_disc_ptrs const * const my_ptrs)
{
   const int dir = 2;
   const int space_len = geo->d_size[dir];

   for (int d = 0; d < space_len; d++){
      my_ptrs->plaq_corr_r[d] = 0;
      my_ptrs->plaq_corr_i[d] = 0;
   }

   long rsp;
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+: my_ptrs->plaq_corr_r[:space_len]) reduction(+: my_ptrs->plaq_corr_i[:space_len])
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r, rsp_other;
      int t_tmp;
      r = sisp_and_t_to_si(geo, rsp, 0);
      for (int d = 0; d < space_len; d++) {
         si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);
         my_ptrs->plaq_corr_r[d] += my_ptrs->plaq_column_r[rsp] * my_ptrs->plaq_column_r[rsp-rsp_other] +
                                    my_ptrs->plaq_column_i[rsp] * my_ptrs->plaq_column_i[rsp-rsp_other];
         my_ptrs->plaq_corr_i[d] += my_ptrs->plaq_column_i[rsp] * my_ptrs->plaq_column_r[rsp-rsp_other] -
                                    my_ptrs->plaq_column_r[rsp] * my_ptrs->plaq_column_i[rsp-rsp_other];
         r = nnp(geo, r, dir);
      }
   }

   fprintf(my_ptrs->plaqfilep, "%ld ", update_index);
   for (int d = 0; d < space_len; d++){
      my_ptrs->plaq_corr_r[d] *= geo->d_inv_space_vol;
      my_ptrs->plaq_corr_i[d] *= geo->d_inv_space_vol;
      fprintf(my_ptrs->plaqfilep, "%.12g %.12g ", my_ptrs->plaq_corr_r[d], my_ptrs->plaq_corr_i[d]);      
   }
   fprintf(my_ptrs->plaqfilep, "\n");
   fflush(my_ptrs->plaqfilep);
}

void perform_measures_tube_disc_highT(Gauge_Conf const * const GC,
                               Geometry const * const geo,
                               GParam const * const param,
                               Tube_disc_ptrs const * const my_ptrs)
{
   long rsp;
   // compute all the polyakov
   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp)
   #endif
   for (rsp=0; rsp<geo->d_space_vol; rsp++) {
      long r;
      GAUGE_GROUP matrix;

      r=sisp_and_t_to_si(geo, rsp, 0);

      one(&matrix);
      for(int t=0; t<geo->d_size[0]; t++)
         {
         times_equal(&matrix, &(GC->lattice[r][0]));
         r=nnp(geo, r, 0);
         }
      my_ptrs->aux_poly_r[rsp] = retr(&matrix);
      my_ptrs->aux_poly_i[rsp] = imtr(&matrix);
   }

   const long space_len = geo->d_size[2];
   double two_pts_r = 0, two_pts_i = 0;
   double *three_pts_r = my_ptrs->three_pts_r, *three_pts_i = my_ptrs->three_pts_i;
   for (int y = 0; y < space_len; y++) {
      three_pts_i[y] = 0;
      three_pts_r[y] = 0;
   }

   #ifdef OPENMP_MODE
   #pragma omp parallel for num_threads(NTHREADS) private(rsp) reduction(+:three_pts_r[:space_len]) reduction(+:three_pts_i[:space_len]) reduction(+:two_pts_r) reduction(+:two_pts_i)
   #endif
   for (rsp = 0; rsp < geo->d_space_vol; rsp++) {
      // computing positions
      long rsp_other, r, r_mid;
      int t_tmp;
      r = sisp_and_t_to_si(geo, rsp, 0);
      for(int i = 0; i < param->d_dist_poly; i++) {
         r = nnp(geo, r, 1);
      }
      r_mid = r;
      r = nnp(geo, r, 1);
      for(int i = 0; i < param->d_dist_poly; i++) {
         r = nnp(geo, r, 1);
      }
      si_to_sisp_and_t(&rsp_other, &t_tmp, geo, r);

      // computing two points function
      double poly_prod_r = my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_r[rsp_other] +
                           my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_i[rsp_other];
      double poly_prod_i = my_ptrs->aux_poly_i[rsp] * my_ptrs->aux_poly_r[rsp_other] -
                           my_ptrs->aux_poly_r[rsp] * my_ptrs->aux_poly_i[rsp_other];
      two_pts_r += poly_prod_r;
      two_pts_i += poly_prod_i;
      
      // loop on the y position of the plaquette
      GAUGE_GROUP matrix;
      double plaq_r, plaq_i;
      for (int y = 0; y < space_len; y++) {
         r = r_mid;
         plaq_r = 0;
         plaq_i = 0;
         for(int t = 0; t < geo->d_size[0]; t++) { // averaging on t position of plaq
            plaquettep_matrix(GC, geo, r, 0, 1, &matrix);
            plaq_r += retr(&matrix);
            plaq_i += imtr(&matrix);
            r = nnp(geo, r, 0);
         }
         plaq_r /= geo->d_size[0];
         plaq_i /= geo->d_size[0];

         three_pts_r[y] += poly_prod_r * plaq_r - poly_prod_i * plaq_i;
         three_pts_i[y] += poly_prod_i * plaq_r + poly_prod_r * plaq_i;

         r_mid = nnp(geo, r_mid, 2); // next y position of plaq
      }
   }

   two_pts_r *= geo->d_inv_space_vol;
   two_pts_i *= geo->d_inv_space_vol;

   fprintf(my_ptrs->tubefilep, "%.12g %.12g ", two_pts_r, two_pts_i);

   for (int y = 0; y < space_len; y++) {
    three_pts_r[y] *= geo->d_inv_space_vol;
    three_pts_i[y] *= geo->d_inv_space_vol;

    fprintf(my_ptrs->tubefilep, "%.12g %.12g ", three_pts_r[y], three_pts_i[y]);
   }
   fprintf(my_ptrs->tubefilep, "\n");
   fflush(my_ptrs->tubefilep);
}

void init_tube_disc_stuff(Geometry const *const geo, GParam const *const param, Tube_disc_ptrs *my_ptrs)
{
   // allocate aux_poly
    int err = posix_memalign((void **) &(my_ptrs->aux_poly_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->aux_poly_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err != 0) {
      fprintf(stderr, "Unable to allocate memory for Polyakov loops\n");
      exit(EXIT_FAILURE);
    }

   // allocate three pts
    err = posix_memalign((void **) &(my_ptrs->three_pts_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[2] * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->three_pts_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[2] * sizeof(double));
    if (err != 0) {
      fprintf(stderr, "Unable to allocate memory for Polyakov-Polyakov-plaquette correlators\n");
      exit(EXIT_FAILURE);
    }

#if POLY_CORR_FLY
   // allocate two pts
    err = posix_memalign((void **) &(my_ptrs->two_pts_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[1] * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->two_pts_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[1] * sizeof(double));
    if (err != 0) {
      fprintf(stderr, 'Unable to allocate memory for Polyakov-Polyakov correlators\n');
      exit(EXIT_FAILURE);
    }
#endif

    // allocate plaq column
    err = posix_memalign((void **) &(my_ptrs->plaq_column_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->plaq_column_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));    
    if (err != 0) {
      fprintf(stderr, "Unable to allocate memory for plaquettes columns\n");
      exit(EXIT_FAILURE);
    }

    // allocate plaq correlators
    err = posix_memalign((void **) &(my_ptrs->plaq_corr_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[1] * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->plaq_corr_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_size[1] * sizeof(double));
    if (err != 0) {
      fprintf(stderr, "Unable to allocate memory for plaquettes correlators\n");
      exit(EXIT_FAILURE);
    }

    init_tube_file(&(my_ptrs->tubefilep), param);
    init_poly_file(&(my_ptrs->polyfilep), param);
    init_plaq_file(&(my_ptrs->plaqfilep), param);

}

void init_with_adj(Geometry const * const geo, GParam const * const param, Tube_disc_ptrs *my_ptrs)
{
   // allocate aux_poly
    int err = posix_memalign((void **) &(my_ptrs->aux_poly_r), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err == 0) err = posix_memalign((void **) &(my_ptrs->aux_poly_i), (size_t) DOUBLE_ALIGN, (size_t) geo->d_space_vol * sizeof(double));
    if (err != 0) {
      fprintf(stderr, "Unable to allocate memory for Polyakov loops\n");
      exit(EXIT_FAILURE);
    }

    init_poly_file(&(my_ptrs->polyfilep), param);
    init_padj_file(&(my_ptrs->padjfilep), param);
}

void clean_tube_disc_stuff(Tube_disc_ptrs *my_ptrs)
{
    free(my_ptrs->aux_poly_r);
    free(my_ptrs->aux_poly_i);

    free(my_ptrs->three_pts_r);
    free(my_ptrs->three_pts_i);

#if POLY_CORR_FLY
    free(my_ptrs->two_pts_r);
    free(my_ptrs->two_pts_i);
#endif

    free(my_ptrs->plaq_column_r);
    free(my_ptrs->plaq_column_i);

    free(my_ptrs->plaq_corr_r);
    free(my_ptrs->plaq_corr_i);

    fclose(my_ptrs->tubefilep);
    fclose(my_ptrs->polyfilep);
    fclose(my_ptrs->plaqfilep);
}

void clean_with_adj(Tube_disc_ptrs *my_ptrs)
{
    free(my_ptrs->aux_poly_r);
    free(my_ptrs->aux_poly_i);

    fclose(my_ptrs->polyfilep);
    fclose(my_ptrs->padjfilep);
}

#endif