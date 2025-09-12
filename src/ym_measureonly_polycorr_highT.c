#ifndef YM_TUBE_DISC_C
#define YM_TUBE_DISC_C

#include"../include/macro.h"

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#ifdef OPENMP_MODE
  #include<omp.h>
#endif

#include"../include/function_pointers.h"
#include"../include/gauge_conf.h"
#include"../include/geometry.h"
#include"../include/gparam.h"
#include"../include/random.h"
#include"../include/tube_disc_highT.h"

int read_conf_name(FILE* input, int max_len, char* buffer){
  for (int i = 0; i < max_len; i++){
    buffer[i] = (char) getc(input);
    if (buffer[i] == '\n') {
      buffer[i] = '\0';
      return 0;
    }
  }
  buffer[max_len - 1] = '\0';
  return 1;
}

void real_main(char *in_file)
{
    Gauge_Conf GC;
    Geometry geo;
    GParam param;

    int count;
    FILE *datafilep;
    time_t time1, time2;

    // to disable nested parallelism
    #ifdef OPENMP_MODE
      omp_set_nested(0);
    #endif

    // read input file
    readinput(in_file, &param);

    int tmp=param.d_sizeg[1];
    for(count=2; count<STDIM; count++)
       {
       if(tmp!= param.d_sizeg[count])
         {
         fprintf(stderr, "When using yang_mills_tube_disc all the spatial sizes have to be of equal length.\n");
         exit(EXIT_FAILURE);
         }
       }

    // initialize random generator
    initrand(param.d_randseed);

    // open data_file
    init_data_file(&datafilep, &param);

    // initialize geometry
    init_geometry(&geo, param.d_sizeg);

    FILE* conf_list = fopen(param.d_conf_file, "r");
    if (conf_list == NULL) {
      fprintf(stderr, "Configration names file not found, %s %d\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    Tube_disc_ptrs tube_ptr;
    init_tube_disc_stuff(&geo, &param, &tube_ptr);

    // montecarlo
    time(&time1);
    // count starts from 1 to avoid problems using 

    param.d_start = 2;
    int reading_err = 0;
    for (count = 0; count < param.d_thermal; count++)
    {
        reading_err = read_conf_name(conf_list, STD_STRING_LENGTH, param.d_conf_file);
        if (reading_err) {
          fprintf(stderr, "conf file name too long (len > STD_STRING_LENGTH = %d), %s %d",
          STD_STRING_LENGTH, __FILE__, __LINE__);
          exit(EXIT_FAILURE);
        }
    }

    for(count=1; count < param.d_sample + 1; count++)
    {
        // initialize gauge configuration
        reading_err = read_conf_name(conf_list, STD_STRING_LENGTH, param.d_conf_file);
        if (reading_err) {
          fprintf(stderr, "conf file name too long (len > STD_STRING_LENGTH = %d), %s %d",
          STD_STRING_LENGTH, __FILE__, __LINE__);
          exit(EXIT_FAILURE);
        }
        init_gauge_conf(&GC, &geo, &param);

        perform_measures_localobs(&GC, &geo, &param, datafilep, NULL);

        // preparation for the correlators
        compute_polyakov_tube_disc_highT(&GC, &geo, &tube_ptr);

        perform_measures_two_pts(GC.update_index, &geo, &param, &tube_ptr);

        // free gauge configuration
        free_gauge_conf(&GC, &geo);
    }

    time(&time2);
    // montecarlo end

    // close data file
    fclose(conf_list);
    fclose(datafilep);

    // save configuration
    print_parameters_tube_disc_highT(&param, time1, time2);

    clean_tube_disc_stuff(&tube_ptr);

    // free geometry
    free_geometry(&geo);
}


void print_template_input(void)
  {
  FILE *fp;

  fp=fopen("template_input.in", "w");

  if(fp==NULL)
    {
    fprintf(stderr, "Error in opening the file template_input.in (%s, %d)\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
    }
  else
    {
    fprintf(fp, "size 4 4 4 4\n");
    fprintf(fp,"\n");
    fprintf(fp, "beta 5.705\n");
    fprintf(fp, "theta 1.5\n");
    fprintf(fp,"\n");
    fprintf(fp, "sample    10   # confs to read \n");
    fprintf(fp, "thermal   0    # confs to skip \n");
    fprintf(fp,"\n");
    fprintf(fp, "#for Polyakov corrrelators\n");
    fprintf(fp, "dist_poly               2  # maximum distance between the Polyakov loops\n");
    fprintf(fp, "\n");
    fprintf(fp, "coolsteps  0     # number of cooling steps to be used\n");
    fprintf(fp, "coolrepeat 0     # number of times 'coolsteps' are repeated\n");
    fprintf(fp, "\n");
    fprintf(fp, "#output files\n");
    fprintf(fp, "conf_file  conf.dat  # file with list of confs to read newline separated\n");
    fprintf(fp, "data_file  dati.dat\n");
    fprintf(fp, "tube_file  tube.dat\n");
    fprintf(fp, "poly_file  poly_corr.dat\n");
    fprintf(fp, "plaq_file  plaq_corr.dat\n");
    fprintf(fp, "log_file   log.dat\n");
    fprintf(fp, "\n");
    fprintf(fp, "randseed 0    #(0=time)\n");
    fclose(fp);
    }
  }


int main (int argc, char **argv)
    {
    char in_file[50];

    if(argc != 2)
      {
      printf("\nPackage %s version %s\n", PACKAGE_NAME, PACKAGE_VERSION);
      printf("Claudio Bonati %s\n", PACKAGE_BUGREPORT);
      printf("Usage: %s input_file\n\n", argv[0]);

      printf("Compilation details:\n");
      printf("\tN_c (number of colors): %d\n", NCOLOR);
      printf("\tST_dim (space-time dimensionality): %d\n", STDIM);
      printf("\n");
      printf("\tINT_ALIGN: %s\n", QUOTEME(INT_ALIGN));
      printf("\tDOUBLE_ALIGN: %s\n", QUOTEME(DOUBLE_ALIGN));

      #ifdef DEBUG
        printf("\n\tDEBUG mode\n");
      #endif

      #ifdef OPENMP_MODE
        printf("\n\tusing OpenMP with %d threads\n", NTHREADS);
      #endif

      #ifdef THETA_MODE
        printf("\n\tusing imaginary theta\n");
      #endif


      printf("\n");

      #ifdef __INTEL_COMPILER
        printf("\tcompiled with icc\n");
      #elif defined(__clang__)
        printf("\tcompiled with clang\n");
      #elif defined( __GNUC__ )
        printf("\tcompiled with gcc version: %d.%d.%d\n",
                __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
      #endif

      print_template_input();

      return EXIT_SUCCESS;
      }
    else
      {
      if(strlen(argv[1]) >= STD_STRING_LENGTH)
        {
        fprintf(stderr, "File name too long. Increse STD_STRING_LENGTH in include/macro.h\n");
        }
      else
        {
        strcpy(in_file, argv[1]);
        }
      }

    real_main(in_file);

    return EXIT_SUCCESS;
    }

#endif

