/* tuner.h
 * This file is the tuning function
 * - Precision conversion function
 * - Tuning policy 
 *
 * */

#ifndef _tuner_cpu_h
#define _tuner_cpu_h

#include <stdio.h>
#include <cstdio>
#include <math.h>
#include <fenv.h>
#include <string.h>
#include <stdlib.h>
#include <mpfr.h>
// #include "my_lib.cu"

#include "CAMPARY/Doubles/src_cpu/multi_prec_certif.h"

// #define CPR 4
//   NOTE: CPR is commented out because we are specifying during
//   the make command. For example:
//   $ make CPP_FLAG="-g -Wall -DCPR=4"
//   [[DEFINING CPR IS CRITICAL TO USE THIS CAMPARY]]

#define PRE double
#define _CAM_G
#define _CMP_G
#define _NORM_G


// app specific functions
//   (removed)




// conversion functions

void convertDToCMP(multi_prec<CPR> *dst, PRE *src, int size) {
  int i;
  for (i=0; i<size; i++)
    dst[i].setData(&src[i], 1); 
}

void convertCMPToD(PRE *dst, multi_prec<CPR> *src, int size) {
  int i,j;   
  PRE tmp;
  //qsort(src.getData(), CPR, sizeof(PRE), compareAscend);
  for (i=0; i<size; i++) {
    tmp = 0.0;
    for (j=0; j<CPR; j++) {
       tmp += (src[i].getData())[j]; 
    } 
    dst[i] = tmp;
  }
}

void convertCMPToMPF(mpf_t *dst, multi_prec<CPR> *src, int size) {
  //convert all to mpf_t
  mpf_t tmp;
  mpf_init2(tmp, 256);
  for (int i=0; i<size; i++) {
    mpf_set_d(dst[i], 0.0);
    const double * c_d = src[i].getData();
    for (int j=0; j<CPR; j++) {
      mpf_set_d(tmp, c_d[j]);
      mpf_add(dst[i], dst[i], tmp);
    }
  }
}

void convertScalarCMPToMPF(mpf_t &dst, multi_prec<CPR> &src) {
  //convert single scalar CAMPARY value to GMP
  
  //  **Usage warning: You must make sure the mpf_t result "dst" has 
  //  been intialized in the location where this function is being
  //  called from. For example, call the function like this:
  /*
        // ... somewhere in main ... 
        mpf_t x;
        mpf_init2(x, 256);    // THIS INIT IS CRITICAL!!
        // ...
        convertScalarCMPToMPF(x, ...);
  */
  mpf_t tmp;
  mpf_init2(tmp, 256);
  mpf_set_d(dst, 0.0);
  const double * c_d = src.getData();
  for (int j=0; j<CPR; j++)
  {
    mpf_set_d(tmp, c_d[j]);
    mpf_add(dst, dst, tmp);
  }
}

void CMP_init(multi_prec<CPR> *data, int size, PRE value) {
  int i, j;
  for (i=0; i<size; i++) {
    for (j=0; j<CPR; j++) {
      // CMP: data[0] is the initial value, all other vars for data[1-n] is 0
      if (!j)
        data[i].setElement(value, j);
      else
        data[i].setElement(0.0, j);
    }
  }
}

void initFromDouble(multi_prec<CPR> *x, double value) {
  x->setElement(value, 0);
  for (int k = 1; k < CPR; k++)
    x->setElement(0.0, k);
}

void init_vector(multi_prec<CPR> *a, int n)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < CPR; j++)
    {
      a[i].setElement(0.0, j);
    }
  }
}

PRE CMP_unif(multi_prec<CPR> src) {
  int i;
  PRE retval;
  retval = 0.0;
  //qsort(src.getData(), CPR, sizeof(PRE), compareAscend);
  for (i=0; i<CPR; i++) {
    retval += (src.getData())[i];
  }
  return retval;
}

// operations & tune_down operations func
void exp_CMP(multi_prec<CPR> *dst, multi_prec<CPR> *src, int size) {
  double expv;
  int i;
  for (i=0; i<size; i++) { 
    convertCMPToD(&expv, &src[i], 1);
    expv = exp(expv);
    convertDToCMP(&dst[i], &expv, 1);
  }
}
// division /
void div_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *div_f, multi_prec<CPR> *div_s) {
  double tmp_d, div_0, div_1;
  convertCMPToD(&div_0, div_f, 1);
  convertCMPToD(&div_1, div_s, 1);
  tmp_d = div_0 / div_1;
  convertDToCMP(res, &tmp_d, 1);
}
void div_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *div_f, double cst_d) {
  double tmp_d, div_0;//, div_1;
  convertCMPToD(&div_0, div_f, 1);
  tmp_d = div_0 / cst_d;
  convertDToCMP(res, &tmp_d, 1);
}
void div_CMPToD(multi_prec<CPR> *res, double cst_d, multi_prec<CPR> *div_s) {
  double tmp_d, div_1;//, div_0;
  convertCMPToD(&div_1, div_s, 1);
  tmp_d = cst_d / div_1;
  convertDToCMP(res, &tmp_d, 1);
}
// multiplication *
void mul_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *mul_f, multi_prec<CPR> *mul_s) {
  double tmp_m, mul_0, mul_1;
  convertCMPToD(&mul_0, mul_f, 1);
  convertCMPToD(&mul_1, mul_s, 1);
  tmp_m = mul_0 * mul_1;
  convertDToCMP(res, &tmp_m, 1);
}
void mul_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *mul_f, double cst_d) {
  double tmp_m, mul_0;//, mul_1;
  convertCMPToD(&mul_0, mul_f, 1);
  tmp_m = mul_0 * cst_d;
  convertDToCMP(res, &tmp_m, 1);
}
// addition +/-
void add_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *add_f, multi_prec<CPR> *add_s) {
  double tmp_a, add_0, add_1;
  convertCMPToD(&add_0, add_f, 1);
  convertCMPToD(&add_1, add_s, 1);
  tmp_a = add_0 + add_1;
  convertDToCMP(res, &tmp_a, 1);
}
void add_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *add_f, double cst_d) {
  double tmp_a, add_0;//, add_1;
  convertCMPToD(&add_0, add_f, 1);
  tmp_a = add_0 + cst_d;
  convertDToCMP(res, &tmp_a, 1);
}
// square root 
//   example: res = std::sqrt(inVal);
//        ==> sqrt_CMPToD(&res, &inVal);
void sqrt_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *inVal) {
  double tmp, value;
  convertCMPToD(&value, inVal, 1);
  tmp = std::sqrt(value);
  convertDToCMP(res, &tmp, 1);
}

#endif
