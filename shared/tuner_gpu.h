/* tuner.h
 * This file is the tuning function
 * - Precision conversion function
 * - Tuning policy 
 *
 * */

#ifndef _tuner_h
#define _tuner_h

#include <stdio.h>
#include <math.h>
#include <cstdio>
#include <fenv.h>
#include <string.h>
#include <stdlib.h>
#include <mpfr.h>
#include "my_lib.cu"

#include "CAMPARY/Doubles/src_gpu/multi_prec_certif.h"

#define CPR 4
#define PRE double
#define _CAM_G
#define _CMP_G
#define _NORM_G

// app specified 

// conversion func

__host__ __device__ void convertDToCMP(multi_prec<CPR> *dst, PRE *src, int size) {
  int i;
  for (i=0; i<size; i++)
    dst[i].setData(&src[i], 1); 
}

__host__ __device__ void convertCMPToD(PRE *dst, multi_prec<CPR> *src, int size) {
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

__host__ __device__ void convertCMPToMPF(mpf_t *dst, multi_prec<CPR> *src, int size) {
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

__host__ __device__ void CMP_init(multi_prec<CPR> *data, int size, PRE value) {
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

__host__ __device__ PRE CMP_unif(multi_prec<CPR> src) {
  int i,j;
  PRE retval;
  retval = 0.0;
  //qsort(src.getData(), CPR, sizeof(PRE), compareAscend);
  for (i=0; i<CPR; i++) {
    retval += (src.getData())[i];
  }
  return retval;
}


// operations & tune_down operations func
__host__ __device__ void exp_CMP(multi_prec<CPR> *dst, multi_prec<CPR> *src, int size) {
  double expv;
  int i;
  for (i=0; i<size; i++) { 
    convertCMPToD(&expv, &src[i], 1);
    expv = exp(expv);
    convertDToCMP(&dst[i], &expv, 1);
  }
}
// division /
__host__ __device__ void div_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *div_f, multi_prec<CPR> *div_s) {
  double tmp_d, div_0, div_1;
  convertCMPToD(&div_0, div_f, 1);
  convertCMPToD(&div_1, div_s, 1);
  tmp_d = div_0 / div_1;
  convertDToCMP(res, &tmp_d, 1);
}
__host__ __device__ void div_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *div_f, double cst_d) {
  double tmp_d, div_0, div_1;
  convertCMPToD(&div_0, div_f, 1);
  tmp_d = div_0 / cst_d;
  convertDToCMP(res, &tmp_d, 1);
}
__host__ __device__ void div_CMPToD(multi_prec<CPR> *res, double cst_d, multi_prec<CPR> *div_s) {
  double tmp_d, div_0, div_1;
  convertCMPToD(&div_1, div_s, 1);
  tmp_d = cst_d / div_1;
  convertDToCMP(res, &tmp_d, 1);
}
// multiplication *
__host__ __device__ void mul_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *mul_f, multi_prec<CPR> *mul_s) {
  double tmp_m, mul_0, mul_1;
  convertCMPToD(&mul_0, mul_f, 1);
  convertCMPToD(&mul_1, mul_s, 1);
  tmp_m = mul_0 * mul_1;
  convertDToCMP(res, &tmp_m, 1);
}
__host__ __device__ void mul_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *mul_f, double cst_d) {
  double tmp_m, mul_0, mul_1;
  convertCMPToD(&mul_0, mul_f, 1);
  tmp_m = mul_0 * cst_d;
  convertDToCMP(res, &tmp_m, 1);
}
// addition +/-
__host__ __device__ void add_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *add_f, multi_prec<CPR> *add_s) {
  double tmp_a, add_0, add_1;
  convertCMPToD(&add_0, add_f, 1);
  convertCMPToD(&add_1, add_s, 1);
  tmp_a = add_0 + add_1;
  convertDToCMP(res, &tmp_a, 1);
}
__host__ __device__ void add_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *add_f, double cst_d) {
  double tmp_a, add_0, add_1;
  convertCMPToD(&add_0, add_f, 1);
  tmp_a = add_0 + cst_d;
  convertDToCMP(res, &tmp_a, 1);
}

#endif
