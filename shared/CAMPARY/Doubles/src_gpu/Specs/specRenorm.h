/*
 * specRenorm.h
 *
 * This file is part of CAMPARY Library
 *
 * Copyright (C) 2014 - 
 *
 * CAMPARY Library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * CAMPARY Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MultiPrecGPU Library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, 
 * Boston, MA  02110-1301  USA
 */
 
/* Contributors: Valentina Popescu valentina.popescu@ens-lyon.fr
 *               Mioara Joldes joldes@laas.fr
 */

#ifndef _specRenorm_h
#define _specRenorm_h

/****************************************************/
template<> __host__ __device__ __forceinline__ void fast_VecSum<1>(double *x){ }
template<> __host__ __device__ __forceinline__ void fast_VecSum<1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void fast_VecSum<2>(double *x){ x[0] = fast_two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_VecSum<2>(const double *x, double *r){ r[0] = fast_two_sum(x[0], x[1], r[1]); }

/****************************************************/
template<> __host__ __device__ __forceinline__ void VecSum<1>(double *x){ }
template<> __host__ __device__ __forceinline__ void VecSum<1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void VecSum<2>(double *x){ x[0] = two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void VecSum<2>(const double *x, double *r){ r[0] = two_sum(x[0], x[1], r[1]); }

/****************************************************/
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<1,1>(double *x){ }
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<1,1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<2,1>(double *x){ x[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<2,1>(const double *x, double *r){ r[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<2,2>(double *x){ x[0] = fast_two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_VecSumErrBranch<2,2>(const double *x, double *r){ r[0] = fast_two_sum(x[0], x[1], r[1]); }

/****************************************************/
template<> __host__ __device__ __forceinline__ void fast_VecSumErr<1>(double *x){ }
template<> __host__ __device__ __forceinline__ void fast_VecSumErr<1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void fast_VecSumErr<2>(double *x){ x[0] = fast_two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_VecSumErr<2>(const double *x, double *r){ r[0] = fast_two_sum(x[0], x[1], r[1]); }

/****************************************************/
template<> __host__ __device__ __forceinline__ void fast_renorm2L<1,1>(double *x){ }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<1,1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<2,1>(double *x){ x[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<2,1>(const double *x, double *r){ r[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<2,2>(double *x){ x[0] = fast_two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<2,2>(const double *x, double *r){ r[0] = fast_two_sum(x[0], x[1], r[1]); }
template<> __host__ __device__ __forceinline__ void fast_renorm2L<3,2>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[1], x[2], x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], x[ptr+1]);

	for(int i=ptr+1; i<2; i++) x[i] = 0.;
}
template<> __host__ __device__ __forceinline__ void fast_renorm2L<3,2>(const double *x, double *r){
	double f[3];
	int ptr=0;
	double pr = fast_two_sum(x[1], x[2], f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=2; ptr<2 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<2 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<2; i++) r[i] = 0.;
}
template<> __host__ __device__ __forceinline__ void fast_renorm2L<4,3>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], x[ptr+1]);

	for(int i=ptr+1; i<3; i++) x[i] = 0.;
}

template<> __host__ __device__ __forceinline__ void fast_renorm2L<4,3>(const double *x, double *r){
	double f[4];
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<4; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}
template<> __host__ __device__ __forceinline__ void fast_renorm2L<5,4>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[3], x[4], x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], x[ptr+1]);

	for(int i=ptr+1; i<4; i++) x[i] = 0.;
}

template<> __host__ __device__ __forceinline__ void fast_renorm2L<5,4>(const double *x, double *r){
	double f[5];
	int ptr=0;
	double pr = fast_two_sum(x[3], x[4], f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=4; ptr<4 && i<5; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<4 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<4; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<6,5>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[4], x[5], x[5]);
	pr = fast_two_sum(x[3], pr, x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[5], x[ptr+1]);

	for(int i=ptr+1; i<5; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<6,5>(const double *x, double *r){
	double f[6];
	int ptr=0;
	double pr = fast_two_sum(x[4], x[5], f[5]);
	pr = fast_two_sum(x[3], pr, f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=5; ptr<5 && i<6; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<5 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<5; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<7,6>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[5], x[6], x[6]);
	pr = fast_two_sum(x[4], pr, x[5]);
	pr = fast_two_sum(x[3], pr, x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[5], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[6], x[ptr+1]);

	for(int i=ptr+1; i<6; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<7,6>(const double *x, double *r){
	double f[7];
	int ptr=0;
	double pr = fast_two_sum(x[5], x[6], f[6]);
	pr = fast_two_sum(x[4], pr, f[5]);
	pr = fast_two_sum(x[3], pr, f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=6; ptr<6 && i<7; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<6 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<6; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<8,7>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[6], x[7], x[7]);
	pr = fast_two_sum(x[5], pr, x[6]);
	pr = fast_two_sum(x[4], pr, x[5]);
	pr = fast_two_sum(x[3], pr, x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[5], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[6], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[7], x[ptr+1]);

	for(int i=ptr+1; i<7; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<8,7>(const double *x, double *r){
	double f[8];
	int ptr=0;
	double pr = fast_two_sum(x[6], x[7], f[7]);
	pr = fast_two_sum(x[5], pr, f[6]);
	pr = fast_two_sum(x[4], pr, f[5]);
	pr = fast_two_sum(x[3], pr, f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=7; ptr<7 && i<8; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<7 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<7; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<9,8>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[7], x[8], x[8]);
	pr = fast_two_sum(x[6], pr, x[7]);
	pr = fast_two_sum(x[5], pr, x[6]);
	pr = fast_two_sum(x[4], pr, x[5]);
	pr = fast_two_sum(x[3], pr, x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[5], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[6], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[7], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[8], x[ptr+1]);

	for(int i=ptr+1; i<8; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<9,8>(const double *x, double *r){
	double f[9];
	int ptr=0;
	double pr = fast_two_sum(x[7], x[8], f[8]);
	pr = fast_two_sum(x[6], pr, f[7]);
	pr = fast_two_sum(x[5], pr, f[6]);
	pr = fast_two_sum(x[4], pr, f[5]);
	pr = fast_two_sum(x[3], pr, f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[7], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=8; ptr<8 && i<9; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<8 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<8; i++) r[i] = 0.;
}

/****************************************************/
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<2,2>(const double *x, const double y, double *r){
	double f[3];
	int ptr=0;
	double pr = two_sum(x[1], y, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=2; ptr<2 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<2 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<2; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<3,3>(const double *x, const double y, double *r){
	double f[4];
	int ptr=0;
	double pr = two_sum(x[2], y, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<4; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<4,4>(const double *x, const double y, double *r){
	double f[5];
	int ptr=0;
	double pr = two_sum(x[3], y, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=4; ptr<4 && i<5; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<4 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<4; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<5,5>(const double *x, const double y, double *r){
	double f[6];
	int ptr=0;
	double pr = two_sum(x[4], y, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=5; ptr<5 && i<6; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<5 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<5; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<6,6>(const double *x, const double y, double *r){
	double f[7];
	int ptr=0;
	double pr = two_sum(x[5], y, f[6]);
	pr = two_sum(x[4], pr, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=6; ptr<6 && i<7; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<6 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<6; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<7,7>(const double *x, const double y, double *r){
	double f[8];
	int ptr=0;
	double pr = two_sum(x[6], y, f[7]);
	pr = two_sum(x[5], pr, f[6]);
	pr = two_sum(x[4], pr, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=7; ptr<7 && i<8; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<7 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<7; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<8,8>(const double *x, const double y, double *r){
	double f[9];
	int ptr=0;
	double pr = two_sum(x[7], y, f[8]);
	pr = two_sum(x[6], pr, f[7]);
	pr = two_sum(x[5], pr, f[6]);
	pr = two_sum(x[4], pr, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[7], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=8; ptr<8 && i<9; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<8 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<8; i++) r[i] = 0.;
}

/****************************************************/
template<> __host__ __device__ __forceinline__ void renorm2L<1,1>(double *x){ }
template<> __host__ __device__ __forceinline__ void renorm2L<1,1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void renorm2L<2,1>(double *x){ x[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm2L<2,1>(const double *x, double *r){ r[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm2L<2,2>(double *x){ x[0] = two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm2L<2,2>(const double *x, double *r){ r[0] = two_sum(x[0], x[1], r[1]); }

/****************************************************/
template<> __host__ __device__ __forceinline__ void renorm_rand2L<1,1>(double *x){ }
template<> __host__ __device__ __forceinline__ void renorm_rand2L<1,1>(const double *x, double *r){ r[0] = x[0]; }
template<> __host__ __device__ __forceinline__ void renorm_rand2L<2,1>(double *x){ x[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm_rand2L<2,1>(const double *x, double *r){ r[0] = FPadd_rn(x[0], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm_rand2L<2,2>(double *x){ x[0] = two_sum(x[0], x[1], x[1]); }
template<> __host__ __device__ __forceinline__ void renorm_rand2L<2,2>(const double *x, double *r){ r[0] = two_sum(x[0], x[1], r[1]); }

#endif

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], x[0]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[0], x[ptr+1]);

	for(int i=ptr+1; i<0; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(const double *x, double *r){
	double f[1];
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], f[0]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<1; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], x[0]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[0], x[ptr+1]);

	for(int i=ptr+1; i<0; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(const double *x, double *r){
	double f[1];
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], f[0]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<1; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#endif
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<0,0>(const double *x, const double y, double *r){
#if 0
	double f[1];
#else
	double f[2];
#endif
	int ptr=0;
	double pr = two_sum(x[-1], y, f[0]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<1; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], x[0]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[0], x[ptr+1]);

	for(int i=ptr+1; i<0; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(const double *x, double *r){
	double f[1];
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], f[0]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<1; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#endif
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<123,0>(const double *x, const double y, double *r){
#if 0
	double f[124];
#else
	double f[468];
#endif
	int ptr=0;
	double pr = two_sum(x[122], y, f[123]);
	pr = two_sum(x[121], pr, f[122]);
	pr = two_sum(x[120], pr, f[121]);
	pr = two_sum(x[119], pr, f[120]);
	pr = two_sum(x[118], pr, f[119]);
	pr = two_sum(x[117], pr, f[118]);
	pr = two_sum(x[116], pr, f[117]);
	pr = two_sum(x[115], pr, f[116]);
	pr = two_sum(x[114], pr, f[115]);
	pr = two_sum(x[113], pr, f[114]);
	pr = two_sum(x[112], pr, f[113]);
	pr = two_sum(x[111], pr, f[112]);
	pr = two_sum(x[110], pr, f[111]);
	pr = two_sum(x[109], pr, f[110]);
	pr = two_sum(x[108], pr, f[109]);
	pr = two_sum(x[107], pr, f[108]);
	pr = two_sum(x[106], pr, f[107]);
	pr = two_sum(x[105], pr, f[106]);
	pr = two_sum(x[104], pr, f[105]);
	pr = two_sum(x[103], pr, f[104]);
	pr = two_sum(x[102], pr, f[103]);
	pr = two_sum(x[101], pr, f[102]);
	pr = two_sum(x[100], pr, f[101]);
	pr = two_sum(x[99], pr, f[100]);
	pr = two_sum(x[98], pr, f[99]);
	pr = two_sum(x[97], pr, f[98]);
	pr = two_sum(x[96], pr, f[97]);
	pr = two_sum(x[95], pr, f[96]);
	pr = two_sum(x[94], pr, f[95]);
	pr = two_sum(x[93], pr, f[94]);
	pr = two_sum(x[92], pr, f[93]);
	pr = two_sum(x[91], pr, f[92]);
	pr = two_sum(x[90], pr, f[91]);
	pr = two_sum(x[89], pr, f[90]);
	pr = two_sum(x[88], pr, f[89]);
	pr = two_sum(x[87], pr, f[88]);
	pr = two_sum(x[86], pr, f[87]);
	pr = two_sum(x[85], pr, f[86]);
	pr = two_sum(x[84], pr, f[85]);
	pr = two_sum(x[83], pr, f[84]);
	pr = two_sum(x[82], pr, f[83]);
	pr = two_sum(x[81], pr, f[82]);
	pr = two_sum(x[80], pr, f[81]);
	pr = two_sum(x[79], pr, f[80]);
	pr = two_sum(x[78], pr, f[79]);
	pr = two_sum(x[77], pr, f[78]);
	pr = two_sum(x[76], pr, f[77]);
	pr = two_sum(x[75], pr, f[76]);
	pr = two_sum(x[74], pr, f[75]);
	pr = two_sum(x[73], pr, f[74]);
	pr = two_sum(x[72], pr, f[73]);
	pr = two_sum(x[71], pr, f[72]);
	pr = two_sum(x[70], pr, f[71]);
	pr = two_sum(x[69], pr, f[70]);
	pr = two_sum(x[68], pr, f[69]);
	pr = two_sum(x[67], pr, f[68]);
	pr = two_sum(x[66], pr, f[67]);
	pr = two_sum(x[65], pr, f[66]);
	pr = two_sum(x[64], pr, f[65]);
	pr = two_sum(x[63], pr, f[64]);
	pr = two_sum(x[62], pr, f[63]);
	pr = two_sum(x[61], pr, f[62]);
	pr = two_sum(x[60], pr, f[61]);
	pr = two_sum(x[59], pr, f[60]);
	pr = two_sum(x[58], pr, f[59]);
	pr = two_sum(x[57], pr, f[58]);
	pr = two_sum(x[56], pr, f[57]);
	pr = two_sum(x[55], pr, f[56]);
	pr = two_sum(x[54], pr, f[55]);
	pr = two_sum(x[53], pr, f[54]);
	pr = two_sum(x[52], pr, f[53]);
	pr = two_sum(x[51], pr, f[52]);
	pr = two_sum(x[50], pr, f[51]);
	pr = two_sum(x[49], pr, f[50]);
	pr = two_sum(x[48], pr, f[49]);
	pr = two_sum(x[47], pr, f[48]);
	pr = two_sum(x[46], pr, f[47]);
	pr = two_sum(x[45], pr, f[46]);
	pr = two_sum(x[44], pr, f[45]);
	pr = two_sum(x[43], pr, f[44]);
	pr = two_sum(x[42], pr, f[43]);
	pr = two_sum(x[41], pr, f[42]);
	pr = two_sum(x[40], pr, f[41]);
	pr = two_sum(x[39], pr, f[40]);
	pr = two_sum(x[38], pr, f[39]);
	pr = two_sum(x[37], pr, f[38]);
	pr = two_sum(x[36], pr, f[37]);
	pr = two_sum(x[35], pr, f[36]);
	pr = two_sum(x[34], pr, f[35]);
	pr = two_sum(x[33], pr, f[34]);
	pr = two_sum(x[32], pr, f[33]);
	pr = two_sum(x[31], pr, f[32]);
	pr = two_sum(x[30], pr, f[31]);
	pr = two_sum(x[29], pr, f[30]);
	pr = two_sum(x[28], pr, f[29]);
	pr = two_sum(x[27], pr, f[28]);
	pr = two_sum(x[26], pr, f[27]);
	pr = two_sum(x[25], pr, f[26]);
	pr = two_sum(x[24], pr, f[25]);
	pr = two_sum(x[23], pr, f[24]);
	pr = two_sum(x[22], pr, f[23]);
	pr = two_sum(x[21], pr, f[22]);
	pr = two_sum(x[20], pr, f[21]);
	pr = two_sum(x[19], pr, f[20]);
	pr = two_sum(x[18], pr, f[19]);
	pr = two_sum(x[17], pr, f[18]);
	pr = two_sum(x[16], pr, f[17]);
	pr = two_sum(x[15], pr, f[16]);
	pr = two_sum(x[14], pr, f[15]);
	pr = two_sum(x[13], pr, f[14]);
	pr = two_sum(x[12], pr, f[13]);
	pr = two_sum(x[11], pr, f[12]);
	pr = two_sum(x[10], pr, f[11]);
	pr = two_sum(x[9], pr, f[10]);
	pr = two_sum(x[8], pr, f[9]);
	pr = two_sum(x[7], pr, f[8]);
	pr = two_sum(x[6], pr, f[7]);
	pr = two_sum(x[5], pr, f[6]);
	pr = two_sum(x[4], pr, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<124; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], x[0]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[0], x[ptr+1]);

	for(int i=ptr+1; i<0; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<1,0>(const double *x, double *r){
	double f[1];
	int ptr=0;
	double pr = fast_two_sum(x[-1], x[0], f[0]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=0; ptr<0 && i<1; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<0 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<0; i++) r[i] = 0.;
}
#endif
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<123,467>(const double *x, const double y, double *r){
#if 0
	double f[124];
#else
	double f[467];
#endif
	int ptr=0;
	double pr = two_sum(x[122], y, f[123]);
	pr = two_sum(x[121], pr, f[122]);
	pr = two_sum(x[120], pr, f[121]);
	pr = two_sum(x[119], pr, f[120]);
	pr = two_sum(x[118], pr, f[119]);
	pr = two_sum(x[117], pr, f[118]);
	pr = two_sum(x[116], pr, f[117]);
	pr = two_sum(x[115], pr, f[116]);
	pr = two_sum(x[114], pr, f[115]);
	pr = two_sum(x[113], pr, f[114]);
	pr = two_sum(x[112], pr, f[113]);
	pr = two_sum(x[111], pr, f[112]);
	pr = two_sum(x[110], pr, f[111]);
	pr = two_sum(x[109], pr, f[110]);
	pr = two_sum(x[108], pr, f[109]);
	pr = two_sum(x[107], pr, f[108]);
	pr = two_sum(x[106], pr, f[107]);
	pr = two_sum(x[105], pr, f[106]);
	pr = two_sum(x[104], pr, f[105]);
	pr = two_sum(x[103], pr, f[104]);
	pr = two_sum(x[102], pr, f[103]);
	pr = two_sum(x[101], pr, f[102]);
	pr = two_sum(x[100], pr, f[101]);
	pr = two_sum(x[99], pr, f[100]);
	pr = two_sum(x[98], pr, f[99]);
	pr = two_sum(x[97], pr, f[98]);
	pr = two_sum(x[96], pr, f[97]);
	pr = two_sum(x[95], pr, f[96]);
	pr = two_sum(x[94], pr, f[95]);
	pr = two_sum(x[93], pr, f[94]);
	pr = two_sum(x[92], pr, f[93]);
	pr = two_sum(x[91], pr, f[92]);
	pr = two_sum(x[90], pr, f[91]);
	pr = two_sum(x[89], pr, f[90]);
	pr = two_sum(x[88], pr, f[89]);
	pr = two_sum(x[87], pr, f[88]);
	pr = two_sum(x[86], pr, f[87]);
	pr = two_sum(x[85], pr, f[86]);
	pr = two_sum(x[84], pr, f[85]);
	pr = two_sum(x[83], pr, f[84]);
	pr = two_sum(x[82], pr, f[83]);
	pr = two_sum(x[81], pr, f[82]);
	pr = two_sum(x[80], pr, f[81]);
	pr = two_sum(x[79], pr, f[80]);
	pr = two_sum(x[78], pr, f[79]);
	pr = two_sum(x[77], pr, f[78]);
	pr = two_sum(x[76], pr, f[77]);
	pr = two_sum(x[75], pr, f[76]);
	pr = two_sum(x[74], pr, f[75]);
	pr = two_sum(x[73], pr, f[74]);
	pr = two_sum(x[72], pr, f[73]);
	pr = two_sum(x[71], pr, f[72]);
	pr = two_sum(x[70], pr, f[71]);
	pr = two_sum(x[69], pr, f[70]);
	pr = two_sum(x[68], pr, f[69]);
	pr = two_sum(x[67], pr, f[68]);
	pr = two_sum(x[66], pr, f[67]);
	pr = two_sum(x[65], pr, f[66]);
	pr = two_sum(x[64], pr, f[65]);
	pr = two_sum(x[63], pr, f[64]);
	pr = two_sum(x[62], pr, f[63]);
	pr = two_sum(x[61], pr, f[62]);
	pr = two_sum(x[60], pr, f[61]);
	pr = two_sum(x[59], pr, f[60]);
	pr = two_sum(x[58], pr, f[59]);
	pr = two_sum(x[57], pr, f[58]);
	pr = two_sum(x[56], pr, f[57]);
	pr = two_sum(x[55], pr, f[56]);
	pr = two_sum(x[54], pr, f[55]);
	pr = two_sum(x[53], pr, f[54]);
	pr = two_sum(x[52], pr, f[53]);
	pr = two_sum(x[51], pr, f[52]);
	pr = two_sum(x[50], pr, f[51]);
	pr = two_sum(x[49], pr, f[50]);
	pr = two_sum(x[48], pr, f[49]);
	pr = two_sum(x[47], pr, f[48]);
	pr = two_sum(x[46], pr, f[47]);
	pr = two_sum(x[45], pr, f[46]);
	pr = two_sum(x[44], pr, f[45]);
	pr = two_sum(x[43], pr, f[44]);
	pr = two_sum(x[42], pr, f[43]);
	pr = two_sum(x[41], pr, f[42]);
	pr = two_sum(x[40], pr, f[41]);
	pr = two_sum(x[39], pr, f[40]);
	pr = two_sum(x[38], pr, f[39]);
	pr = two_sum(x[37], pr, f[38]);
	pr = two_sum(x[36], pr, f[37]);
	pr = two_sum(x[35], pr, f[36]);
	pr = two_sum(x[34], pr, f[35]);
	pr = two_sum(x[33], pr, f[34]);
	pr = two_sum(x[32], pr, f[33]);
	pr = two_sum(x[31], pr, f[32]);
	pr = two_sum(x[30], pr, f[31]);
	pr = two_sum(x[29], pr, f[30]);
	pr = two_sum(x[28], pr, f[29]);
	pr = two_sum(x[27], pr, f[28]);
	pr = two_sum(x[26], pr, f[27]);
	pr = two_sum(x[25], pr, f[26]);
	pr = two_sum(x[24], pr, f[25]);
	pr = two_sum(x[23], pr, f[24]);
	pr = two_sum(x[22], pr, f[23]);
	pr = two_sum(x[21], pr, f[22]);
	pr = two_sum(x[20], pr, f[21]);
	pr = two_sum(x[19], pr, f[20]);
	pr = two_sum(x[18], pr, f[19]);
	pr = two_sum(x[17], pr, f[18]);
	pr = two_sum(x[16], pr, f[17]);
	pr = two_sum(x[15], pr, f[16]);
	pr = two_sum(x[14], pr, f[15]);
	pr = two_sum(x[13], pr, f[14]);
	pr = two_sum(x[12], pr, f[13]);
	pr = two_sum(x[11], pr, f[12]);
	pr = two_sum(x[10], pr, f[11]);
	pr = two_sum(x[9], pr, f[10]);
	pr = two_sum(x[8], pr, f[9]);
	pr = two_sum(x[7], pr, f[8]);
	pr = two_sum(x[6], pr, f[7]);
	pr = two_sum(x[5], pr, f[6]);
	pr = two_sum(x[4], pr, f[5]);
	pr = two_sum(x[3], pr, f[4]);
	pr = two_sum(x[2], pr, f[3]);
	pr = two_sum(x[1], pr, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[7], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[8], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[9], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[10], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[11], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[12], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[13], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[14], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[15], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[16], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[17], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[18], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[19], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[20], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[21], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[22], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[23], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[24], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[25], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[26], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[27], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[28], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[29], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[30], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[31], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[32], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[33], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[34], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[35], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[36], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[37], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[38], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[39], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[40], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[41], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[42], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[43], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[44], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[45], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[46], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[47], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[48], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[49], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[50], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[51], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[52], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[53], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[54], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[55], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[56], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[57], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[58], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[59], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[60], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[61], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[62], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[63], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[64], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[65], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[66], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[67], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[68], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[69], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[70], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[71], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[72], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[73], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[74], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[75], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[76], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[77], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[78], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[79], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[80], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[81], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[82], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[83], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[84], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[85], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[86], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[87], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[88], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[89], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[90], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[91], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[92], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[93], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[94], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[95], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[96], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[97], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[98], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[99], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[100], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[101], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[102], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[103], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[104], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[105], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[106], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[107], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[108], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[109], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[110], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[111], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[112], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[113], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[114], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[115], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[116], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[117], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[118], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[119], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[120], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[121], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[122], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[123], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[124], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[125], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[126], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[127], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[128], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[129], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[130], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[131], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[132], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[133], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[134], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[135], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[136], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[137], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[138], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[139], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[140], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[141], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[142], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[143], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[144], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[145], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[146], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[147], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[148], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[149], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[150], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[151], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[152], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[153], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[154], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[155], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[156], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[157], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[158], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[159], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[160], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[161], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[162], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[163], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[164], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[165], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[166], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[167], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[168], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[169], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[170], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[171], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[172], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[173], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[174], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[175], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[176], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[177], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[178], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[179], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[180], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[181], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[182], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[183], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[184], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[185], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[186], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[187], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[188], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[189], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[190], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[191], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[192], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[193], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[194], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[195], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[196], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[197], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[198], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[199], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[200], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[201], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[202], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[203], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[204], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[205], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[206], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[207], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[208], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[209], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[210], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[211], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[212], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[213], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[214], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[215], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[216], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[217], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[218], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[219], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[220], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[221], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[222], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[223], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[224], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[225], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[226], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[227], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[228], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[229], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[230], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[231], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[232], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[233], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[234], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[235], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[236], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[237], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[238], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[239], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[240], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[241], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[242], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[243], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[244], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[245], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[246], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[247], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[248], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[249], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[250], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[251], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[252], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[253], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[254], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[255], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[256], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[257], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[258], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[259], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[260], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[261], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[262], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[263], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[264], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[265], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[266], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[267], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[268], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[269], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[270], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[271], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[272], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[273], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[274], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[275], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[276], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[277], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[278], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[279], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[280], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[281], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[282], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[283], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[284], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[285], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[286], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[287], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[288], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[289], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[290], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[291], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[292], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[293], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[294], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[295], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[296], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[297], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[298], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[299], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[300], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[301], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[302], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[303], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[304], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[305], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[306], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[307], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[308], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[309], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[310], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[311], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[312], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[313], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[314], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[315], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[316], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[317], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[318], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[319], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[320], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[321], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[322], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[323], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[324], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[325], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[326], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[327], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[328], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[329], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[330], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[331], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[332], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[333], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[334], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[335], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[336], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[337], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[338], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[339], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[340], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[341], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[342], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[343], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[344], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[345], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[346], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[347], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[348], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[349], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[350], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[351], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[352], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[353], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[354], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[355], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[356], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[357], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[358], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[359], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[360], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[361], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[362], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[363], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[364], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[365], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[366], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[367], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[368], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[369], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[370], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[371], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[372], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[373], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[374], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[375], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[376], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[377], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[378], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[379], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[380], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[381], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[382], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[383], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[384], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[385], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[386], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[387], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[388], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[389], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[390], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[391], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[392], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[393], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[394], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[395], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[396], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[397], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[398], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[399], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[400], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[401], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[402], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[403], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[404], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[405], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[406], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[407], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[408], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[409], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[410], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[411], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[412], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[413], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[414], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[415], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[416], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[417], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[418], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[419], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[420], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[421], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[422], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[423], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[424], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[425], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[426], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[427], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[428], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[429], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[430], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[431], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[432], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[433], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[434], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[435], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[436], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[437], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[438], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[439], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[440], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[441], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[442], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[443], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[444], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[445], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[446], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[447], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[448], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[449], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[450], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[451], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[452], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[453], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[454], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[455], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[456], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[457], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[458], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[459], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[460], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[461], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[462], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[463], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[464], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[465], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[466], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=467; ptr<467 && i<124; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<467 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<467; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<468,467>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[466], x[467], x[467]);
	pr = fast_two_sum(x[465], pr, x[466]);
	pr = fast_two_sum(x[464], pr, x[465]);
	pr = fast_two_sum(x[463], pr, x[464]);
	pr = fast_two_sum(x[462], pr, x[463]);
	pr = fast_two_sum(x[461], pr, x[462]);
	pr = fast_two_sum(x[460], pr, x[461]);
	pr = fast_two_sum(x[459], pr, x[460]);
	pr = fast_two_sum(x[458], pr, x[459]);
	pr = fast_two_sum(x[457], pr, x[458]);
	pr = fast_two_sum(x[456], pr, x[457]);
	pr = fast_two_sum(x[455], pr, x[456]);
	pr = fast_two_sum(x[454], pr, x[455]);
	pr = fast_two_sum(x[453], pr, x[454]);
	pr = fast_two_sum(x[452], pr, x[453]);
	pr = fast_two_sum(x[451], pr, x[452]);
	pr = fast_two_sum(x[450], pr, x[451]);
	pr = fast_two_sum(x[449], pr, x[450]);
	pr = fast_two_sum(x[448], pr, x[449]);
	pr = fast_two_sum(x[447], pr, x[448]);
	pr = fast_two_sum(x[446], pr, x[447]);
	pr = fast_two_sum(x[445], pr, x[446]);
	pr = fast_two_sum(x[444], pr, x[445]);
	pr = fast_two_sum(x[443], pr, x[444]);
	pr = fast_two_sum(x[442], pr, x[443]);
	pr = fast_two_sum(x[441], pr, x[442]);
	pr = fast_two_sum(x[440], pr, x[441]);
	pr = fast_two_sum(x[439], pr, x[440]);
	pr = fast_two_sum(x[438], pr, x[439]);
	pr = fast_two_sum(x[437], pr, x[438]);
	pr = fast_two_sum(x[436], pr, x[437]);
	pr = fast_two_sum(x[435], pr, x[436]);
	pr = fast_two_sum(x[434], pr, x[435]);
	pr = fast_two_sum(x[433], pr, x[434]);
	pr = fast_two_sum(x[432], pr, x[433]);
	pr = fast_two_sum(x[431], pr, x[432]);
	pr = fast_two_sum(x[430], pr, x[431]);
	pr = fast_two_sum(x[429], pr, x[430]);
	pr = fast_two_sum(x[428], pr, x[429]);
	pr = fast_two_sum(x[427], pr, x[428]);
	pr = fast_two_sum(x[426], pr, x[427]);
	pr = fast_two_sum(x[425], pr, x[426]);
	pr = fast_two_sum(x[424], pr, x[425]);
	pr = fast_two_sum(x[423], pr, x[424]);
	pr = fast_two_sum(x[422], pr, x[423]);
	pr = fast_two_sum(x[421], pr, x[422]);
	pr = fast_two_sum(x[420], pr, x[421]);
	pr = fast_two_sum(x[419], pr, x[420]);
	pr = fast_two_sum(x[418], pr, x[419]);
	pr = fast_two_sum(x[417], pr, x[418]);
	pr = fast_two_sum(x[416], pr, x[417]);
	pr = fast_two_sum(x[415], pr, x[416]);
	pr = fast_two_sum(x[414], pr, x[415]);
	pr = fast_two_sum(x[413], pr, x[414]);
	pr = fast_two_sum(x[412], pr, x[413]);
	pr = fast_two_sum(x[411], pr, x[412]);
	pr = fast_two_sum(x[410], pr, x[411]);
	pr = fast_two_sum(x[409], pr, x[410]);
	pr = fast_two_sum(x[408], pr, x[409]);
	pr = fast_two_sum(x[407], pr, x[408]);
	pr = fast_two_sum(x[406], pr, x[407]);
	pr = fast_two_sum(x[405], pr, x[406]);
	pr = fast_two_sum(x[404], pr, x[405]);
	pr = fast_two_sum(x[403], pr, x[404]);
	pr = fast_two_sum(x[402], pr, x[403]);
	pr = fast_two_sum(x[401], pr, x[402]);
	pr = fast_two_sum(x[400], pr, x[401]);
	pr = fast_two_sum(x[399], pr, x[400]);
	pr = fast_two_sum(x[398], pr, x[399]);
	pr = fast_two_sum(x[397], pr, x[398]);
	pr = fast_two_sum(x[396], pr, x[397]);
	pr = fast_two_sum(x[395], pr, x[396]);
	pr = fast_two_sum(x[394], pr, x[395]);
	pr = fast_two_sum(x[393], pr, x[394]);
	pr = fast_two_sum(x[392], pr, x[393]);
	pr = fast_two_sum(x[391], pr, x[392]);
	pr = fast_two_sum(x[390], pr, x[391]);
	pr = fast_two_sum(x[389], pr, x[390]);
	pr = fast_two_sum(x[388], pr, x[389]);
	pr = fast_two_sum(x[387], pr, x[388]);
	pr = fast_two_sum(x[386], pr, x[387]);
	pr = fast_two_sum(x[385], pr, x[386]);
	pr = fast_two_sum(x[384], pr, x[385]);
	pr = fast_two_sum(x[383], pr, x[384]);
	pr = fast_two_sum(x[382], pr, x[383]);
	pr = fast_two_sum(x[381], pr, x[382]);
	pr = fast_two_sum(x[380], pr, x[381]);
	pr = fast_two_sum(x[379], pr, x[380]);
	pr = fast_two_sum(x[378], pr, x[379]);
	pr = fast_two_sum(x[377], pr, x[378]);
	pr = fast_two_sum(x[376], pr, x[377]);
	pr = fast_two_sum(x[375], pr, x[376]);
	pr = fast_two_sum(x[374], pr, x[375]);
	pr = fast_two_sum(x[373], pr, x[374]);
	pr = fast_two_sum(x[372], pr, x[373]);
	pr = fast_two_sum(x[371], pr, x[372]);
	pr = fast_two_sum(x[370], pr, x[371]);
	pr = fast_two_sum(x[369], pr, x[370]);
	pr = fast_two_sum(x[368], pr, x[369]);
	pr = fast_two_sum(x[367], pr, x[368]);
	pr = fast_two_sum(x[366], pr, x[367]);
	pr = fast_two_sum(x[365], pr, x[366]);
	pr = fast_two_sum(x[364], pr, x[365]);
	pr = fast_two_sum(x[363], pr, x[364]);
	pr = fast_two_sum(x[362], pr, x[363]);
	pr = fast_two_sum(x[361], pr, x[362]);
	pr = fast_two_sum(x[360], pr, x[361]);
	pr = fast_two_sum(x[359], pr, x[360]);
	pr = fast_two_sum(x[358], pr, x[359]);
	pr = fast_two_sum(x[357], pr, x[358]);
	pr = fast_two_sum(x[356], pr, x[357]);
	pr = fast_two_sum(x[355], pr, x[356]);
	pr = fast_two_sum(x[354], pr, x[355]);
	pr = fast_two_sum(x[353], pr, x[354]);
	pr = fast_two_sum(x[352], pr, x[353]);
	pr = fast_two_sum(x[351], pr, x[352]);
	pr = fast_two_sum(x[350], pr, x[351]);
	pr = fast_two_sum(x[349], pr, x[350]);
	pr = fast_two_sum(x[348], pr, x[349]);
	pr = fast_two_sum(x[347], pr, x[348]);
	pr = fast_two_sum(x[346], pr, x[347]);
	pr = fast_two_sum(x[345], pr, x[346]);
	pr = fast_two_sum(x[344], pr, x[345]);
	pr = fast_two_sum(x[343], pr, x[344]);
	pr = fast_two_sum(x[342], pr, x[343]);
	pr = fast_two_sum(x[341], pr, x[342]);
	pr = fast_two_sum(x[340], pr, x[341]);
	pr = fast_two_sum(x[339], pr, x[340]);
	pr = fast_two_sum(x[338], pr, x[339]);
	pr = fast_two_sum(x[337], pr, x[338]);
	pr = fast_two_sum(x[336], pr, x[337]);
	pr = fast_two_sum(x[335], pr, x[336]);
	pr = fast_two_sum(x[334], pr, x[335]);
	pr = fast_two_sum(x[333], pr, x[334]);
	pr = fast_two_sum(x[332], pr, x[333]);
	pr = fast_two_sum(x[331], pr, x[332]);
	pr = fast_two_sum(x[330], pr, x[331]);
	pr = fast_two_sum(x[329], pr, x[330]);
	pr = fast_two_sum(x[328], pr, x[329]);
	pr = fast_two_sum(x[327], pr, x[328]);
	pr = fast_two_sum(x[326], pr, x[327]);
	pr = fast_two_sum(x[325], pr, x[326]);
	pr = fast_two_sum(x[324], pr, x[325]);
	pr = fast_two_sum(x[323], pr, x[324]);
	pr = fast_two_sum(x[322], pr, x[323]);
	pr = fast_two_sum(x[321], pr, x[322]);
	pr = fast_two_sum(x[320], pr, x[321]);
	pr = fast_two_sum(x[319], pr, x[320]);
	pr = fast_two_sum(x[318], pr, x[319]);
	pr = fast_two_sum(x[317], pr, x[318]);
	pr = fast_two_sum(x[316], pr, x[317]);
	pr = fast_two_sum(x[315], pr, x[316]);
	pr = fast_two_sum(x[314], pr, x[315]);
	pr = fast_two_sum(x[313], pr, x[314]);
	pr = fast_two_sum(x[312], pr, x[313]);
	pr = fast_two_sum(x[311], pr, x[312]);
	pr = fast_two_sum(x[310], pr, x[311]);
	pr = fast_two_sum(x[309], pr, x[310]);
	pr = fast_two_sum(x[308], pr, x[309]);
	pr = fast_two_sum(x[307], pr, x[308]);
	pr = fast_two_sum(x[306], pr, x[307]);
	pr = fast_two_sum(x[305], pr, x[306]);
	pr = fast_two_sum(x[304], pr, x[305]);
	pr = fast_two_sum(x[303], pr, x[304]);
	pr = fast_two_sum(x[302], pr, x[303]);
	pr = fast_two_sum(x[301], pr, x[302]);
	pr = fast_two_sum(x[300], pr, x[301]);
	pr = fast_two_sum(x[299], pr, x[300]);
	pr = fast_two_sum(x[298], pr, x[299]);
	pr = fast_two_sum(x[297], pr, x[298]);
	pr = fast_two_sum(x[296], pr, x[297]);
	pr = fast_two_sum(x[295], pr, x[296]);
	pr = fast_two_sum(x[294], pr, x[295]);
	pr = fast_two_sum(x[293], pr, x[294]);
	pr = fast_two_sum(x[292], pr, x[293]);
	pr = fast_two_sum(x[291], pr, x[292]);
	pr = fast_two_sum(x[290], pr, x[291]);
	pr = fast_two_sum(x[289], pr, x[290]);
	pr = fast_two_sum(x[288], pr, x[289]);
	pr = fast_two_sum(x[287], pr, x[288]);
	pr = fast_two_sum(x[286], pr, x[287]);
	pr = fast_two_sum(x[285], pr, x[286]);
	pr = fast_two_sum(x[284], pr, x[285]);
	pr = fast_two_sum(x[283], pr, x[284]);
	pr = fast_two_sum(x[282], pr, x[283]);
	pr = fast_two_sum(x[281], pr, x[282]);
	pr = fast_two_sum(x[280], pr, x[281]);
	pr = fast_two_sum(x[279], pr, x[280]);
	pr = fast_two_sum(x[278], pr, x[279]);
	pr = fast_two_sum(x[277], pr, x[278]);
	pr = fast_two_sum(x[276], pr, x[277]);
	pr = fast_two_sum(x[275], pr, x[276]);
	pr = fast_two_sum(x[274], pr, x[275]);
	pr = fast_two_sum(x[273], pr, x[274]);
	pr = fast_two_sum(x[272], pr, x[273]);
	pr = fast_two_sum(x[271], pr, x[272]);
	pr = fast_two_sum(x[270], pr, x[271]);
	pr = fast_two_sum(x[269], pr, x[270]);
	pr = fast_two_sum(x[268], pr, x[269]);
	pr = fast_two_sum(x[267], pr, x[268]);
	pr = fast_two_sum(x[266], pr, x[267]);
	pr = fast_two_sum(x[265], pr, x[266]);
	pr = fast_two_sum(x[264], pr, x[265]);
	pr = fast_two_sum(x[263], pr, x[264]);
	pr = fast_two_sum(x[262], pr, x[263]);
	pr = fast_two_sum(x[261], pr, x[262]);
	pr = fast_two_sum(x[260], pr, x[261]);
	pr = fast_two_sum(x[259], pr, x[260]);
	pr = fast_two_sum(x[258], pr, x[259]);
	pr = fast_two_sum(x[257], pr, x[258]);
	pr = fast_two_sum(x[256], pr, x[257]);
	pr = fast_two_sum(x[255], pr, x[256]);
	pr = fast_two_sum(x[254], pr, x[255]);
	pr = fast_two_sum(x[253], pr, x[254]);
	pr = fast_two_sum(x[252], pr, x[253]);
	pr = fast_two_sum(x[251], pr, x[252]);
	pr = fast_two_sum(x[250], pr, x[251]);
	pr = fast_two_sum(x[249], pr, x[250]);
	pr = fast_two_sum(x[248], pr, x[249]);
	pr = fast_two_sum(x[247], pr, x[248]);
	pr = fast_two_sum(x[246], pr, x[247]);
	pr = fast_two_sum(x[245], pr, x[246]);
	pr = fast_two_sum(x[244], pr, x[245]);
	pr = fast_two_sum(x[243], pr, x[244]);
	pr = fast_two_sum(x[242], pr, x[243]);
	pr = fast_two_sum(x[241], pr, x[242]);
	pr = fast_two_sum(x[240], pr, x[241]);
	pr = fast_two_sum(x[239], pr, x[240]);
	pr = fast_two_sum(x[238], pr, x[239]);
	pr = fast_two_sum(x[237], pr, x[238]);
	pr = fast_two_sum(x[236], pr, x[237]);
	pr = fast_two_sum(x[235], pr, x[236]);
	pr = fast_two_sum(x[234], pr, x[235]);
	pr = fast_two_sum(x[233], pr, x[234]);
	pr = fast_two_sum(x[232], pr, x[233]);
	pr = fast_two_sum(x[231], pr, x[232]);
	pr = fast_two_sum(x[230], pr, x[231]);
	pr = fast_two_sum(x[229], pr, x[230]);
	pr = fast_two_sum(x[228], pr, x[229]);
	pr = fast_two_sum(x[227], pr, x[228]);
	pr = fast_two_sum(x[226], pr, x[227]);
	pr = fast_two_sum(x[225], pr, x[226]);
	pr = fast_two_sum(x[224], pr, x[225]);
	pr = fast_two_sum(x[223], pr, x[224]);
	pr = fast_two_sum(x[222], pr, x[223]);
	pr = fast_two_sum(x[221], pr, x[222]);
	pr = fast_two_sum(x[220], pr, x[221]);
	pr = fast_two_sum(x[219], pr, x[220]);
	pr = fast_two_sum(x[218], pr, x[219]);
	pr = fast_two_sum(x[217], pr, x[218]);
	pr = fast_two_sum(x[216], pr, x[217]);
	pr = fast_two_sum(x[215], pr, x[216]);
	pr = fast_two_sum(x[214], pr, x[215]);
	pr = fast_two_sum(x[213], pr, x[214]);
	pr = fast_two_sum(x[212], pr, x[213]);
	pr = fast_two_sum(x[211], pr, x[212]);
	pr = fast_two_sum(x[210], pr, x[211]);
	pr = fast_two_sum(x[209], pr, x[210]);
	pr = fast_two_sum(x[208], pr, x[209]);
	pr = fast_two_sum(x[207], pr, x[208]);
	pr = fast_two_sum(x[206], pr, x[207]);
	pr = fast_two_sum(x[205], pr, x[206]);
	pr = fast_two_sum(x[204], pr, x[205]);
	pr = fast_two_sum(x[203], pr, x[204]);
	pr = fast_two_sum(x[202], pr, x[203]);
	pr = fast_two_sum(x[201], pr, x[202]);
	pr = fast_two_sum(x[200], pr, x[201]);
	pr = fast_two_sum(x[199], pr, x[200]);
	pr = fast_two_sum(x[198], pr, x[199]);
	pr = fast_two_sum(x[197], pr, x[198]);
	pr = fast_two_sum(x[196], pr, x[197]);
	pr = fast_two_sum(x[195], pr, x[196]);
	pr = fast_two_sum(x[194], pr, x[195]);
	pr = fast_two_sum(x[193], pr, x[194]);
	pr = fast_two_sum(x[192], pr, x[193]);
	pr = fast_two_sum(x[191], pr, x[192]);
	pr = fast_two_sum(x[190], pr, x[191]);
	pr = fast_two_sum(x[189], pr, x[190]);
	pr = fast_two_sum(x[188], pr, x[189]);
	pr = fast_two_sum(x[187], pr, x[188]);
	pr = fast_two_sum(x[186], pr, x[187]);
	pr = fast_two_sum(x[185], pr, x[186]);
	pr = fast_two_sum(x[184], pr, x[185]);
	pr = fast_two_sum(x[183], pr, x[184]);
	pr = fast_two_sum(x[182], pr, x[183]);
	pr = fast_two_sum(x[181], pr, x[182]);
	pr = fast_two_sum(x[180], pr, x[181]);
	pr = fast_two_sum(x[179], pr, x[180]);
	pr = fast_two_sum(x[178], pr, x[179]);
	pr = fast_two_sum(x[177], pr, x[178]);
	pr = fast_two_sum(x[176], pr, x[177]);
	pr = fast_two_sum(x[175], pr, x[176]);
	pr = fast_two_sum(x[174], pr, x[175]);
	pr = fast_two_sum(x[173], pr, x[174]);
	pr = fast_two_sum(x[172], pr, x[173]);
	pr = fast_two_sum(x[171], pr, x[172]);
	pr = fast_two_sum(x[170], pr, x[171]);
	pr = fast_two_sum(x[169], pr, x[170]);
	pr = fast_two_sum(x[168], pr, x[169]);
	pr = fast_two_sum(x[167], pr, x[168]);
	pr = fast_two_sum(x[166], pr, x[167]);
	pr = fast_two_sum(x[165], pr, x[166]);
	pr = fast_two_sum(x[164], pr, x[165]);
	pr = fast_two_sum(x[163], pr, x[164]);
	pr = fast_two_sum(x[162], pr, x[163]);
	pr = fast_two_sum(x[161], pr, x[162]);
	pr = fast_two_sum(x[160], pr, x[161]);
	pr = fast_two_sum(x[159], pr, x[160]);
	pr = fast_two_sum(x[158], pr, x[159]);
	pr = fast_two_sum(x[157], pr, x[158]);
	pr = fast_two_sum(x[156], pr, x[157]);
	pr = fast_two_sum(x[155], pr, x[156]);
	pr = fast_two_sum(x[154], pr, x[155]);
	pr = fast_two_sum(x[153], pr, x[154]);
	pr = fast_two_sum(x[152], pr, x[153]);
	pr = fast_two_sum(x[151], pr, x[152]);
	pr = fast_two_sum(x[150], pr, x[151]);
	pr = fast_two_sum(x[149], pr, x[150]);
	pr = fast_two_sum(x[148], pr, x[149]);
	pr = fast_two_sum(x[147], pr, x[148]);
	pr = fast_two_sum(x[146], pr, x[147]);
	pr = fast_two_sum(x[145], pr, x[146]);
	pr = fast_two_sum(x[144], pr, x[145]);
	pr = fast_two_sum(x[143], pr, x[144]);
	pr = fast_two_sum(x[142], pr, x[143]);
	pr = fast_two_sum(x[141], pr, x[142]);
	pr = fast_two_sum(x[140], pr, x[141]);
	pr = fast_two_sum(x[139], pr, x[140]);
	pr = fast_two_sum(x[138], pr, x[139]);
	pr = fast_two_sum(x[137], pr, x[138]);
	pr = fast_two_sum(x[136], pr, x[137]);
	pr = fast_two_sum(x[135], pr, x[136]);
	pr = fast_two_sum(x[134], pr, x[135]);
	pr = fast_two_sum(x[133], pr, x[134]);
	pr = fast_two_sum(x[132], pr, x[133]);
	pr = fast_two_sum(x[131], pr, x[132]);
	pr = fast_two_sum(x[130], pr, x[131]);
	pr = fast_two_sum(x[129], pr, x[130]);
	pr = fast_two_sum(x[128], pr, x[129]);
	pr = fast_two_sum(x[127], pr, x[128]);
	pr = fast_two_sum(x[126], pr, x[127]);
	pr = fast_two_sum(x[125], pr, x[126]);
	pr = fast_two_sum(x[124], pr, x[125]);
	pr = fast_two_sum(x[123], pr, x[124]);
	pr = fast_two_sum(x[122], pr, x[123]);
	pr = fast_two_sum(x[121], pr, x[122]);
	pr = fast_two_sum(x[120], pr, x[121]);
	pr = fast_two_sum(x[119], pr, x[120]);
	pr = fast_two_sum(x[118], pr, x[119]);
	pr = fast_two_sum(x[117], pr, x[118]);
	pr = fast_two_sum(x[116], pr, x[117]);
	pr = fast_two_sum(x[115], pr, x[116]);
	pr = fast_two_sum(x[114], pr, x[115]);
	pr = fast_two_sum(x[113], pr, x[114]);
	pr = fast_two_sum(x[112], pr, x[113]);
	pr = fast_two_sum(x[111], pr, x[112]);
	pr = fast_two_sum(x[110], pr, x[111]);
	pr = fast_two_sum(x[109], pr, x[110]);
	pr = fast_two_sum(x[108], pr, x[109]);
	pr = fast_two_sum(x[107], pr, x[108]);
	pr = fast_two_sum(x[106], pr, x[107]);
	pr = fast_two_sum(x[105], pr, x[106]);
	pr = fast_two_sum(x[104], pr, x[105]);
	pr = fast_two_sum(x[103], pr, x[104]);
	pr = fast_two_sum(x[102], pr, x[103]);
	pr = fast_two_sum(x[101], pr, x[102]);
	pr = fast_two_sum(x[100], pr, x[101]);
	pr = fast_two_sum(x[99], pr, x[100]);
	pr = fast_two_sum(x[98], pr, x[99]);
	pr = fast_two_sum(x[97], pr, x[98]);
	pr = fast_two_sum(x[96], pr, x[97]);
	pr = fast_two_sum(x[95], pr, x[96]);
	pr = fast_two_sum(x[94], pr, x[95]);
	pr = fast_two_sum(x[93], pr, x[94]);
	pr = fast_two_sum(x[92], pr, x[93]);
	pr = fast_two_sum(x[91], pr, x[92]);
	pr = fast_two_sum(x[90], pr, x[91]);
	pr = fast_two_sum(x[89], pr, x[90]);
	pr = fast_two_sum(x[88], pr, x[89]);
	pr = fast_two_sum(x[87], pr, x[88]);
	pr = fast_two_sum(x[86], pr, x[87]);
	pr = fast_two_sum(x[85], pr, x[86]);
	pr = fast_two_sum(x[84], pr, x[85]);
	pr = fast_two_sum(x[83], pr, x[84]);
	pr = fast_two_sum(x[82], pr, x[83]);
	pr = fast_two_sum(x[81], pr, x[82]);
	pr = fast_two_sum(x[80], pr, x[81]);
	pr = fast_two_sum(x[79], pr, x[80]);
	pr = fast_two_sum(x[78], pr, x[79]);
	pr = fast_two_sum(x[77], pr, x[78]);
	pr = fast_two_sum(x[76], pr, x[77]);
	pr = fast_two_sum(x[75], pr, x[76]);
	pr = fast_two_sum(x[74], pr, x[75]);
	pr = fast_two_sum(x[73], pr, x[74]);
	pr = fast_two_sum(x[72], pr, x[73]);
	pr = fast_two_sum(x[71], pr, x[72]);
	pr = fast_two_sum(x[70], pr, x[71]);
	pr = fast_two_sum(x[69], pr, x[70]);
	pr = fast_two_sum(x[68], pr, x[69]);
	pr = fast_two_sum(x[67], pr, x[68]);
	pr = fast_two_sum(x[66], pr, x[67]);
	pr = fast_two_sum(x[65], pr, x[66]);
	pr = fast_two_sum(x[64], pr, x[65]);
	pr = fast_two_sum(x[63], pr, x[64]);
	pr = fast_two_sum(x[62], pr, x[63]);
	pr = fast_two_sum(x[61], pr, x[62]);
	pr = fast_two_sum(x[60], pr, x[61]);
	pr = fast_two_sum(x[59], pr, x[60]);
	pr = fast_two_sum(x[58], pr, x[59]);
	pr = fast_two_sum(x[57], pr, x[58]);
	pr = fast_two_sum(x[56], pr, x[57]);
	pr = fast_two_sum(x[55], pr, x[56]);
	pr = fast_two_sum(x[54], pr, x[55]);
	pr = fast_two_sum(x[53], pr, x[54]);
	pr = fast_two_sum(x[52], pr, x[53]);
	pr = fast_two_sum(x[51], pr, x[52]);
	pr = fast_two_sum(x[50], pr, x[51]);
	pr = fast_two_sum(x[49], pr, x[50]);
	pr = fast_two_sum(x[48], pr, x[49]);
	pr = fast_two_sum(x[47], pr, x[48]);
	pr = fast_two_sum(x[46], pr, x[47]);
	pr = fast_two_sum(x[45], pr, x[46]);
	pr = fast_two_sum(x[44], pr, x[45]);
	pr = fast_two_sum(x[43], pr, x[44]);
	pr = fast_two_sum(x[42], pr, x[43]);
	pr = fast_two_sum(x[41], pr, x[42]);
	pr = fast_two_sum(x[40], pr, x[41]);
	pr = fast_two_sum(x[39], pr, x[40]);
	pr = fast_two_sum(x[38], pr, x[39]);
	pr = fast_two_sum(x[37], pr, x[38]);
	pr = fast_two_sum(x[36], pr, x[37]);
	pr = fast_two_sum(x[35], pr, x[36]);
	pr = fast_two_sum(x[34], pr, x[35]);
	pr = fast_two_sum(x[33], pr, x[34]);
	pr = fast_two_sum(x[32], pr, x[33]);
	pr = fast_two_sum(x[31], pr, x[32]);
	pr = fast_two_sum(x[30], pr, x[31]);
	pr = fast_two_sum(x[29], pr, x[30]);
	pr = fast_two_sum(x[28], pr, x[29]);
	pr = fast_two_sum(x[27], pr, x[28]);
	pr = fast_two_sum(x[26], pr, x[27]);
	pr = fast_two_sum(x[25], pr, x[26]);
	pr = fast_two_sum(x[24], pr, x[25]);
	pr = fast_two_sum(x[23], pr, x[24]);
	pr = fast_two_sum(x[22], pr, x[23]);
	pr = fast_two_sum(x[21], pr, x[22]);
	pr = fast_two_sum(x[20], pr, x[21]);
	pr = fast_two_sum(x[19], pr, x[20]);
	pr = fast_two_sum(x[18], pr, x[19]);
	pr = fast_two_sum(x[17], pr, x[18]);
	pr = fast_two_sum(x[16], pr, x[17]);
	pr = fast_two_sum(x[15], pr, x[16]);
	pr = fast_two_sum(x[14], pr, x[15]);
	pr = fast_two_sum(x[13], pr, x[14]);
	pr = fast_two_sum(x[12], pr, x[13]);
	pr = fast_two_sum(x[11], pr, x[12]);
	pr = fast_two_sum(x[10], pr, x[11]);
	pr = fast_two_sum(x[9], pr, x[10]);
	pr = fast_two_sum(x[8], pr, x[9]);
	pr = fast_two_sum(x[7], pr, x[8]);
	pr = fast_two_sum(x[6], pr, x[7]);
	pr = fast_two_sum(x[5], pr, x[6]);
	pr = fast_two_sum(x[4], pr, x[5]);
	pr = fast_two_sum(x[3], pr, x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[5], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[6], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[7], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[8], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[9], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[10], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[11], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[12], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[13], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[14], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[15], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[16], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[17], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[18], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[19], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[20], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[21], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[22], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[23], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[24], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[25], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[26], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[27], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[28], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[29], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[30], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[31], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[32], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[33], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[34], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[35], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[36], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[37], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[38], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[39], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[40], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[41], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[42], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[43], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[44], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[45], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[46], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[47], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[48], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[49], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[50], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[51], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[52], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[53], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[54], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[55], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[56], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[57], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[58], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[59], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[60], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[61], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[62], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[63], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[64], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[65], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[66], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[67], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[68], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[69], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[70], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[71], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[72], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[73], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[74], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[75], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[76], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[77], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[78], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[79], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[80], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[81], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[82], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[83], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[84], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[85], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[86], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[87], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[88], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[89], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[90], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[91], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[92], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[93], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[94], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[95], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[96], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[97], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[98], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[99], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[100], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[101], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[102], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[103], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[104], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[105], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[106], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[107], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[108], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[109], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[110], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[111], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[112], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[113], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[114], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[115], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[116], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[117], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[118], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[119], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[120], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[121], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[122], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[123], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[124], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[125], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[126], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[127], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[128], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[129], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[130], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[131], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[132], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[133], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[134], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[135], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[136], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[137], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[138], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[139], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[140], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[141], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[142], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[143], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[144], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[145], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[146], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[147], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[148], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[149], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[150], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[151], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[152], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[153], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[154], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[155], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[156], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[157], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[158], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[159], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[160], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[161], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[162], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[163], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[164], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[165], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[166], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[167], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[168], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[169], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[170], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[171], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[172], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[173], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[174], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[175], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[176], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[177], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[178], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[179], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[180], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[181], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[182], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[183], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[184], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[185], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[186], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[187], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[188], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[189], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[190], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[191], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[192], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[193], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[194], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[195], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[196], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[197], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[198], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[199], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[200], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[201], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[202], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[203], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[204], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[205], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[206], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[207], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[208], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[209], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[210], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[211], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[212], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[213], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[214], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[215], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[216], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[217], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[218], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[219], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[220], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[221], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[222], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[223], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[224], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[225], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[226], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[227], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[228], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[229], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[230], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[231], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[232], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[233], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[234], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[235], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[236], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[237], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[238], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[239], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[240], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[241], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[242], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[243], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[244], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[245], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[246], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[247], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[248], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[249], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[250], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[251], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[252], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[253], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[254], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[255], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[256], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[257], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[258], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[259], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[260], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[261], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[262], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[263], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[264], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[265], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[266], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[267], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[268], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[269], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[270], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[271], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[272], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[273], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[274], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[275], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[276], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[277], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[278], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[279], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[280], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[281], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[282], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[283], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[284], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[285], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[286], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[287], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[288], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[289], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[290], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[291], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[292], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[293], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[294], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[295], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[296], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[297], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[298], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[299], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[300], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[301], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[302], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[303], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[304], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[305], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[306], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[307], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[308], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[309], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[310], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[311], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[312], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[313], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[314], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[315], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[316], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[317], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[318], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[319], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[320], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[321], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[322], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[323], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[324], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[325], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[326], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[327], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[328], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[329], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[330], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[331], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[332], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[333], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[334], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[335], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[336], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[337], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[338], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[339], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[340], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[341], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[342], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[343], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[344], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[345], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[346], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[347], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[348], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[349], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[350], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[351], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[352], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[353], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[354], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[355], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[356], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[357], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[358], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[359], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[360], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[361], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[362], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[363], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[364], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[365], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[366], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[367], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[368], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[369], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[370], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[371], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[372], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[373], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[374], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[375], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[376], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[377], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[378], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[379], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[380], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[381], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[382], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[383], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[384], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[385], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[386], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[387], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[388], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[389], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[390], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[391], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[392], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[393], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[394], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[395], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[396], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[397], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[398], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[399], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[400], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[401], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[402], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[403], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[404], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[405], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[406], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[407], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[408], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[409], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[410], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[411], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[412], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[413], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[414], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[415], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[416], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[417], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[418], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[419], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[420], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[421], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[422], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[423], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[424], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[425], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[426], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[427], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[428], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[429], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[430], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[431], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[432], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[433], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[434], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[435], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[436], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[437], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[438], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[439], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[440], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[441], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[442], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[443], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[444], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[445], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[446], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[447], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[448], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[449], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[450], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[451], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[452], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[453], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[454], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[455], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[456], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[457], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[458], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[459], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[460], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[461], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[462], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[463], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[464], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[465], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[466], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[467], x[ptr+1]);

	for(int i=ptr+1; i<467; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<468,467>(const double *x, double *r){
	double f[468];
	int ptr=0;
	double pr = fast_two_sum(x[466], x[467], f[467]);
	pr = fast_two_sum(x[465], pr, f[466]);
	pr = fast_two_sum(x[464], pr, f[465]);
	pr = fast_two_sum(x[463], pr, f[464]);
	pr = fast_two_sum(x[462], pr, f[463]);
	pr = fast_two_sum(x[461], pr, f[462]);
	pr = fast_two_sum(x[460], pr, f[461]);
	pr = fast_two_sum(x[459], pr, f[460]);
	pr = fast_two_sum(x[458], pr, f[459]);
	pr = fast_two_sum(x[457], pr, f[458]);
	pr = fast_two_sum(x[456], pr, f[457]);
	pr = fast_two_sum(x[455], pr, f[456]);
	pr = fast_two_sum(x[454], pr, f[455]);
	pr = fast_two_sum(x[453], pr, f[454]);
	pr = fast_two_sum(x[452], pr, f[453]);
	pr = fast_two_sum(x[451], pr, f[452]);
	pr = fast_two_sum(x[450], pr, f[451]);
	pr = fast_two_sum(x[449], pr, f[450]);
	pr = fast_two_sum(x[448], pr, f[449]);
	pr = fast_two_sum(x[447], pr, f[448]);
	pr = fast_two_sum(x[446], pr, f[447]);
	pr = fast_two_sum(x[445], pr, f[446]);
	pr = fast_two_sum(x[444], pr, f[445]);
	pr = fast_two_sum(x[443], pr, f[444]);
	pr = fast_two_sum(x[442], pr, f[443]);
	pr = fast_two_sum(x[441], pr, f[442]);
	pr = fast_two_sum(x[440], pr, f[441]);
	pr = fast_two_sum(x[439], pr, f[440]);
	pr = fast_two_sum(x[438], pr, f[439]);
	pr = fast_two_sum(x[437], pr, f[438]);
	pr = fast_two_sum(x[436], pr, f[437]);
	pr = fast_two_sum(x[435], pr, f[436]);
	pr = fast_two_sum(x[434], pr, f[435]);
	pr = fast_two_sum(x[433], pr, f[434]);
	pr = fast_two_sum(x[432], pr, f[433]);
	pr = fast_two_sum(x[431], pr, f[432]);
	pr = fast_two_sum(x[430], pr, f[431]);
	pr = fast_two_sum(x[429], pr, f[430]);
	pr = fast_two_sum(x[428], pr, f[429]);
	pr = fast_two_sum(x[427], pr, f[428]);
	pr = fast_two_sum(x[426], pr, f[427]);
	pr = fast_two_sum(x[425], pr, f[426]);
	pr = fast_two_sum(x[424], pr, f[425]);
	pr = fast_two_sum(x[423], pr, f[424]);
	pr = fast_two_sum(x[422], pr, f[423]);
	pr = fast_two_sum(x[421], pr, f[422]);
	pr = fast_two_sum(x[420], pr, f[421]);
	pr = fast_two_sum(x[419], pr, f[420]);
	pr = fast_two_sum(x[418], pr, f[419]);
	pr = fast_two_sum(x[417], pr, f[418]);
	pr = fast_two_sum(x[416], pr, f[417]);
	pr = fast_two_sum(x[415], pr, f[416]);
	pr = fast_two_sum(x[414], pr, f[415]);
	pr = fast_two_sum(x[413], pr, f[414]);
	pr = fast_two_sum(x[412], pr, f[413]);
	pr = fast_two_sum(x[411], pr, f[412]);
	pr = fast_two_sum(x[410], pr, f[411]);
	pr = fast_two_sum(x[409], pr, f[410]);
	pr = fast_two_sum(x[408], pr, f[409]);
	pr = fast_two_sum(x[407], pr, f[408]);
	pr = fast_two_sum(x[406], pr, f[407]);
	pr = fast_two_sum(x[405], pr, f[406]);
	pr = fast_two_sum(x[404], pr, f[405]);
	pr = fast_two_sum(x[403], pr, f[404]);
	pr = fast_two_sum(x[402], pr, f[403]);
	pr = fast_two_sum(x[401], pr, f[402]);
	pr = fast_two_sum(x[400], pr, f[401]);
	pr = fast_two_sum(x[399], pr, f[400]);
	pr = fast_two_sum(x[398], pr, f[399]);
	pr = fast_two_sum(x[397], pr, f[398]);
	pr = fast_two_sum(x[396], pr, f[397]);
	pr = fast_two_sum(x[395], pr, f[396]);
	pr = fast_two_sum(x[394], pr, f[395]);
	pr = fast_two_sum(x[393], pr, f[394]);
	pr = fast_two_sum(x[392], pr, f[393]);
	pr = fast_two_sum(x[391], pr, f[392]);
	pr = fast_two_sum(x[390], pr, f[391]);
	pr = fast_two_sum(x[389], pr, f[390]);
	pr = fast_two_sum(x[388], pr, f[389]);
	pr = fast_two_sum(x[387], pr, f[388]);
	pr = fast_two_sum(x[386], pr, f[387]);
	pr = fast_two_sum(x[385], pr, f[386]);
	pr = fast_two_sum(x[384], pr, f[385]);
	pr = fast_two_sum(x[383], pr, f[384]);
	pr = fast_two_sum(x[382], pr, f[383]);
	pr = fast_two_sum(x[381], pr, f[382]);
	pr = fast_two_sum(x[380], pr, f[381]);
	pr = fast_two_sum(x[379], pr, f[380]);
	pr = fast_two_sum(x[378], pr, f[379]);
	pr = fast_two_sum(x[377], pr, f[378]);
	pr = fast_two_sum(x[376], pr, f[377]);
	pr = fast_two_sum(x[375], pr, f[376]);
	pr = fast_two_sum(x[374], pr, f[375]);
	pr = fast_two_sum(x[373], pr, f[374]);
	pr = fast_two_sum(x[372], pr, f[373]);
	pr = fast_two_sum(x[371], pr, f[372]);
	pr = fast_two_sum(x[370], pr, f[371]);
	pr = fast_two_sum(x[369], pr, f[370]);
	pr = fast_two_sum(x[368], pr, f[369]);
	pr = fast_two_sum(x[367], pr, f[368]);
	pr = fast_two_sum(x[366], pr, f[367]);
	pr = fast_two_sum(x[365], pr, f[366]);
	pr = fast_two_sum(x[364], pr, f[365]);
	pr = fast_two_sum(x[363], pr, f[364]);
	pr = fast_two_sum(x[362], pr, f[363]);
	pr = fast_two_sum(x[361], pr, f[362]);
	pr = fast_two_sum(x[360], pr, f[361]);
	pr = fast_two_sum(x[359], pr, f[360]);
	pr = fast_two_sum(x[358], pr, f[359]);
	pr = fast_two_sum(x[357], pr, f[358]);
	pr = fast_two_sum(x[356], pr, f[357]);
	pr = fast_two_sum(x[355], pr, f[356]);
	pr = fast_two_sum(x[354], pr, f[355]);
	pr = fast_two_sum(x[353], pr, f[354]);
	pr = fast_two_sum(x[352], pr, f[353]);
	pr = fast_two_sum(x[351], pr, f[352]);
	pr = fast_two_sum(x[350], pr, f[351]);
	pr = fast_two_sum(x[349], pr, f[350]);
	pr = fast_two_sum(x[348], pr, f[349]);
	pr = fast_two_sum(x[347], pr, f[348]);
	pr = fast_two_sum(x[346], pr, f[347]);
	pr = fast_two_sum(x[345], pr, f[346]);
	pr = fast_two_sum(x[344], pr, f[345]);
	pr = fast_two_sum(x[343], pr, f[344]);
	pr = fast_two_sum(x[342], pr, f[343]);
	pr = fast_two_sum(x[341], pr, f[342]);
	pr = fast_two_sum(x[340], pr, f[341]);
	pr = fast_two_sum(x[339], pr, f[340]);
	pr = fast_two_sum(x[338], pr, f[339]);
	pr = fast_two_sum(x[337], pr, f[338]);
	pr = fast_two_sum(x[336], pr, f[337]);
	pr = fast_two_sum(x[335], pr, f[336]);
	pr = fast_two_sum(x[334], pr, f[335]);
	pr = fast_two_sum(x[333], pr, f[334]);
	pr = fast_two_sum(x[332], pr, f[333]);
	pr = fast_two_sum(x[331], pr, f[332]);
	pr = fast_two_sum(x[330], pr, f[331]);
	pr = fast_two_sum(x[329], pr, f[330]);
	pr = fast_two_sum(x[328], pr, f[329]);
	pr = fast_two_sum(x[327], pr, f[328]);
	pr = fast_two_sum(x[326], pr, f[327]);
	pr = fast_two_sum(x[325], pr, f[326]);
	pr = fast_two_sum(x[324], pr, f[325]);
	pr = fast_two_sum(x[323], pr, f[324]);
	pr = fast_two_sum(x[322], pr, f[323]);
	pr = fast_two_sum(x[321], pr, f[322]);
	pr = fast_two_sum(x[320], pr, f[321]);
	pr = fast_two_sum(x[319], pr, f[320]);
	pr = fast_two_sum(x[318], pr, f[319]);
	pr = fast_two_sum(x[317], pr, f[318]);
	pr = fast_two_sum(x[316], pr, f[317]);
	pr = fast_two_sum(x[315], pr, f[316]);
	pr = fast_two_sum(x[314], pr, f[315]);
	pr = fast_two_sum(x[313], pr, f[314]);
	pr = fast_two_sum(x[312], pr, f[313]);
	pr = fast_two_sum(x[311], pr, f[312]);
	pr = fast_two_sum(x[310], pr, f[311]);
	pr = fast_two_sum(x[309], pr, f[310]);
	pr = fast_two_sum(x[308], pr, f[309]);
	pr = fast_two_sum(x[307], pr, f[308]);
	pr = fast_two_sum(x[306], pr, f[307]);
	pr = fast_two_sum(x[305], pr, f[306]);
	pr = fast_two_sum(x[304], pr, f[305]);
	pr = fast_two_sum(x[303], pr, f[304]);
	pr = fast_two_sum(x[302], pr, f[303]);
	pr = fast_two_sum(x[301], pr, f[302]);
	pr = fast_two_sum(x[300], pr, f[301]);
	pr = fast_two_sum(x[299], pr, f[300]);
	pr = fast_two_sum(x[298], pr, f[299]);
	pr = fast_two_sum(x[297], pr, f[298]);
	pr = fast_two_sum(x[296], pr, f[297]);
	pr = fast_two_sum(x[295], pr, f[296]);
	pr = fast_two_sum(x[294], pr, f[295]);
	pr = fast_two_sum(x[293], pr, f[294]);
	pr = fast_two_sum(x[292], pr, f[293]);
	pr = fast_two_sum(x[291], pr, f[292]);
	pr = fast_two_sum(x[290], pr, f[291]);
	pr = fast_two_sum(x[289], pr, f[290]);
	pr = fast_two_sum(x[288], pr, f[289]);
	pr = fast_two_sum(x[287], pr, f[288]);
	pr = fast_two_sum(x[286], pr, f[287]);
	pr = fast_two_sum(x[285], pr, f[286]);
	pr = fast_two_sum(x[284], pr, f[285]);
	pr = fast_two_sum(x[283], pr, f[284]);
	pr = fast_two_sum(x[282], pr, f[283]);
	pr = fast_two_sum(x[281], pr, f[282]);
	pr = fast_two_sum(x[280], pr, f[281]);
	pr = fast_two_sum(x[279], pr, f[280]);
	pr = fast_two_sum(x[278], pr, f[279]);
	pr = fast_two_sum(x[277], pr, f[278]);
	pr = fast_two_sum(x[276], pr, f[277]);
	pr = fast_two_sum(x[275], pr, f[276]);
	pr = fast_two_sum(x[274], pr, f[275]);
	pr = fast_two_sum(x[273], pr, f[274]);
	pr = fast_two_sum(x[272], pr, f[273]);
	pr = fast_two_sum(x[271], pr, f[272]);
	pr = fast_two_sum(x[270], pr, f[271]);
	pr = fast_two_sum(x[269], pr, f[270]);
	pr = fast_two_sum(x[268], pr, f[269]);
	pr = fast_two_sum(x[267], pr, f[268]);
	pr = fast_two_sum(x[266], pr, f[267]);
	pr = fast_two_sum(x[265], pr, f[266]);
	pr = fast_two_sum(x[264], pr, f[265]);
	pr = fast_two_sum(x[263], pr, f[264]);
	pr = fast_two_sum(x[262], pr, f[263]);
	pr = fast_two_sum(x[261], pr, f[262]);
	pr = fast_two_sum(x[260], pr, f[261]);
	pr = fast_two_sum(x[259], pr, f[260]);
	pr = fast_two_sum(x[258], pr, f[259]);
	pr = fast_two_sum(x[257], pr, f[258]);
	pr = fast_two_sum(x[256], pr, f[257]);
	pr = fast_two_sum(x[255], pr, f[256]);
	pr = fast_two_sum(x[254], pr, f[255]);
	pr = fast_two_sum(x[253], pr, f[254]);
	pr = fast_two_sum(x[252], pr, f[253]);
	pr = fast_two_sum(x[251], pr, f[252]);
	pr = fast_two_sum(x[250], pr, f[251]);
	pr = fast_two_sum(x[249], pr, f[250]);
	pr = fast_two_sum(x[248], pr, f[249]);
	pr = fast_two_sum(x[247], pr, f[248]);
	pr = fast_two_sum(x[246], pr, f[247]);
	pr = fast_two_sum(x[245], pr, f[246]);
	pr = fast_two_sum(x[244], pr, f[245]);
	pr = fast_two_sum(x[243], pr, f[244]);
	pr = fast_two_sum(x[242], pr, f[243]);
	pr = fast_two_sum(x[241], pr, f[242]);
	pr = fast_two_sum(x[240], pr, f[241]);
	pr = fast_two_sum(x[239], pr, f[240]);
	pr = fast_two_sum(x[238], pr, f[239]);
	pr = fast_two_sum(x[237], pr, f[238]);
	pr = fast_two_sum(x[236], pr, f[237]);
	pr = fast_two_sum(x[235], pr, f[236]);
	pr = fast_two_sum(x[234], pr, f[235]);
	pr = fast_two_sum(x[233], pr, f[234]);
	pr = fast_two_sum(x[232], pr, f[233]);
	pr = fast_two_sum(x[231], pr, f[232]);
	pr = fast_two_sum(x[230], pr, f[231]);
	pr = fast_two_sum(x[229], pr, f[230]);
	pr = fast_two_sum(x[228], pr, f[229]);
	pr = fast_two_sum(x[227], pr, f[228]);
	pr = fast_two_sum(x[226], pr, f[227]);
	pr = fast_two_sum(x[225], pr, f[226]);
	pr = fast_two_sum(x[224], pr, f[225]);
	pr = fast_two_sum(x[223], pr, f[224]);
	pr = fast_two_sum(x[222], pr, f[223]);
	pr = fast_two_sum(x[221], pr, f[222]);
	pr = fast_two_sum(x[220], pr, f[221]);
	pr = fast_two_sum(x[219], pr, f[220]);
	pr = fast_two_sum(x[218], pr, f[219]);
	pr = fast_two_sum(x[217], pr, f[218]);
	pr = fast_two_sum(x[216], pr, f[217]);
	pr = fast_two_sum(x[215], pr, f[216]);
	pr = fast_two_sum(x[214], pr, f[215]);
	pr = fast_two_sum(x[213], pr, f[214]);
	pr = fast_two_sum(x[212], pr, f[213]);
	pr = fast_two_sum(x[211], pr, f[212]);
	pr = fast_two_sum(x[210], pr, f[211]);
	pr = fast_two_sum(x[209], pr, f[210]);
	pr = fast_two_sum(x[208], pr, f[209]);
	pr = fast_two_sum(x[207], pr, f[208]);
	pr = fast_two_sum(x[206], pr, f[207]);
	pr = fast_two_sum(x[205], pr, f[206]);
	pr = fast_two_sum(x[204], pr, f[205]);
	pr = fast_two_sum(x[203], pr, f[204]);
	pr = fast_two_sum(x[202], pr, f[203]);
	pr = fast_two_sum(x[201], pr, f[202]);
	pr = fast_two_sum(x[200], pr, f[201]);
	pr = fast_two_sum(x[199], pr, f[200]);
	pr = fast_two_sum(x[198], pr, f[199]);
	pr = fast_two_sum(x[197], pr, f[198]);
	pr = fast_two_sum(x[196], pr, f[197]);
	pr = fast_two_sum(x[195], pr, f[196]);
	pr = fast_two_sum(x[194], pr, f[195]);
	pr = fast_two_sum(x[193], pr, f[194]);
	pr = fast_two_sum(x[192], pr, f[193]);
	pr = fast_two_sum(x[191], pr, f[192]);
	pr = fast_two_sum(x[190], pr, f[191]);
	pr = fast_two_sum(x[189], pr, f[190]);
	pr = fast_two_sum(x[188], pr, f[189]);
	pr = fast_two_sum(x[187], pr, f[188]);
	pr = fast_two_sum(x[186], pr, f[187]);
	pr = fast_two_sum(x[185], pr, f[186]);
	pr = fast_two_sum(x[184], pr, f[185]);
	pr = fast_two_sum(x[183], pr, f[184]);
	pr = fast_two_sum(x[182], pr, f[183]);
	pr = fast_two_sum(x[181], pr, f[182]);
	pr = fast_two_sum(x[180], pr, f[181]);
	pr = fast_two_sum(x[179], pr, f[180]);
	pr = fast_two_sum(x[178], pr, f[179]);
	pr = fast_two_sum(x[177], pr, f[178]);
	pr = fast_two_sum(x[176], pr, f[177]);
	pr = fast_two_sum(x[175], pr, f[176]);
	pr = fast_two_sum(x[174], pr, f[175]);
	pr = fast_two_sum(x[173], pr, f[174]);
	pr = fast_two_sum(x[172], pr, f[173]);
	pr = fast_two_sum(x[171], pr, f[172]);
	pr = fast_two_sum(x[170], pr, f[171]);
	pr = fast_two_sum(x[169], pr, f[170]);
	pr = fast_two_sum(x[168], pr, f[169]);
	pr = fast_two_sum(x[167], pr, f[168]);
	pr = fast_two_sum(x[166], pr, f[167]);
	pr = fast_two_sum(x[165], pr, f[166]);
	pr = fast_two_sum(x[164], pr, f[165]);
	pr = fast_two_sum(x[163], pr, f[164]);
	pr = fast_two_sum(x[162], pr, f[163]);
	pr = fast_two_sum(x[161], pr, f[162]);
	pr = fast_two_sum(x[160], pr, f[161]);
	pr = fast_two_sum(x[159], pr, f[160]);
	pr = fast_two_sum(x[158], pr, f[159]);
	pr = fast_two_sum(x[157], pr, f[158]);
	pr = fast_two_sum(x[156], pr, f[157]);
	pr = fast_two_sum(x[155], pr, f[156]);
	pr = fast_two_sum(x[154], pr, f[155]);
	pr = fast_two_sum(x[153], pr, f[154]);
	pr = fast_two_sum(x[152], pr, f[153]);
	pr = fast_two_sum(x[151], pr, f[152]);
	pr = fast_two_sum(x[150], pr, f[151]);
	pr = fast_two_sum(x[149], pr, f[150]);
	pr = fast_two_sum(x[148], pr, f[149]);
	pr = fast_two_sum(x[147], pr, f[148]);
	pr = fast_two_sum(x[146], pr, f[147]);
	pr = fast_two_sum(x[145], pr, f[146]);
	pr = fast_two_sum(x[144], pr, f[145]);
	pr = fast_two_sum(x[143], pr, f[144]);
	pr = fast_two_sum(x[142], pr, f[143]);
	pr = fast_two_sum(x[141], pr, f[142]);
	pr = fast_two_sum(x[140], pr, f[141]);
	pr = fast_two_sum(x[139], pr, f[140]);
	pr = fast_two_sum(x[138], pr, f[139]);
	pr = fast_two_sum(x[137], pr, f[138]);
	pr = fast_two_sum(x[136], pr, f[137]);
	pr = fast_two_sum(x[135], pr, f[136]);
	pr = fast_two_sum(x[134], pr, f[135]);
	pr = fast_two_sum(x[133], pr, f[134]);
	pr = fast_two_sum(x[132], pr, f[133]);
	pr = fast_two_sum(x[131], pr, f[132]);
	pr = fast_two_sum(x[130], pr, f[131]);
	pr = fast_two_sum(x[129], pr, f[130]);
	pr = fast_two_sum(x[128], pr, f[129]);
	pr = fast_two_sum(x[127], pr, f[128]);
	pr = fast_two_sum(x[126], pr, f[127]);
	pr = fast_two_sum(x[125], pr, f[126]);
	pr = fast_two_sum(x[124], pr, f[125]);
	pr = fast_two_sum(x[123], pr, f[124]);
	pr = fast_two_sum(x[122], pr, f[123]);
	pr = fast_two_sum(x[121], pr, f[122]);
	pr = fast_two_sum(x[120], pr, f[121]);
	pr = fast_two_sum(x[119], pr, f[120]);
	pr = fast_two_sum(x[118], pr, f[119]);
	pr = fast_two_sum(x[117], pr, f[118]);
	pr = fast_two_sum(x[116], pr, f[117]);
	pr = fast_two_sum(x[115], pr, f[116]);
	pr = fast_two_sum(x[114], pr, f[115]);
	pr = fast_two_sum(x[113], pr, f[114]);
	pr = fast_two_sum(x[112], pr, f[113]);
	pr = fast_two_sum(x[111], pr, f[112]);
	pr = fast_two_sum(x[110], pr, f[111]);
	pr = fast_two_sum(x[109], pr, f[110]);
	pr = fast_two_sum(x[108], pr, f[109]);
	pr = fast_two_sum(x[107], pr, f[108]);
	pr = fast_two_sum(x[106], pr, f[107]);
	pr = fast_two_sum(x[105], pr, f[106]);
	pr = fast_two_sum(x[104], pr, f[105]);
	pr = fast_two_sum(x[103], pr, f[104]);
	pr = fast_two_sum(x[102], pr, f[103]);
	pr = fast_two_sum(x[101], pr, f[102]);
	pr = fast_two_sum(x[100], pr, f[101]);
	pr = fast_two_sum(x[99], pr, f[100]);
	pr = fast_two_sum(x[98], pr, f[99]);
	pr = fast_two_sum(x[97], pr, f[98]);
	pr = fast_two_sum(x[96], pr, f[97]);
	pr = fast_two_sum(x[95], pr, f[96]);
	pr = fast_two_sum(x[94], pr, f[95]);
	pr = fast_two_sum(x[93], pr, f[94]);
	pr = fast_two_sum(x[92], pr, f[93]);
	pr = fast_two_sum(x[91], pr, f[92]);
	pr = fast_two_sum(x[90], pr, f[91]);
	pr = fast_two_sum(x[89], pr, f[90]);
	pr = fast_two_sum(x[88], pr, f[89]);
	pr = fast_two_sum(x[87], pr, f[88]);
	pr = fast_two_sum(x[86], pr, f[87]);
	pr = fast_two_sum(x[85], pr, f[86]);
	pr = fast_two_sum(x[84], pr, f[85]);
	pr = fast_two_sum(x[83], pr, f[84]);
	pr = fast_two_sum(x[82], pr, f[83]);
	pr = fast_two_sum(x[81], pr, f[82]);
	pr = fast_two_sum(x[80], pr, f[81]);
	pr = fast_two_sum(x[79], pr, f[80]);
	pr = fast_two_sum(x[78], pr, f[79]);
	pr = fast_two_sum(x[77], pr, f[78]);
	pr = fast_two_sum(x[76], pr, f[77]);
	pr = fast_two_sum(x[75], pr, f[76]);
	pr = fast_two_sum(x[74], pr, f[75]);
	pr = fast_two_sum(x[73], pr, f[74]);
	pr = fast_two_sum(x[72], pr, f[73]);
	pr = fast_two_sum(x[71], pr, f[72]);
	pr = fast_two_sum(x[70], pr, f[71]);
	pr = fast_two_sum(x[69], pr, f[70]);
	pr = fast_two_sum(x[68], pr, f[69]);
	pr = fast_two_sum(x[67], pr, f[68]);
	pr = fast_two_sum(x[66], pr, f[67]);
	pr = fast_two_sum(x[65], pr, f[66]);
	pr = fast_two_sum(x[64], pr, f[65]);
	pr = fast_two_sum(x[63], pr, f[64]);
	pr = fast_two_sum(x[62], pr, f[63]);
	pr = fast_two_sum(x[61], pr, f[62]);
	pr = fast_two_sum(x[60], pr, f[61]);
	pr = fast_two_sum(x[59], pr, f[60]);
	pr = fast_two_sum(x[58], pr, f[59]);
	pr = fast_two_sum(x[57], pr, f[58]);
	pr = fast_two_sum(x[56], pr, f[57]);
	pr = fast_two_sum(x[55], pr, f[56]);
	pr = fast_two_sum(x[54], pr, f[55]);
	pr = fast_two_sum(x[53], pr, f[54]);
	pr = fast_two_sum(x[52], pr, f[53]);
	pr = fast_two_sum(x[51], pr, f[52]);
	pr = fast_two_sum(x[50], pr, f[51]);
	pr = fast_two_sum(x[49], pr, f[50]);
	pr = fast_two_sum(x[48], pr, f[49]);
	pr = fast_two_sum(x[47], pr, f[48]);
	pr = fast_two_sum(x[46], pr, f[47]);
	pr = fast_two_sum(x[45], pr, f[46]);
	pr = fast_two_sum(x[44], pr, f[45]);
	pr = fast_two_sum(x[43], pr, f[44]);
	pr = fast_two_sum(x[42], pr, f[43]);
	pr = fast_two_sum(x[41], pr, f[42]);
	pr = fast_two_sum(x[40], pr, f[41]);
	pr = fast_two_sum(x[39], pr, f[40]);
	pr = fast_two_sum(x[38], pr, f[39]);
	pr = fast_two_sum(x[37], pr, f[38]);
	pr = fast_two_sum(x[36], pr, f[37]);
	pr = fast_two_sum(x[35], pr, f[36]);
	pr = fast_two_sum(x[34], pr, f[35]);
	pr = fast_two_sum(x[33], pr, f[34]);
	pr = fast_two_sum(x[32], pr, f[33]);
	pr = fast_two_sum(x[31], pr, f[32]);
	pr = fast_two_sum(x[30], pr, f[31]);
	pr = fast_two_sum(x[29], pr, f[30]);
	pr = fast_two_sum(x[28], pr, f[29]);
	pr = fast_two_sum(x[27], pr, f[28]);
	pr = fast_two_sum(x[26], pr, f[27]);
	pr = fast_two_sum(x[25], pr, f[26]);
	pr = fast_two_sum(x[24], pr, f[25]);
	pr = fast_two_sum(x[23], pr, f[24]);
	pr = fast_two_sum(x[22], pr, f[23]);
	pr = fast_two_sum(x[21], pr, f[22]);
	pr = fast_two_sum(x[20], pr, f[21]);
	pr = fast_two_sum(x[19], pr, f[20]);
	pr = fast_two_sum(x[18], pr, f[19]);
	pr = fast_two_sum(x[17], pr, f[18]);
	pr = fast_two_sum(x[16], pr, f[17]);
	pr = fast_two_sum(x[15], pr, f[16]);
	pr = fast_two_sum(x[14], pr, f[15]);
	pr = fast_two_sum(x[13], pr, f[14]);
	pr = fast_two_sum(x[12], pr, f[13]);
	pr = fast_two_sum(x[11], pr, f[12]);
	pr = fast_two_sum(x[10], pr, f[11]);
	pr = fast_two_sum(x[9], pr, f[10]);
	pr = fast_two_sum(x[8], pr, f[9]);
	pr = fast_two_sum(x[7], pr, f[8]);
	pr = fast_two_sum(x[6], pr, f[7]);
	pr = fast_two_sum(x[5], pr, f[6]);
	pr = fast_two_sum(x[4], pr, f[5]);
	pr = fast_two_sum(x[3], pr, f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[4], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[5], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[6], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[7], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[8], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[9], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[10], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[11], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[12], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[13], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[14], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[15], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[16], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[17], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[18], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[19], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[20], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[21], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[22], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[23], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[24], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[25], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[26], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[27], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[28], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[29], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[30], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[31], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[32], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[33], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[34], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[35], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[36], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[37], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[38], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[39], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[40], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[41], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[42], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[43], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[44], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[45], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[46], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[47], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[48], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[49], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[50], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[51], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[52], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[53], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[54], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[55], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[56], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[57], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[58], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[59], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[60], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[61], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[62], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[63], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[64], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[65], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[66], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[67], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[68], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[69], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[70], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[71], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[72], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[73], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[74], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[75], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[76], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[77], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[78], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[79], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[80], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[81], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[82], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[83], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[84], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[85], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[86], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[87], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[88], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[89], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[90], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[91], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[92], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[93], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[94], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[95], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[96], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[97], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[98], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[99], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[100], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[101], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[102], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[103], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[104], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[105], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[106], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[107], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[108], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[109], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[110], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[111], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[112], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[113], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[114], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[115], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[116], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[117], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[118], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[119], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[120], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[121], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[122], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[123], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[124], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[125], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[126], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[127], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[128], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[129], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[130], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[131], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[132], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[133], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[134], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[135], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[136], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[137], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[138], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[139], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[140], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[141], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[142], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[143], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[144], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[145], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[146], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[147], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[148], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[149], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[150], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[151], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[152], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[153], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[154], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[155], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[156], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[157], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[158], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[159], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[160], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[161], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[162], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[163], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[164], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[165], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[166], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[167], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[168], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[169], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[170], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[171], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[172], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[173], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[174], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[175], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[176], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[177], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[178], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[179], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[180], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[181], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[182], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[183], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[184], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[185], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[186], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[187], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[188], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[189], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[190], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[191], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[192], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[193], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[194], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[195], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[196], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[197], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[198], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[199], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[200], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[201], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[202], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[203], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[204], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[205], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[206], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[207], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[208], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[209], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[210], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[211], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[212], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[213], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[214], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[215], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[216], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[217], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[218], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[219], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[220], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[221], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[222], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[223], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[224], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[225], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[226], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[227], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[228], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[229], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[230], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[231], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[232], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[233], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[234], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[235], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[236], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[237], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[238], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[239], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[240], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[241], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[242], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[243], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[244], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[245], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[246], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[247], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[248], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[249], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[250], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[251], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[252], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[253], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[254], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[255], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[256], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[257], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[258], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[259], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[260], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[261], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[262], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[263], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[264], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[265], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[266], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[267], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[268], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[269], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[270], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[271], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[272], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[273], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[274], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[275], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[276], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[277], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[278], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[279], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[280], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[281], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[282], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[283], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[284], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[285], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[286], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[287], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[288], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[289], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[290], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[291], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[292], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[293], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[294], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[295], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[296], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[297], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[298], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[299], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[300], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[301], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[302], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[303], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[304], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[305], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[306], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[307], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[308], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[309], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[310], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[311], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[312], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[313], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[314], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[315], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[316], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[317], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[318], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[319], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[320], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[321], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[322], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[323], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[324], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[325], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[326], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[327], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[328], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[329], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[330], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[331], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[332], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[333], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[334], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[335], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[336], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[337], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[338], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[339], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[340], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[341], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[342], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[343], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[344], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[345], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[346], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[347], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[348], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[349], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[350], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[351], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[352], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[353], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[354], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[355], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[356], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[357], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[358], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[359], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[360], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[361], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[362], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[363], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[364], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[365], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[366], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[367], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[368], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[369], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[370], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[371], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[372], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[373], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[374], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[375], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[376], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[377], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[378], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[379], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[380], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[381], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[382], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[383], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[384], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[385], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[386], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[387], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[388], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[389], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[390], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[391], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[392], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[393], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[394], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[395], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[396], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[397], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[398], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[399], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[400], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[401], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[402], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[403], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[404], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[405], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[406], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[407], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[408], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[409], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[410], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[411], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[412], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[413], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[414], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[415], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[416], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[417], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[418], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[419], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[420], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[421], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[422], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[423], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[424], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[425], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[426], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[427], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[428], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[429], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[430], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[431], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[432], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[433], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[434], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[435], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[436], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[437], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[438], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[439], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[440], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[441], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[442], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[443], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[444], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[445], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[446], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[447], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[448], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[449], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[450], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[451], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[452], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[453], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[454], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[455], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[456], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[457], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[458], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[459], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[460], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[461], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[462], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[463], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[464], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[465], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[466], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=467; ptr<467 && i<468; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<467 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<467; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<2,3>(const double *x, const double y, double *r){
	double f[3];
	int ptr=0;
	double pr = two_sum(x[1], y, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<4,3>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], x[ptr+1]);

	for(int i=ptr+1; i<3; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<4,3>(const double *x, double *r){
	double f[4];
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<4; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}
#endif
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<2,1>(const double *x, const double y, double *r){
	double f[3];
	int ptr=0;
	double pr = two_sum(x[1], y, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=1; ptr<1 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<1 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<1; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<2,1>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[0], x[1], x[1]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[1], x[ptr+1]);

	for(int i=ptr+1; i<1; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<2,1>(const double *x, double *r){
	double f[2];
	int ptr=0;
	double pr = fast_two_sum(x[0], x[1], f[1]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }

	for(int i=1; ptr<1 && i<2; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<1 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<1; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<2,3>(const double *x, const double y, double *r){
	double f[3];
	int ptr=0;
	double pr = two_sum(x[1], y, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<4,3>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], x[ptr+1]);

	for(int i=ptr+1; i<3; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<4,3>(const double *x, double *r){
	double f[4];
	int ptr=0;
	double pr = fast_two_sum(x[2], x[3], f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=3; ptr<3 && i<4; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<3 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<3; i++) r[i] = 0.;
}
#endif
template<>
__host__ __device__ __forceinline__ void renorm2L_4Add1<2,4>(const double *x, const double y, double *r){
#if 0
	double f[3];
#else
	double f[4];
#endif
	int ptr=0;
	double pr = two_sum(x[1], y, f[2]);
	f[0] = two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=4; ptr<4 && i<3; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<4 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<4; i++) r[i] = 0.;
}
#if 0
template<>
__host__ __device__ __forceinline__ void fast_renorm2L<5,4>(double *x){
	int ptr=0;
	double pr = fast_two_sum(x[3], x[4], x[4]);
	pr = fast_two_sum(x[2], pr, x[3]);
	pr = fast_two_sum(x[1], pr, x[2]);
	x[0] = fast_two_sum(x[0], pr, x[1]);

	if(x[1] == 0.) pr = x[0];
	else { pr = x[1]; ptr++; }
	x[ptr] = fast_two_sum(pr, x[2], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[3], pr);
	if(pr == 0.) pr = x[ptr]; else ptr++;
	x[ptr] = fast_two_sum(pr, x[4], x[ptr+1]);

	for(int i=ptr+1; i<4; i++) x[i] = 0.;
}

template<>
__host__ __device__ __forceinline__ void fast_renorm2L<5,4>(const double *x, double *r){
	double f[5];
	int ptr=0;
	double pr = fast_two_sum(x[3], x[4], f[4]);
	pr = fast_two_sum(x[2], pr, f[3]);
	pr = fast_two_sum(x[1], pr, f[2]);
	f[0] = fast_two_sum(x[0], pr, f[1]);

	if(f[1] == 0.) pr = f[0];
	else { r[0] = f[0]; pr = f[1]; ptr++; }
	r[ptr] = fast_two_sum(pr, f[2], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;
	r[ptr] = fast_two_sum(pr, f[3], pr);
	if(pr == 0.) pr = r[ptr]; else ptr++;

	for(int i=4; ptr<4 && i<5; i++){
		r[ptr] = fast_two_sum(pr, f[i], pr);
		if(pr == 0.) pr = r[ptr]; else ptr++;
	}

	if(ptr<4 && pr!=0.){ r[ptr] = pr; ptr++; }
	for(int i=ptr; i<4; i++) r[i] = 0.;
}
#endif
