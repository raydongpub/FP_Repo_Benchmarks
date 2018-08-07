/********************************************************
 * I would run md simulations with the best versions from 
 * the synthetic code.
 *
 * The most correct algorithm seems to be the avg_div
 * The fastest algorithm seems to be the newton_raphson
 * The simplest algorithm is the tricky_div 
 * Each of these has a description before the function
 *
 * Its like a horse race or something, which will perform the best?
 ********************************************************/

#ifndef MYLIB
#define MYLIB

// INCLUDES {{{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "real.cu"
// }}}
// MACROS {{{
#define ABS_TOL .000001
#define REL_TOL .000001
#define HARDWARE_ERROR_PREC .5
#define MAX_ITER 10
#define ABS(x) (((x) < 0) ? (-(x)) : (x))
#define NaN (0.0/0.0)
// }}}
// PROTOS {{{
/*
__device__ real my_sin(real);
__device__ real my_asin(real);
__device__ real my_cos(real);
__device__ real my_acos(real);
__device__ real my_sqrt(real);
*/
// }}}

__device__ real my_sqrt(real x) { // {{{
	real retval, absErr, relErr, lastGuess;
	const real absTol = 1e-5;
	const real relTol = 1e-3;

	if(x<=0) return (real)(0.0);

#ifdef DOUBLE_PRECISION
	retval = sqrt(x);
#else
	retval = sqrtf(x);
#endif
	if(isinf(retval) || isnan(retval)) return retval;
	absErr = retval*retval - x; if(absErr<0) absErr = -absErr;
	if(absErr<=absTol) return retval;

	for(;;) {
		lastGuess = retval;
		retval -= (retval*retval-x)/(2*retval);
		absErr = retval*retval - x ; if(absErr<0) absErr = -absErr;
		relErr = lastGuess - retval; if(relErr<0) relErr = -relErr;
		if(absErr<=absTol || relErr<=relTol) return retval;
	}
} // }}}
__device__ real omar_div(real x, real y) { // {{{
	__shared__ real retval;
	__shared__ real xy;
	xy = __fdividef(x,y);
	retval = xy;

	retval = __fmul_rn(0.5, __fadd_rn(xy, retval));
	return retval;
} // }}}
__device__ float my_newton_div(real x, real y) { // {{{
/*
  float2 div
  Uses float2 with the second float being the error
 */

/*
  Newton Rhapson iterations 
  VERY fast, faster than the hardware division
  However, as of right now, it still drifts
*/
  float r;
  double rd;
  r = __fdividef(1, y);
  rd = (double)r;
  rd *= ((double)2.0 - ((double)y * rd));
 
  return rd * x;

} // }}}
real my_newton_cpu_div(real x, real y) { // {{{
  //unsigned int l1;
  real r;
  
  r = (1.0f/y);
  r *= 2 - (r*y);
  r *= x;
  
  //l1 = (((unsigned int *)(&r))[0] & 0x0000000f);
	
  
  return r;
	
} // }}}
__device__ real my_trick_div(real x, real y) { // {{{
/*
  My trick division uses the idea from avg division
  Just add 1 then subtract 1 from x/y. This fixes the drifting in the
  synthetic code. 
  Is as fast as hardware division.
  
  I have no idea why this fixes the drifting.
  Is there something going on in the hardware behind the scenes
  or is this a bug in the synthetic code? 

*/
  return __fadd_rn(__fadd_rn(__fdividef(x, y), 1),-1); 
} // }}}
__device__ real my_avg_div(real x, real y) { // {{{
/*
  My avg division calculates x/y, then takes the average between a lower
  and upper bound.
  Decently fast, about 4 times as slow as hardware, but fixes the drifting
  
  It seems to me that we are calculating x/y, then symmetrically making the 
  bounds, and picking the mid-point, which should return the original x/y.
  
  However, for all values i can test with, this seems to be correct.

 */
  int xisNegative;
  int yisNegative;
  int i;
  real lower_bound;
  real upper_bound;
  real guess;
 

  i = x < 0;
  xisNegative = (1 - i - i);
  x = __fmul_rn(x, xisNegative);
  
  i = y < 0;
  yisNegative = (1 - i - i);
  y = __fmul_rn(y, yisNegative);
  
  guess = x/y;

  
  upper_bound = __fmul_rn(guess, (1 + HARDWARE_ERROR_PREC));
  lower_bound = __fmul_rn(guess, (1 - HARDWARE_ERROR_PREC));
  
   guess = lower_bound + (.5 * (upper_bound - lower_bound));
  
    return __fmul_rn(__fmul_rn(guess, yisNegative), xisNegative);

} // }}}
__device__ real my_root_div_with_ifs(real x, real y) { // {{{
/*
  A root finding algorithm that makes its original guess at x/y

  x/y = z --> x = z*y, therefore we know a way to check if our guess, z,
  is correct. Adjust boundaries as needed and repeat.

  Uses if statements and is very slow. However, this also fixes the drift.
 */
 int isNegative = 0;
  int i;
  float lower_bound;
  float upper_bound;
  float guess;
  float prev_guess;
  float abs_error;
  float rel_error;
  float solution;
  
  if(x < 0){
    x *= -1;
    isNegative = !isNegative;
  }
  if(y < 0){
    y *= -1;
    isNegative = !isNegative;
  }
  
  guess = x/y;

  
  upper_bound = __fmul_rn(guess, (1 + HARDWARE_ERROR_PREC));
  lower_bound = __fmul_rn(guess, (1 - HARDWARE_ERROR_PREC));
  
  guess = lower_bound + (.5 * (upper_bound - lower_bound));
  
  solution = guess * y;
  abs_error = solution - x; 
  if(ABS(abs_error) <= ABS_TOL){
    if(isNegative && guess > 0)
      return guess * -1;
    else
      return guess;
  }
  
  for(i = 0; i < MAX_ITER; i++){
    prev_guess = guess;
    if(solution < x) lower_bound = guess;
    else             upper_bound = guess;

    guess = lower_bound + (.5 * (upper_bound - lower_bound));
    solution = guess * y;
    abs_error = ABS(solution - x);
    rel_error = ABS((prev_guess - guess)/prev_guess);
    if(abs_error <= ABS_TOL || rel_error <= REL_TOL){
      if(isNegative && guess > 0) {
		  return guess * -1;
	  } else {
		  return guess;
	  }
    }

  }
  if(isNegative && guess > 0)
    return guess * -1;
     
  return guess;

} // }}}
__device__ real my_root_div(real x, real y) { // {{{
/*
 Same algorithm as root div, but without any ifs. Still drifts, however.
 */
  int xisNegative;
  int yisNegative;
  int i;
  real lower_bound;
  real upper_bound;
  real guess;
  real solution;
  real prev_guess;
  real abs_error;
  real rel_error;
   real tempx;
   real tempy;

  tempx = x;
  tempy = y;

  i = x < 0;
  xisNegative = (1 - i - i);
  tempx = __fmul_rn(x, xisNegative);
  
  i = y < 0;
  yisNegative = (1 - i - i);
  tempy = __fmul_rn(y, yisNegative);
  
  guess = 1 / tempy;
  guess = __fmul_rn(guess, tempx);
  
  upper_bound = __fmul_rn(guess, (1 + HARDWARE_ERROR_PREC));
  lower_bound = __fmul_rn(guess, (1 - HARDWARE_ERROR_PREC));
  // guess = lower_bound + (.5 * (upper_bound - lower_bound));
  
  solution = __fmul_rn(guess, y);
  // lower_bound = (solution <= x) * lower_bound + (solution > x) * guess;
  //upper_bound = (solution >= x) * upper_bound + (solution < x) * guess;
  for(i = 0; i <MAX_ITER; i++){
    prev_guess = guess;
    lower_bound = __fadd_rn(__fmul_rn(lower_bound, (solution <= x)) 
			    , __fmul_rn(guess, (solution > x)));
    
    upper_bound = __fadd_rn(__fmul_rn(upper_bound, (solution >= x)) 
			    , __fmul_rn(guess, (solution < x))); 
    
    guess = __fadd_rn(lower_bound, __fmul_rn(.5, (upper_bound - lower_bound)));
    solution = __fmul_rn(guess, y);

    abs_error = ABS(solution - x);
    rel_error = ABS((prev_guess - guess)/prev_guess);
    if(abs_error <= ABS_TOL || rel_error <= REL_TOL){
      return __fmul_rn(__fmul_rn(guess, yisNegative), xisNegative);
    }
  }
  
 
  return __fmul_rn(__fmul_rn(guess, yisNegative), xisNegative);
  
} // }}}
__device__ real gs_div(real x, real y) { // {{{
	real N, D, F;
	N = x;
	D = y;

	N = __fdividef(N, y);
	D = __fdividef(D, y);

	F = __fadd_rn(2, -D);
	N = __fmul_rn(N, F);
	D = __fmul_rn(D, F);

	return N;
} // }}}
__device__ real my_sin(real x) { // {{{
#ifdef DOUBLE_PRECISION
  const real sqrt2 =             sqrt(2);
  
  const real pi4   =  0.7853981633974483;
  const real pi2   =  1.5707963267948966;
  const real pi    =  3.1415926535897932;
  const real _3pi2 =  4.7123889803846899;
  const real _2pi  =  6.2831853071795865;
  
  const real A1    = -0.3535533905932737;
  const real A2    =  0.1178511301977579;
  const real A3    =  0.0294627825494394;
  const real A4    = -0.0058925565098879;
  const real A5    = -0.0009820927516480;
  const real A6    =  0.0001402989645211;
  const real A7    =  0.0000175373705651;
#else
  const real sqrt2 =    sqrtf(2);
  
  const real pi4   =  0.7853982f;
  const real pi2   =  1.5707963f;
  const real pi    =  3.1415927f;
  const real _3pi2 =  4.7123890f;
  const real _2pi  =  6.2831853f;
  
  const real A1    = -0.3535534f;
  const real A2    =  0.1178511f;
  const real A3    =  0.0294628f;
  const real A4    = -0.0058926f;
  const real A5    = -0.0009821f;
  const real A6    =  0.0001403f;
  const real A7    =  0.0000175f;
#endif
  int negative;
  real retval;
  real pi4mx;
  real temp;
  
  negative = x<0;
  if(negative) x = -x;
  x = fmodf(x, _2pi);
  
  if       (   pi2  < x && x <=   pi  ) {
    temp = x-pi2;
    x -= temp+temp;
  } else if(   pi   < x && x <= _3pi2 ) {
    x -= pi;
    negative = !negative;
  } else if( _3pi2  < x && x <= _2pi  ) {
    x -= pi;
    temp = x-pi2;
    x -= temp+temp;
    negative = !negative;
  }
  
  pi4mx = (pi4 - x);
  temp = pi4mx*pi4mx;
  
  retval =  A1*temp; temp *= pi4mx;
  retval += A2*temp; temp *= pi4mx;
  retval += A3*temp; temp *= pi4mx;
  retval += A4*temp; temp *= pi4mx;
  retval += A5*temp; temp *= pi4mx;
  retval += A6*temp; temp *= pi4mx;
  retval += A7*temp + 0.5*sqrt2*(1-pi4mx);
  return negative ? -retval : retval;
} // }}}
__device__ real my_asin(real x) { // {{{
  real a, b, ab, absErr, relErr, lastGuess, msab;
  int isnegative;
  const real absTol = 1e-5;
  const real relTol = 1e-3;
  const real pi2   =  1.5707963267948966;
  
  if(x<=-1) return (-1-x>1e-5) ? (real)(NaN) : -pi2;
  if(x>= 1) return ( x-1>1e-5) ? (real)(NaN) :  pi2;
  if(x==0) return (real)(0.0);
  
  isnegative = x<0;
  if(isnegative) x = -x;
  
#ifdef DOUBLE_PRECISION
  a = asin(x);
#else
  a = asinf(x);
#endif
  
  b  = a*1.1;
  a *= 0.9;
  if(b> pi2) b =  pi2;
  if(a<   0) a =    0;
  ab = a+0.5*(b-a); msab = my_sin(ab);
  absErr = msab - x; if(absErr<0) absErr = -absErr;
  if(absErr<=absTol) return isnegative ? -ab : ab;
  
  for(;;) {
    lastGuess = ab;
    if(msab<x) a = ab;
    else      b = ab;
    ab = a+0.5*(b-a); msab = my_sin(ab);
    
    absErr = msab - x; if(absErr<0) absErr = -absErr;
    relErr = (lastGuess - ab)/lastGuess; if(relErr<0) relErr = -relErr;
    if(absErr<=absTol || relErr<=relTol) return isnegative ? -ab : ab;
  }
} // }}}
real my_cos(real x) { // {{{
#ifdef DOUBLE_PRECISION
  const real sqrt2 =             sqrt(2);
  
  const real pi4   =  0.7853981633974483;
  const real pi2   =  1.5707963267948966;
  const real pi    =  3.1415926535897932;
  const real _3pi2 =  4.7123889803846899;
  const real _2pi  =  6.2831853071795865;
  
  const real A1    = -0.3535533905932737;
  const real A2    =  0.1178511301977579;
  const real A3    =  0.0294627825494394;
  const real A4    = -0.0058925565098879;
  const real A5    = -0.0009820927516480;
  const real A6    =  0.0001402989645211;
  const real A7    =  0.0000175373705651;
#else
  const real sqrt2 =    sqrtf(2);
  
  const real pi4   =  0.7853982f;
  const real pi2   =  1.5707963f;
  const real pi    =  3.1415927f;
  const real _3pi2 =  4.7123890f;
  const real _2pi  =  6.2831853f;
  
  const real A1    = -0.3535534f;
  const real A2    = -0.1178511f;
  const real A3    =  0.0294628f;
  const real A4    =  0.0058926f;
  const real A5    = -0.0009821f;
  const real A6    = -0.0001403f;
  const real A7    =  0.0000175f;
#endif
  int negative;
  real retval;
  real pi4mx;
  real temp;
  
  negative = x<0;
  if(negative) x = -x;
  x = fmodf(x, _2pi);
  
  if       (   pi2  < x && x <=   pi  ) {
    x = pi-x;
    negative = !negative;
  } else if(   pi   < x && x <= _3pi2 ) {
    x -= pi;
    negative = !negative;
  } else if( _3pi2  < x && x <= _2pi  ) {
    x = _2pi-x;
  }
  
  pi4mx = (pi4 - x);
  temp = pi4mx*pi4mx;
  retval =  A1*temp; temp *= pi4mx;
  retval += A2*temp; temp *= pi4mx;
  retval += A3*temp; temp *= pi4mx;
  retval += A4*temp; temp *= pi4mx;
  retval += A5*temp; temp *= pi4mx;
  retval += A6*temp; temp *= pi4mx;
  retval += A7*temp + 0.5*sqrt2*(1+pi4mx);
  return negative ? -retval : retval;
  
} 
real my_acos(real x) { 
  real a, b, ab, absErr, relErr, lastGuess, mcab;
  int isnegative;
  const real absTol = 1e-10;
  const real relTol = 1e-10;
  const real pi    =  3.1415926535897932;
  const real pi2   =  1.5707963267948966;
  
  if(x<=-1) return (-1-x>1e-5) ? (real)(NaN) :        pi;
  if(x>= 1) return ( x-1>1e-5) ? (real)(NaN) : (real)0.0;
  
  isnegative = x<0;
  if(isnegative) x = -x;
  
#ifdef DOUBLE_PRECISION
  a = acos(x);
#else
  a = acosf(x);
#endif
  
  b  = a*1.1;
  a  = a*0.9;
  if(b> pi2) b = pi2;
  if(a<   0) a =   0;
  ab = a+0.5*(b-a); mcab = my_cos(ab);
  absErr = mcab - x; if(absErr<0) absErr = -absErr;
  if(absErr<=absTol) return isnegative ? pi-ab : ab;
  
  for(;;) {
    lastGuess = ab;
    if(mcab<x) b = ab;
    else       a = ab;
    ab = a+0.5*(b-a); mcab = my_cos(ab);
    
    absErr = mcab - x; if(absErr<0) absErr = -absErr;
    relErr = (lastGuess - ab)/lastGuess; if(relErr<0) relErr = -relErr;
    if(absErr<=absTol || relErr<=relTol) return isnegative ? pi-ab : ab;
  }
} // }}}
__device__ float div_recip(float x, float y){

  float recip = 1/y;
  return x*recip;
}

/* float2 code ================================================= {{{ */

/*
  float2 div
  Uses float2 with the second float being the error
 */
__device__ float2 float_to_float2_d(float x){
  float t = x * 4097;
  float2 retval;
  retval.x = t - (t - x);
  retval.y = x - retval.x;

  return retval;
}
__device__ float2 twoProd(float a, float b){
  float p = __fmul_rn(a, b);
  float2 aS = float_to_float2_d(a);
  float2 bS = float_to_float2_d(b);
  float err = ((__fmul_rn(aS.x,bS.x) - p)
	       + __fmul_rn(aS.x,bS.y) + __fmul_rn(aS.y,bS.x))
    + __fmul_rn(aS.y,bS.y);
  float2 retval;
  retval.x = p;
  retval.y = err;
  return retval;
}

__device__ float2 quickTwoSum(float a, float b){

  float s = a+b;
  float e = b-(s-a);
  float2 retval;
  retval.x = s;
  retval.y = e;
  return retval;
}

__device__ float2 my_ext_div_d(float2 x, float2 y){
  
  float2 retval;
  float xn = 1/y.x;
  float yn = __fmul_rn(x.x, xn);
  float fDiff = __fadd_rn(x.x ,-(__fmul_rn(y.x, yn)));
  

  retval.y = __fmul_rn(xn, fDiff);
  retval.x = __fadd_rn(yn, retval.y);

  
  return retval;
}

__device__ float2 my_ext_mul_d(float2 x, float2 y){
    float2 retval;
  retval.x = __fmul_rn(x.x, y.x);
  retval.y = __fadd_rn(__fadd_rn(__fmul_rn(x.x, y.y), __fmul_rn(x.y, y.x)), __fmul_rn(x.y,y.y));

  return retval;
  
}
__device__ float2 mul_d_exact(float2 a, float2 b){
  float2 p;

  p = twoProd(a.x, b.x);
  p.y += a.x*b.y;
  p.y += a.y*b.x;
  p = quickTwoSum(p.x,p.y);

  return p;
}
__device__ float2 my_ext_add_d(float2 x, float2 y){
  float2 retval;
  float temp;
  retval.x = x.x + y.x;
  temp = retval.x - x.x;
  retval.y = (x.x - (retval.x - temp)) + (y.x - temp) + x.y + y.y;

  return retval;
}
__device__ float2 SCS_d(float a, float b){
  float sum = __fadd_rn(a, b);
  float error = __fadd_rn(b, __fadd_rn(a , -sum));
  float2 retval;
  retval.x = sum;
  retval.y = error;

  return retval;
}
__device__ float2 my_ext_diff_d(float2 x, float2 y){
  y.x = -y.x;
  y.y = -y.y;
  return my_ext_add_d(x, y);
}
__host__ float2 float_to_float2(float x) { // {{{
/*
  float2 div
  Uses float2 with the second float being the error
 */
  float t = x * 4097;
  float2 retval;
  retval.x = t - (t - x);
  retval.y = x - retval.x;

  return retval;
} // }}}
 __host__ float2 my_ext_div(float2 x, float2 y) { // {{{
   float2 retval;
  float xn = 1/y.x;
  float yn = x.x* xn;
  float fDiff = x.x - (y.x* yn);
  

  retval.y = xn* fDiff;
  retval.x = yn+retval.y;
  return retval;

} // }}}
__host__ float2 my_ext_mul(float2 x, float2 y) { // {{{
  float2 retval;
  retval.x = x.x*y.x;
  retval.y = x.x*y.y + x.y*y.x;

  return retval;
} // }}}
__host__ float2 my_ext_add(float2 x, float2 y) { // {{{
  float2 retval;
  float temp;
  retval.x = x.x + y.x;
  temp = retval.x - x.x;
  retval.y = x.x - (retval.x - temp) + (y.x - temp) + x.y + y.y;

  return retval;
} // }}}
__host__ double2 SCS(double a, double b){
  double sum = a + b;
  double error = b - (sum - a);
  double2 retval;
  retval.x = sum;
  retval.y = error;

  return retval;
}
__host__ float2 my_ext_diff(float2 x, float2 y) { // {{{
  y.x = -y.x;
  y.y = -y.y;
  return my_ext_add(x, y);
} // }}}
// }}}

/* double2 code ================================================= {{{ */


__device__ double2 double_to_double2_d(double x){ //TODO
  //double t = x * 134217729; //see the docs to have the explanation.
#if 0
  double t = __dmul_rn(x, 134217729); //crosscheck
  double2 retval;
  retval.x = __dsub_rn(t, __dsub_rn(t, x));
  retval.y = __dsub_rn(x, retval.x);
#else 
  double t = x * 134217729; //crosscheck
  double2 retval;
  retval.x = (t - (t - x));
  retval.y = (x - retval.x);

#endif

  return retval;
}
__device__ double2 twoProd(double a, double b){
  double p = __fmul_rn(a, b);
  double2 aS = double_to_double2_d(a);
  double2 bS = double_to_double2_d(b);
  double err = ((__fmul_rn(aS.x,bS.x) - p)
	       + __fmul_rn(aS.x,bS.y) + __fmul_rn(aS.y,bS.x))
    + __fmul_rn(aS.y,bS.y);
  double2 retval;
  retval.x = p;
  retval.y = err;
  return retval;
}

__device__ double2 quickTwoSum(double a, double b){

  double s = a+b;
  double e = b-(s-a);
  double2 retval;
  retval.x = s;
  retval.y = e;
  return retval;
}

__device__ double2 my_ext_div_d(double2 x, double2 y){

  double2 retval;
  double xn = 1/y.x;
  double yn = __dmul_rn(x.x, xn);
  double fDiff = __dadd_rn(x.x ,-(__dmul_rn(y.x, yn)));


  retval.y = __dmul_rn(xn, fDiff);
  retval.x = __dadd_rn(yn, retval.y);


  return retval;
}

__device__ double2 my_ext_mul_d(double2 x, double2 y){
  double2 retval;
  retval.x = __dmul_rn(x.x, y.x);
  retval.y = __dadd_rn(__dadd_rn(__dmul_rn(x.x, y.y), __dmul_rn(x.y, y.x)), __dmul_rn(x.y,y.y));

  return retval;

}
__device__ double2 mul_d_exact(double2 a, double2 b){
  double2 p;

  p = twoProd(a.x, b.x);
  p.y += a.x*b.y;
  p.y += a.y*b.x;
  p = quickTwoSum(p.x,p.y);

  return p;
}
__device__ double2 my_ext_add_d(double2 x, double2 y){
  double2 retval;
  double temp;
  retval.x = x.x + y.x;
  temp = retval.x - x.x;
  retval.y = (x.x - (retval.x - temp)) + (y.x - temp) + x.y + y.y;

  return retval;
}

__device__ double2 SCS_d(double a, double b){
  double sum = __fadd_rn(a, b);
  double error = __fadd_rn(b, __fadd_rn(a , -sum));
  double2 retval;
  retval.x = sum;
  retval.y = error;

  return retval;
}
__device__ double2 my_ext_diff_d(double2 x, double2 y){
  y.x = -y.x;
  y.y = -y.y;
  return my_ext_add_d(x, y);
}

__host__ double2 double_to_double2(double x) { // {{{
/*
  double2 div
  Uses double2 with the second double being the error
 */
  double t = x * 134217729;
  double2 retval;
  retval.x = t - (t - x);
  retval.y = x - retval.x;

  return retval;
} // }}}

 __host__ double2 my_ext_div(double2 x, double2 y) { // {{{
   double2 retval;
  double xn = 1/y.x;
  double yn = x.x* xn;
  double fDiff = x.x - (y.x* yn);


  retval.y = xn* fDiff;
  retval.x = yn+retval.y;
  return retval;

} // }}}
__host__ double2 my_ext_mul(double2 x, double2 y) { // {{{
  double2 retval;
  retval.x = x.x*y.x;
  retval.y = x.x*y.y + x.y*y.x;

  return retval;
} // }}}
__host__ double2 my_ext_add(double2 x, double2 y) { // {{{
  double2 retval;
  double temp;
  retval.x = x.x + y.x;
  temp = retval.x - x.x;
  retval.y = x.x - (retval.x - temp) + (y.x - temp) + x.y + y.y;

  return retval;
} // }}}

__host__ double2 my_ext_diff(double2 x, double2 y) { // {{{
  y.x = -y.x;
  y.y = -y.y;
  return my_ext_add(x, y);
} // }}}
// }}}

#endif

