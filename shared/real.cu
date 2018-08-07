
#ifndef _REAL_CU
#define _REAL_CU

#ifdef DOUBLE_PRECISION
  typedef double real;
  #define FLOAT_FORMAT "%lf"
#else
  typedef float real;
  #define FLOAT_FORMAT "%f"
#endif

#endif

