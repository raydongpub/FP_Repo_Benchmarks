// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

/*
Modified in 2018 
Paul A. Beata
North Carolina State University

Starting from the original code provided for double-precision
computations, we have adapted to include floating-point
arithmetic using the CAMPARY library: multi_prec variables
and certified operations. 
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <omp.h>
#include <iomanip>
#include <fenv.h>

// Include the tuner file that contains access to our 
// CAMPARY functions and data types, and GNU MPFR too:
#include <mpfr.h>
#include "../shared/tuner_cpu.h"

// 3D data structures:
struct campary3 { multi_prec<CPR> x, y, z; };

#ifndef block_length
#error "you need to define block_length"
#endif

/*
// ==============================================
// Tuning iterations are declared during "make":
//   (example:  $ make TUNE=LSQR3)
// I. outside loops: "O" + {op}
#define OSQR
#define ODIV
#define OADD
#define OMUL  // AD = 2
// II. inside loops: "L" + {op} + {loop_id}
  // Loop #1
  #define LDIV1
  #define LMUL1
  #define LADD1
  #define LSQR1
  // Loop #3
  #define LSQR3
  // Loop #5
  #define LDIV5
  #define LMUL5
  #define LADD5
  // Loop #4
  #define LSQR4
  #define LMUL4
  #define LADD4 // AD = 1
// III. shared functions: "F" + {op} + {func_id}
  // Func #5
  #define FADD5
  #define FMUL5
  // Func #1
  #define FDIV1
  // Func #4
  #define FSQR4
  #define FDIV4
  #define FMUL4
  // Func #2
  #define FMUL2
  #define FADD2 // AD = 1
  // Func #3 
  #define FADD3
  #define FMUL3 // AD = 2
// ==============================================
*/

/*
 * Options
 *
 */
// NOTE: The integer "iterations" was replaced 
// by a command line argument now:
/*
#define iterations 2000
*/
#define GAMMA 1.4
#define NDIM 3
#define NNB 4

#define RK 3  // 3rd order RK
#define ff_mach 1.2
// NOTE: we assume angle of attack remains 0.0
#define deg_angle_of_attack 0.0

/*
 * not options
 */
#define GMP 256
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


// check tuning
void check_tuning()
{
#ifdef OSQR
  std::cout << "iteration = 1\n";
#endif
#ifdef ODIV
  std::cout << "iteration = 2\n";
#endif
#ifdef OADD
  std::cout << "iteration = 3\n";
#endif
#ifdef OMUL
  std::cout << "iteration = 4\n";
#endif  
}


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
  return new T[N];
}

template <typename T>
void dealloc(T* array)
{
  delete[] array;
}

template <typename T>
void copy(T* dst, T* src, int N)
{
  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < N; i++)
  {
    dst[i] = src[i];
  }
}

void init_gmp(mpf_t * A, int n)
{
  for (int i = 0; i < n; ++i)
    mpf_init2(A[i], GMP);
}


/*
 * [CAMPARY] Element-based Cell-centered FVM solver functions 
 */
multi_prec<CPR> gamma_cmp, gamma_minus_one;
multi_prec<CPR> one, half;
multi_prec<CPR> ff_variable_cmp[NVAR];
campary3 ff_flux_contribution_momentum_x_cmp;
campary3 ff_flux_contribution_momentum_y_cmp;
campary3 ff_flux_contribution_momentum_z_cmp;
campary3 ff_flux_contribution_density_energy_cmp;

void initialize_variables_cmp(int nelr, multi_prec<CPR>* variables)
{
  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    for(int j = 0; j < NVAR; j++) variables[i*NVAR + j] = ff_variable_cmp[j];
  }
}

// FUNCTION TUNING
inline void compute_flux_contribution_cmp(
  multi_prec<CPR> & density, 
  campary3 & momentum, 
  multi_prec<CPR> & density_energy, 
  multi_prec<CPR> & pressure, 
  campary3 & velocity, 
  campary3 & fc_momentum_x, 
  campary3 & fc_momentum_y, 
  campary3 & fc_momentum_z, 
  campary3 & fc_density_energy)
{
  multi_prec<CPR> tmp, de_p;

#ifdef FMUL5
  mul_CMPToD(&tmp, &velocity.x, &momentum.x);
#else
  certifMulExpans(tmp, velocity.x, momentum.x);
#endif
#ifdef FADD5
  add_CMPToD(&fc_momentum_x.x, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_x.x,  tmp,  pressure);
#endif
#ifdef FMUL5
  mul_CMPToD(&fc_momentum_x.y, &velocity.x, &momentum.y);
#else
  certifMulExpans(fc_momentum_x.y, velocity.x, momentum.y);
#endif
#ifdef FMUL5
  mul_CMPToD(&fc_momentum_x.z, &velocity.x, &momentum.z);
#else
  certifMulExpans(fc_momentum_x.z, velocity.x, momentum.z);
#endif

  fc_momentum_y.x = fc_momentum_x.y;
#ifdef FMUL5
  mul_CMPToD(&tmp, &velocity.y, &momentum.y);
#else  
  certifMulExpans(tmp, velocity.y, momentum.y);
#endif
#ifdef FADD5
  add_CMPToD(&fc_momentum_y.y, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_y.y,  tmp,  pressure);
#endif
#ifdef FMUL5
  mul_CMPToD(&fc_momentum_y.z, &velocity.y, &momentum.z);
#else
  certifMulExpans(fc_momentum_y.z, velocity.y, momentum.z);
#endif  

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
#ifdef FMUL5  
  mul_CMPToD(&tmp, &velocity.z, &momentum.z);
#else
  certifMulExpans(tmp, velocity.z, momentum.z);
#endif
#ifdef FADD5
  add_CMPToD(&fc_momentum_z.z, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_z.z,  tmp,  pressure);
#endif

#ifdef FADD5
  add_CMPToD(&de_p, &density_energy, &pressure);
#else
  certifAddExpans(de_p,  density_energy,  pressure);
#endif
#ifdef FMUL5  
  mul_CMPToD(&fc_density_energy.x, &velocity.x, &de_p);
#else
  certifMulExpans(fc_density_energy.x, velocity.x, de_p);
#endif
#ifdef FMUL5  
  mul_CMPToD(&fc_density_energy.y, &velocity.y, &de_p);
#else
  certifMulExpans(fc_density_energy.y, velocity.y, de_p);
#endif
#ifdef FMUL5
  mul_CMPToD(&fc_density_energy.z, &velocity.z, &de_p);
#else
  certifMulExpans(fc_density_energy.z, velocity.z, de_p);  
#endif
}

inline void compute_velocity_cmp(
  multi_prec<CPR>& density, 
  campary3& momentum, 
  campary3& velocity)
{
#ifdef FDIV1
  div_CMPToD(&velocity.x, &momentum.x, &density);
#else
  divExpans(velocity.x, momentum.x, density);
#endif  
#ifdef FDIV1
  div_CMPToD(&velocity.y, &momentum.y, &density);
#else
  divExpans(velocity.y, momentum.y, density);
#endif  
#ifdef FDIV1  
  div_CMPToD(&velocity.z, &momentum.z, &density);
#else
  divExpans(velocity.z, momentum.z, density);  
#endif
}

// FUNCTION TUNING
inline multi_prec<CPR> compute_speed_sqd_cmp(campary3& velocity)
{
  multi_prec<CPR> result, tmp;
  initFromDouble(&result, 0.0);

#ifdef FMUL2
  mul_CMPToD(&tmp, &velocity.x, &velocity.x);
#else
  certifMulExpans(tmp, velocity.x, velocity.x);
#endif  
#ifdef FADD2
  add_CMPToD(&result, &result, &tmp);
#else
  certifAddExpans(result, result,  tmp);
#endif
#ifdef FMUL2  
  mul_CMPToD(&tmp, &velocity.y, &velocity.y);
#else
  certifMulExpans(tmp, velocity.y, velocity.y);
#endif    
#ifdef FADD2  
  add_CMPToD(&result,&result,&tmp);
#else
  certifAddExpans(result, result, tmp);
#endif
#ifdef FMUL2  
  mul_CMPToD(&tmp, &velocity.z, &velocity.z);
#else
  certifMulExpans(tmp, velocity.z, velocity.z);
#endif    
#ifdef FADD2  
  add_CMPToD(&result,&result,&tmp);
#else
  certifAddExpans(result, result, tmp);
#endif

  return result;
}

// FUNCTION TUNING
inline multi_prec<CPR> compute_pressure_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& density_energy, 
  multi_prec<CPR>& speed_sqd)
{
  multi_prec<CPR> tmp1, tmp2, result;

#ifdef FMUL3  
  mul_CMPToD(&tmp1, &half, &density);
#else
  certifMulExpans(tmp1, half, density);
#endif  
#ifdef FMUL3    
  mul_CMPToD(&tmp2, &tmp1, &speed_sqd);
#else
  certifMulExpans(tmp2, tmp1, speed_sqd);
#endif
#ifdef FADD3
  tmp2 = -tmp2;
  add_CMPToD(&tmp1, &density_energy, &tmp2);
#else
  certifAddExpans(tmp1, density_energy, -tmp2);
#endif
#ifdef FMUL3  
  mul_CMPToD(&result, &gamma_minus_one, &tmp1);
#else
  certifMulExpans(result, gamma_minus_one, tmp1);
#endif

  return result;
}

// FUNCTION TUNING
inline multi_prec<CPR> compute_speed_of_sound_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& pressure)
{
  multi_prec<CPR> tmp1, tmp2, result;
#ifdef FMUL4 
  mul_CMPToD(&tmp1, &gamma_cmp, &pressure);
#else
  certifMulExpans(tmp1, gamma_cmp, pressure);
#endif
#ifdef FDIV4  
  div_CMPToD(&tmp2, &tmp1, &density);
#else
  divExpans(tmp2, tmp1, density);
#endif  
#ifdef FSQR4
  sqrt_CMPToD(&result, &tmp2);
#else
  sqrtNewtonExpans(result, tmp2);
#endif

  return result;
}

void compute_step_factor_cmp(
  int nelr, 
  multi_prec<CPR>* variables, 
  multi_prec<CPR>* areas, 
  multi_prec<CPR>* step_factors)
{
  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    multi_prec<CPR> density;
    density = variables[NVAR*i + VAR_DENSITY];

    campary3 momentum;
    momentum.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
    momentum.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
    momentum.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

    multi_prec<CPR> density_energy;
    density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];

    campary3 velocity;
    compute_velocity_cmp(density, momentum, velocity);

    multi_prec<CPR> speed_sqd;
    speed_sqd = compute_speed_sqd_cmp(velocity);

    multi_prec<CPR> pressure;
    pressure = compute_pressure_cmp(density, density_energy, speed_sqd);

    multi_prec<CPR> speed_of_sound;
    speed_of_sound = compute_speed_of_sound_cmp(density, pressure);

    multi_prec<CPR> tmp1, tmp2, tmp3, denom;
    
#ifdef LSQR1
    sqrt_CMPToD(&tmp1, &speed_sqd);
#else
    sqrtNewtonExpans(tmp1, speed_sqd);
#endif

#ifdef LADD1
    add_CMPToD(&tmp2,&tmp1,&speed_of_sound);
#else
    certifAddExpans(tmp2, tmp1, speed_of_sound);
#endif    
    
#ifdef LSQR1
    sqrt_CMPToD(&tmp3, &areas[i]);
#else
    sqrtNewtonExpans(tmp3, areas[i]);
#endif

#ifdef LMUL1
    mul_CMPToD(&denom, &tmp3, &tmp2);
#else
    certifMulExpans(denom, tmp3, tmp2);
#endif

#ifdef LDIV1
    div_CMPToD(&step_factors[i], &half, &denom);
#else
    divExpans(step_factors[i], half, denom);
#endif    
  }
}

inline multi_prec<CPR> compute_normal_len_cmp(campary3& normal)
{
  multi_prec<CPR> result, tmp;

  initFromDouble(&tmp, 0.0);
  initFromDouble(&result, 0.0);

#ifdef LMUL4 
  mul_CMPToD(&result, &normal.x, &normal.x);
#else
  certifMulExpans(result, normal.x, normal.x);
#endif  
#ifdef LADD4  
  add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

#ifdef LMUL4
  mul_CMPToD(&result, &normal.y, &normal.y);
#else
  certifMulExpans(result, normal.y, normal.y);
#endif  
#ifdef LADD4   
  add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

#ifdef LMUL4
  mul_CMPToD(&result, &normal.z, &normal.z);
#else
  certifMulExpans(result, normal.z, normal.z);
#endif  
#ifdef LADD4    
  add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

#ifdef LSQR4
  sqrt_CMPToD(&result, &tmp);
#else  
  sqrtNewtonExpans(result, tmp);
#endif

  return result;
}

// This function provides:
//      a += b * (c + d)  [default]
//  or  a += b * (c - d)  [subtract_flag = 1]
void accumulate_cmp(
  multi_prec<CPR> &a,
  multi_prec<CPR> &b,
  multi_prec<CPR> &c,
  multi_prec<CPR> &d,
  int subtract_flag = 0)
{
  multi_prec<CPR> tmp1, tmp2;

  if (subtract_flag == 1)
  {
#ifdef LADD4
    // certifMulExpans(tmp2, d, -one);
    tmp2 = -d;
    add_CMPToD(&tmp1, &c, &tmp2);
#else
    certifAddExpans(tmp1, c, -d);
#endif    
  }
  else
  {
#ifdef LADD4
    add_CMPToD(&tmp1, &c, &d);
#else
    certifAddExpans(tmp1, c, d);
#endif    
  }

#ifdef LMUL4
  mul_CMPToD(&tmp2, &b, &tmp1);
#else
  certifMulExpans(tmp2, b, tmp1);
#endif  
#ifdef LADD4
  add_CMPToD(&tmp1, &a, &tmp2);
#else
  certifAddExpans(tmp1, a, tmp2);
#endif  

  a = tmp1;
}

void compute_flux_cmp(
  int nelr, 
  int* elements_surrounding_elements,
  multi_prec<CPR>* normals, 
  multi_prec<CPR>* variables, 
  multi_prec<CPR>* fluxes)
{
  multi_prec<CPR> smoothing_coefficient;
  initFromDouble(&smoothing_coefficient, 0.2);

  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    int j, nb;
    campary3 normal; 
    multi_prec<CPR> normal_len, factor;
    multi_prec<CPR> tmp1, tmp2, tmp3;

    multi_prec<CPR> density_i;
    density_i = variables[NVAR*i + VAR_DENSITY];

    campary3 momentum_i;
    momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
    momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
    momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

    multi_prec<CPR> density_energy_i;
    density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

    campary3 velocity_i;
    compute_velocity_cmp(density_i, momentum_i, velocity_i);

    multi_prec<CPR> speed_sqd_i;
    speed_sqd_i = compute_speed_sqd_cmp(velocity_i);

    multi_prec<CPR> speed_i;
#ifdef LSQR3
    sqrt_CMPToD(&speed_i, &speed_sqd_i);
#else
    sqrtNewtonExpans(speed_i, speed_sqd_i);
#endif

    multi_prec<CPR> pressure_i;
    pressure_i = compute_pressure_cmp(density_i, density_energy_i, speed_sqd_i);

    multi_prec<CPR> speed_of_sound_i;
    speed_of_sound_i = compute_speed_of_sound_cmp(density_i, pressure_i);

    campary3 flux_contribution_i_momentum_x, 
            flux_contribution_i_momentum_y, 
            flux_contribution_i_momentum_z,
            flux_contribution_i_density_energy;

    compute_flux_contribution_cmp(            
      density_i, 
      momentum_i, 
      density_energy_i, 
      pressure_i, 
      velocity_i, 
      flux_contribution_i_momentum_x, 
      flux_contribution_i_momentum_y, 
      flux_contribution_i_momentum_z, 
      flux_contribution_i_density_energy);

    multi_prec<CPR> flux_i_density;
    initFromDouble(&flux_i_density, 0.0);

    campary3 flux_i_momentum;
    initFromDouble(&flux_i_momentum.x, 0.0);
    initFromDouble(&flux_i_momentum.y, 0.0);
    initFromDouble(&flux_i_momentum.z, 0.0);

    multi_prec<CPR> flux_i_density_energy;
    initFromDouble(&flux_i_density_energy, 0.0);

    campary3 velocity_nb;
    multi_prec<CPR> density_nb, density_energy_nb;
    
    campary3  momentum_nb,
              flux_contribution_nb_momentum_x, 
              flux_contribution_nb_momentum_y, 
              flux_contribution_nb_momentum_z,
              flux_contribution_nb_density_energy;

    multi_prec<CPR> speed_sqd_nb, speed_of_sound_nb, pressure_nb;

    for(j = 0; j < NNB; j++)
    {
      nb = elements_surrounding_elements[i*NNB + j];
      normal.x = normals[(i*NNB + j)*NDIM + 0];
      normal.y = normals[(i*NNB + j)*NDIM + 1];
      normal.z = normals[(i*NNB + j)*NDIM + 2];
      normal_len = compute_normal_len_cmp(normal);

      if(nb >= 0)   // a legitimate neighbor
      {
        density_nb =        variables[nb*NVAR + VAR_DENSITY];
        momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
        momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
        momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
        density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];

        compute_velocity_cmp(density_nb, momentum_nb, velocity_nb);
        speed_sqd_nb = compute_speed_sqd_cmp(velocity_nb);
        pressure_nb = compute_pressure_cmp(density_nb, density_energy_nb, speed_sqd_nb);
        speed_of_sound_nb = compute_speed_of_sound_cmp(density_nb, pressure_nb);

        compute_flux_contribution_cmp(
          density_nb, 
          momentum_nb, 
          density_energy_nb, 
          pressure_nb, 
          velocity_nb, 
          flux_contribution_nb_momentum_x, 
          flux_contribution_nb_momentum_y, 
          flux_contribution_nb_momentum_z, 
          flux_contribution_nb_density_energy);

        // artificial viscosity
#ifdef LSQR4 
        sqrt_CMPToD(&tmp1, &speed_sqd_nb);
#else        
        sqrtNewtonExpans(tmp1, speed_sqd_nb);
#endif
#ifdef LADD4
        add_CMPToD(&tmp2, &speed_i, &tmp1);
#else
        certifAddExpans(tmp2, speed_i, tmp1);
#endif        
#ifdef LADD4
        add_CMPToD(&tmp2, &tmp2, &speed_of_sound_i);
#else
        certifAddExpans(tmp2, tmp2, speed_of_sound_i);
#endif
#ifdef LADD4     
        add_CMPToD(&tmp2, &tmp2, &speed_of_sound_nb);
#else
        certifAddExpans(tmp2, tmp2, speed_of_sound_nb);
#endif
#ifdef LMUL4
        mul_CMPToD(&tmp1, &normal_len, &smoothing_coefficient);
#else
        certifMulExpans(tmp1, normal_len, smoothing_coefficient);
#endif
#ifdef LMUL4
        mul_CMPToD(&tmp3, &tmp1, &half);
        tmp3 = -tmp3;
#else
        certifMulExpans(tmp3, tmp1, -half);
#endif
#ifdef LMUL4
        mul_CMPToD(&factor, &tmp3, &tmp2);
#else
        certifMulExpans(factor, tmp3, tmp2);
#endif
        accumulate_cmp(flux_i_density, factor, density_i, density_nb, 1);
        accumulate_cmp(flux_i_density_energy, factor, density_energy_i, density_energy_nb, 1);
        accumulate_cmp(flux_i_momentum.x, factor, momentum_i.x, momentum_nb.x, 1);
        accumulate_cmp(flux_i_momentum.y, factor, momentum_i.y, momentum_nb.y, 1);
        accumulate_cmp(flux_i_momentum.z, factor, momentum_i.z, momentum_nb.z, 1);

        // accumulate cell-centered fluxes
#ifdef LMUL4
        mul_CMPToD(&factor, &half, &normal.x);
#else
        certifMulExpans(factor, half, normal.x);
#endif
        accumulate_cmp(flux_i_density, factor, momentum_nb.x, momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.x, flux_contribution_i_momentum_z.x);

#ifdef LMUL4
        mul_CMPToD(&factor, &half, &normal.y);
#else
        certifMulExpans(factor, half, normal.y);
#endif
        accumulate_cmp(flux_i_density, factor, momentum_nb.y, momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.y, flux_contribution_i_momentum_z.y);        

#ifdef LMUL4
        mul_CMPToD(&factor, &half, &normal.z);
#else
        certifMulExpans(factor, half, normal.z);
#endif
        accumulate_cmp(flux_i_density, factor, momentum_nb.z, momentum_i.z);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.z, flux_contribution_i_density_energy.z);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.z, flux_contribution_i_momentum_x.z);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.z, flux_contribution_i_momentum_y.z);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.z, flux_contribution_i_momentum_z.z);
      }
      else if(nb == -1) // a wing boundary
      {
#ifdef LMUL4
        mul_CMPToD(&tmp1, &pressure_i, &normal.x);
#else
        certifMulExpans(tmp1, pressure_i, normal.x);
#endif
#ifdef LADD4
        add_CMPToD(&flux_i_momentum.x, &flux_i_momentum.x, &tmp1);
#else
        certifAddExpans(flux_i_momentum.x, flux_i_momentum.x, tmp1);
#endif        

#ifdef LMUL4
        mul_CMPToD(&tmp1, &pressure_i, &normal.y);
#else
        certifMulExpans(tmp1, pressure_i, normal.y);
#endif
#ifdef LADD4
        add_CMPToD(&flux_i_momentum.y, &flux_i_momentum.y, &tmp1);
#else
        certifAddExpans(flux_i_momentum.y, flux_i_momentum.y, tmp1);
#endif

#ifdef LMUL4
        mul_CMPToD(&tmp1, &pressure_i, &normal.z);
#else
        certifMulExpans(tmp1, pressure_i, normal.z); 
#endif
#ifdef LADD4
        add_CMPToD(&flux_i_momentum.z, &flux_i_momentum.z, &tmp1);
#else
        certifAddExpans(flux_i_momentum.z, flux_i_momentum.z, tmp1);
#endif
      }
      else if(nb == -2) // a far field boundary
      {
#ifdef LMUL4     
        mul_CMPToD(&factor, &half, &normal.x);
#else
        certifMulExpans(factor, half, normal.x);
#endif
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+0], momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.x, flux_contribution_i_momentum_z.x);        

#ifdef LMUL4
        mul_CMPToD(&factor, &half, &normal.y);
#else
        certifMulExpans(factor, half, normal.y);
#endif
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+1], momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.y, flux_contribution_i_momentum_z.y);        

#ifdef LMUL4
        mul_CMPToD(&factor, &half, &normal.z);
#else
        certifMulExpans(factor, half, normal.z);
#endif
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+2], momentum_i.z);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.z, flux_contribution_i_density_energy.z);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.z, flux_contribution_i_momentum_x.z);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.z, flux_contribution_i_momentum_y.z);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.z, flux_contribution_i_momentum_z.z);
      }
    }

    fluxes[i*NVAR + VAR_DENSITY] = flux_i_density;
    fluxes[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
    fluxes[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
    fluxes[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
    fluxes[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
  }
}

void time_step_cmp(
  int j, 
  int nelr, 
  multi_prec<CPR>* old_variables, 
  multi_prec<CPR>* variables, 
  multi_prec<CPR>* step_factors, 
  multi_prec<CPR>* fluxes)
{
  #pragma omp parallel for  default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    multi_prec<CPR> factor, tmp;
    initFromDouble(&tmp, double(RK+1-j));

#ifdef LDIV5
    div_CMPToD(&factor, &step_factors[i], &tmp);
#else
    divExpans(factor, step_factors[i], tmp);
#endif    
    
#ifdef LMUL5  
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + VAR_DENSITY]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY]);
#endif    
#ifdef LADD5
    add_CMPToD(&variables[NVAR*i + VAR_DENSITY], &old_variables[NVAR*i + VAR_DENSITY], &tmp);
#else
    certifAddExpans(variables[NVAR*i + VAR_DENSITY], old_variables[NVAR*i + VAR_DENSITY], tmp);
#endif    

#ifdef LMUL5
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
#endif
#ifdef LADD5
    add_CMPToD(&variables[NVAR*i + VAR_DENSITY_ENERGY], &old_variables[NVAR*i + VAR_DENSITY_ENERGY], &tmp);
#else
    certifAddExpans(variables[NVAR*i + VAR_DENSITY_ENERGY], old_variables[NVAR*i + VAR_DENSITY_ENERGY], tmp);
#endif

#ifdef LMUL5
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
#endif
#ifdef LADD5
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+0)], &old_variables[NVAR*i + (VAR_MOMENTUM+0)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+0)], old_variables[NVAR*i + (VAR_MOMENTUM+0)], tmp);
#endif

#ifdef LMUL5
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
#endif
#ifdef LADD5
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+1)], &old_variables[NVAR*i + (VAR_MOMENTUM+1)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+1)], old_variables[NVAR*i + (VAR_MOMENTUM+1)], tmp);
#endif

#ifdef LMUL5
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
#endif
#ifdef LADD5
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+2)], &old_variables[NVAR*i + (VAR_MOMENTUM+2)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+2)], old_variables[NVAR*i + (VAR_MOMENTUM+2)], tmp);
#endif
  }
}

void dump_gmp(mpf_t* variables, int nel, int nelr)
{
  {
    FILE *file = fopen("density_cmp.txt", "w");
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++) 
    {
      gmp_fprintf(file, "%.80Fe\n", variables[i*NVAR + VAR_DENSITY]);
    }
    fclose(file);
  }

  {
    FILE *file = fopen("momentum_cmp.txt", "w");
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++) 
      {
        gmp_fprintf(file, "%.80Fe \n", variables[i*NVAR + (VAR_MOMENTUM+j)]);
      }
      fprintf(file, "\n");
    }
    fclose(file);
  }

  {
    FILE *file = fopen("density_energy_cmp.txt", "w");
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++) 
    {
      gmp_fprintf(file, "%.80Fe\n", variables[i*NVAR + VAR_DENSITY_ENERGY]);
    }
    fclose(file);
  }
}




/*
 * Main function
 */
int main(int argc, char** argv)
{
  // 0. Intro
  // std::cout << "\n[CAMPARY]\n" << std::endl;

  // std::cout << "check tuning: \n";
  // check_tuning();
/*===============================================
// CHECK TUNING ITERATION
#ifdef TUNE0
  std::cout << "No tuning!\n";
#else
  std::cout << " *** WARNING: Please check the updated tuning policy! *** \n";
#endif
*///=============================================

  // Use this call to catch nan errors:
  //  (enables all floating point exceptions but FE_INEXACT)
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT); 

  // Manage command line input arguments
  if (argc < 2)
  {
    std::cout << "specify data file name" << std::endl;
    std::cout << "example usage: [progam.ex] [data_file] [iterations] \n";
    std::cout << "$ ./euler3d_cpu_mixed.ex ./data/fvcorr.domn.097K 2000" << std::endl;
    return 0;
  }
  else if (argc < 3)
  {
    std::cout << "specify number of iterations" << std::endl;
    return 0;
  }
  const char* data_file_name = argv[1];
  const int iterations = atoi(argv[2]);

  
  // 1. Global constants for CAMPARY:
  initFromDouble(&gamma_cmp, double(GAMMA));
  initFromDouble(&one, double(1.0));
  initFromDouble(&half, 0.5);
#ifdef OADD
  one = -one;
  add_CMPToD(&gamma_minus_one, &gamma_cmp, &one);
  one = -one;
#else
  certifAddExpans(gamma_minus_one, gamma_cmp, -one);
#endif


  // 2. Set far field conditions:
  {
    initFromDouble(&ff_variable_cmp[VAR_DENSITY], double(1.4));

    multi_prec<CPR> ff_pressure_cmp;
    initFromDouble(&ff_pressure_cmp, double(1.0));

    multi_prec<CPR> ff_speed_of_sound_cmp, tmp1, tmp2;
#ifdef OMUL
    mul_CMPToD(&tmp1, &ff_pressure_cmp, &gamma_cmp);
#else
    certifMulExpans(tmp1, ff_pressure_cmp, gamma_cmp);
#endif
#ifdef ODIV
    div_CMPToD(&tmp2, &tmp1, &ff_variable_cmp[VAR_DENSITY]);
#else
    divExpans(tmp2, tmp1, ff_variable_cmp[VAR_DENSITY]);
#endif
#ifdef OSQR
    sqrt_CMPToD(&ff_speed_of_sound_cmp, &tmp2);
#else
    sqrtNewtonExpans(ff_speed_of_sound_cmp, tmp2);
#endif

    multi_prec<CPR> ff_speed_cmp, ff_mach_cmp;
    initFromDouble(&ff_mach_cmp, double(ff_mach));
#ifdef OMUL
    mul_CMPToD(&ff_speed_cmp, &ff_speed_of_sound_cmp, &ff_mach_cmp);
#else    
    certifMulExpans(ff_speed_cmp, ff_speed_of_sound_cmp, ff_mach_cmp);
#endif

    const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);
    const double angle_tol = pow(10.0, -14);
    campary3 ff_velocity_cmp;
    if ( angle_of_attack < angle_tol )
    {
      ff_velocity_cmp.x = ff_speed_cmp;
      initFromDouble(&ff_velocity_cmp.y, 0.0);      
    }
    else
    {
      std::cout << "***Error: the CAMPARY version does not support non-zero attack angle yet\n";
      return 1;
    }
    initFromDouble(&ff_velocity_cmp.z, 0.0);

#ifdef OMUL
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+0], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.x);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+0], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.x);
#endif
#ifdef OMUL    
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+1], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.y);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+1], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.y);
#endif
#ifdef OMUL
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+2], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.z);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+2], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.z);
#endif    

#ifdef OMUL
    mul_CMPToD(&tmp1, &ff_speed_cmp, &ff_speed_cmp);
#else
    certifMulExpans(tmp1, ff_speed_cmp, ff_speed_cmp);
#endif
#ifdef OMUL    
    mul_CMPToD(&tmp2, &tmp1, &half);
#else
    certifMulExpans(tmp2, tmp1, half);
#endif
#ifdef OMUL    
    mul_CMPToD(&tmp1, &ff_variable_cmp[VAR_DENSITY], &tmp2);
#else
    certifMulExpans(tmp1, ff_variable_cmp[VAR_DENSITY], tmp2);
#endif
#ifdef ODIV
    div_CMPToD(&tmp2, &ff_pressure_cmp, &gamma_minus_one);
#else
    divExpans(tmp2, ff_pressure_cmp, gamma_minus_one);
#endif
#ifdef OADD
    add_CMPToD(&ff_variable_cmp[VAR_DENSITY_ENERGY], &tmp1, &tmp2);
#else
    certifAddExpans(ff_variable_cmp[VAR_DENSITY_ENERGY], tmp1, tmp2);
#endif
    
    campary3 ff_momentum_cmp;
    ff_momentum_cmp.x = *(ff_variable_cmp+VAR_MOMENTUM+0);
    ff_momentum_cmp.y = *(ff_variable_cmp+VAR_MOMENTUM+1);
    ff_momentum_cmp.z = *(ff_variable_cmp+VAR_MOMENTUM+2);

    compute_flux_contribution_cmp(
      ff_variable_cmp[VAR_DENSITY], 
      ff_momentum_cmp, 
      ff_variable_cmp[VAR_DENSITY_ENERGY], 
      ff_pressure_cmp, 
      ff_velocity_cmp, 
      ff_flux_contribution_momentum_x_cmp, 
      ff_flux_contribution_momentum_y_cmp, 
      ff_flux_contribution_momentum_z_cmp, 
      ff_flux_contribution_density_energy_cmp);    
  }


  // 3. Read in domain geometry once:
  int nel;
  int nelr;
  double* areas;
  int* elements_surrounding_elements;
  double* normals;
  {
    std::ifstream file(data_file_name);

    file >> nel;
    nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

    areas = new double[nelr];
    elements_surrounding_elements = new int[nelr*NNB];
    normals = new double[NDIM*NNB*nelr];

    // read in data
    for(int i = 0; i < nel; i++)
    {
      file >> areas[i];
      for(int j = 0; j < NNB; j++)
      {
        file >> elements_surrounding_elements[i*NNB + j];
        if(elements_surrounding_elements[i*NNB+j] < 0) elements_surrounding_elements[i*NNB+j] = -1;
        elements_surrounding_elements[i*NNB + j]--; //it's coming in with Fortran numbering

        for(int k = 0; k < NDIM; k++)
        {
          file >>  normals[(i*NNB + j)*NDIM + k];
          normals[(i*NNB + j)*NDIM + k] = -normals[(i*NNB + j)*NDIM + k];
        }
      }
    }

    // fill in remaining data
    int last = nel-1;
    for(int i = nel; i < nelr; i++)
    {
      areas[i] = areas[last];
      for(int j = 0; j < NNB; j++)
      {
        // duplicate the last element
        elements_surrounding_elements[i*NNB + j] = elements_surrounding_elements[last*NNB + j];
        for(int k = 0; k < NDIM; k++) normals[(i*NNB + j)*NDIM + k] = normals[(last*NNB + j)*NDIM + k];
      }
    }
  }
  // Convert areas and normals to CAMPARY types:
  multi_prec<CPR>* areas_cmp;
  multi_prec<CPR>* normals_cmp;
  areas_cmp = new multi_prec<CPR>[nelr];
  normals_cmp = new multi_prec<CPR>[NDIM*NNB*nelr];
  convertDToCMP(areas_cmp, areas, nelr);
  convertDToCMP(normals_cmp, normals, NDIM*NNB*nelr);
  
 
  // ==========================================================================
  // 4. Compute solution:
  // ==========================================================================
  // Create arrays and set initial conditions:
  multi_prec<CPR>* variables_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  initialize_variables_cmp(nelr, variables_cmp);
  multi_prec<CPR>* old_variables_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  multi_prec<CPR>* fluxes_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  multi_prec<CPR>* step_factors_cmp = alloc< multi_prec<CPR> >(nelr); 

  // Main simulation kernel:
  // std::cout << "\nStarting CAMPARY multi_prec<" << CPR << "> simulation..." << std::endl;
  double start, end;
  start = omp_get_wtime();
  for(int i = 0; i < iterations; i++)
  {
    // std::cout << "  begin iteration #" << i << "\n";
    copy< multi_prec<CPR> >(old_variables_cmp, variables_cmp, nelr*NVAR);

    // for the first iteration we compute the time step
    compute_step_factor_cmp(nelr, variables_cmp, areas_cmp, step_factors_cmp);

    for(int j = 0; j < RK; j++)
    {
      compute_flux_cmp(nelr, elements_surrounding_elements, normals_cmp, variables_cmp, fluxes_cmp);
      time_step_cmp(j, nelr, old_variables_cmp, variables_cmp, step_factors_cmp, fluxes_cmp);      
    }
  }
  end = omp_get_wtime();
  // std::cout << "Completed in " << (end-start)  / iterations << " seconds per iteration" << std::endl;
  // std::cout << "Completed in " << (end-start) << " seconds total\n" << std::endl;
  std::cout << (end-start) << std::endl;
  // ==========================================================================
  // ==========================================================================


  // DEBUG
  // Perform a quick visual check:
  // std::cout << "Visually check output..." << std::endl;
  // int start_i = 10000;
  // int end_i =   10010;
  // mpf_t x;
  // mpf_init2(x, 256);
  // for (int i = start_i; i < end_i; ++i)
  // {
    // convertScalarCMPToMPF(x, variables_cmp[i]);
    // std::cout << "  multi_prec:\tvariables[" << i << "] =\t";
    // gmp_printf("%.64Ff\n", x);
  // }  
  // std::cout << "Visual check completed..." << std::endl;

  // Save the solution to output files: 
  // std::cout << "\nSaving solution..." << std::endl;
  // mpf_t* variables_gmp = alloc<mpf_t>(nelr*NVAR);
  // init_gmp(variables_gmp, nelr*NVAR);
  // convertCMPToMPF(variables_gmp, variables_cmp, nelr*NVAR);
  // dump_gmp(variables_gmp, nel, nelr);
  // std::cout << "Saved solution..." << std::endl;
  
  // Free the dynamic memory: 
  // std::cout << "\nCleaning up..." << std::endl;
  dealloc<int>(elements_surrounding_elements);
  // double
  dealloc<double>(areas);
  dealloc<double>(normals);
  // multi_prec
  dealloc< multi_prec<CPR> >(areas_cmp);
  dealloc< multi_prec<CPR> >(normals_cmp);
  dealloc< multi_prec<CPR> >(variables_cmp);
  dealloc< multi_prec<CPR> >(old_variables_cmp);
  dealloc< multi_prec<CPR> >(fluxes_cmp);
  dealloc< multi_prec<CPR> >(step_factors_cmp);  

  // (end of main program)
  // std::cout << "Done..." << std::endl;
  return 0;
}
