// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

/*
Modified in 2018 
Paul A. Beata
North Carolina State University

Starting from the original code provided for double-precision
computations, we have extended to include in the same program
floating-point operations using the CAMPARY library. 
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
struct double3  { double x, y, z; };
struct campary3 { multi_prec<CPR> x, y, z; };

#ifndef block_length
#error "you need to define block_length"
#endif

// ==============================================
// Tuning iterations manually-enforced recursion:

/* Tuning Legend:
iter  oper  loop
0     all   all
1     div   out
2     mul   out
3     add   out
4     mul   in
5     div   in
6     add   in
*/
#ifdef  TUNE8   // sqrt inside loop
#define TUNE7
#endif
#ifdef  TUNE7   // sqrt outside loop
#define TUNE6
#endif
#ifdef  TUNE6   // additions inside loop
#define TUNE5
#endif
#ifdef  TUNE5   // divisions inside loop
#define TUNE4
#endif
#ifdef  TUNE4   // multiplications inside loop
#define TUNE3
#endif
#ifdef  TUNE3   // additions outside loop
#define TUNE2
#endif
#ifdef  TUNE2   // multiplications outside loop
#define TUNE1   // divisions outside loop
#endif

// DEBUG
// #define TUNE1
// #define TUNE2
// #define TUNE3
// #define TUNE4
// #define TUNE5
// #define TUNE6
// ==============================================

/*
 * Options
 *
 */
// NOTE: The integer "iterations" was replaced 
// by a command line argument now:
// #define iterations 2000

#define GAMMA 1.4
#define NDIM 3
#define NNB 4

#define RK 3  // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0

/*
 * not options
 */
#define GMP 256
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


// MOVED TO tuner_cpu.h
// New tuner function:
// void sqrt_CMPToD(multi_prec<CPR> *res, multi_prec<CPR> *inVal) {
//   double tmp, value;
//   convertCMPToD(&value, inVal, 1);
//   tmp = std::sqrt(value);
//   convertDToCMP(res, &tmp, 1);
// }

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
  {
    mpf_init2(A[i], GMP);
  }
}

void dump(double* variables, int nel, int nelr)
{
  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) 
      file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY] << std::endl;
  }

  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++) 
        file << std::fixed << std::setprecision(24) << variables[i*NVAR + (VAR_MOMENTUM+j)] << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) 
      file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY_ENERGY] << std::endl;
  }
}

void compute_difference(
  double* variables, 
  multi_prec<CPR>* variables_cmp, 
  int nel, 
  int nelr)
{
  // Convert double and multi_prec results to GNU MPFR:
  mpf_t *gmp_vars_db = alloc<mpf_t>(nelr*NVAR);   // db = double
  mpf_t *gmp_vars_mp = alloc<mpf_t>(nelr*NVAR);   // mp = multi_prec
  for (int k = 0; k < nelr*NVAR; k++)
  {
    mpf_init2(gmp_vars_db[k], 256);
    mpf_set_d(gmp_vars_db[k], variables[k]);
    mpf_init2(gmp_vars_mp[k], 256);
  }
  convertCMPToMPF(gmp_vars_mp, variables_cmp, nelr*NVAR);

  // Density
  mpf_t tmp, abs_val, avg_diff_dens;
  mpf_init2(tmp, 256);
  mpf_init2(abs_val, 256);
  mpf_init2(avg_diff_dens, 256);
  mpf_set_d(avg_diff_dens, 0.0);
  for(int i = 0; i < nel; i++) 
  {
    mpf_sub(tmp, gmp_vars_db[i*NVAR + VAR_DENSITY], gmp_vars_mp[i*NVAR + VAR_DENSITY]);
    mpf_abs(abs_val, tmp);
    mpf_add(avg_diff_dens, avg_diff_dens, abs_val);
  }
  mpf_div_ui(avg_diff_dens, avg_diff_dens, nel);
  gmp_printf("Average difference for DENSITY \t\t= %.8Fe\n", avg_diff_dens);

  // Momentum [with NDIM components]
  mpf_t *avg_diff_mom = alloc<mpf_t>(NDIM);
  for (int j = 0; j < NDIM; j++)
  {
    mpf_init2(avg_diff_mom[j], 256);
    mpf_set_d(avg_diff_mom[j], 0.0);
  }
  for(int i = 0; i < nel; i++)
  {
    for(int j = 0; j != NDIM; j++) 
    {
      mpf_sub(tmp, gmp_vars_db[i*NVAR + (VAR_MOMENTUM+j)], gmp_vars_mp[i*NVAR + (VAR_MOMENTUM+j)]);
      mpf_abs(abs_val, tmp);
      mpf_add(avg_diff_mom[j], avg_diff_mom[j], abs_val);
    }
  }
  for (int j = 0; j < NDIM; j++)
  {
    mpf_div_ui(avg_diff_mom[j], avg_diff_mom[j], nel);
    gmp_printf("Average difference for MOMENTUM[%d] \t= %.8Fe\n", j, avg_diff_mom[j]);
  }

  // Density-Energy
  mpf_t avg_diff_dens_eng;
  mpf_init2(avg_diff_dens_eng, 256);
  mpf_set_d(avg_diff_dens_eng, 0.0);
  for(int i = 0; i < nel; i++) 
  {
    mpf_sub(tmp, gmp_vars_db[i*NVAR + VAR_DENSITY_ENERGY], gmp_vars_mp[i*NVAR + VAR_DENSITY_ENERGY]);
    mpf_abs(abs_val, tmp);
    mpf_add(avg_diff_dens_eng, avg_diff_dens_eng, abs_val);
  }
  mpf_div_ui(avg_diff_dens_eng, avg_diff_dens_eng, nel);
  gmp_printf("Average difference for DENSITY-ENERGY \t= %.8Fe\n", avg_diff_dens_eng);
}

void compute_array_difference(
  double* variables, 
  multi_prec<CPR>* variables_cmp, 
  int array_size)
{
  // Convert double and multi_prec results to GNU MPFR:
  mpf_t *gmp_vars_db = alloc<mpf_t>(array_size);   // db = double
  mpf_t *gmp_vars_mp = alloc<mpf_t>(array_size);   // mp = multi_prec
  for (int i = 0; i < array_size; i++)
  {
    mpf_init2(gmp_vars_db[i], 256);
    mpf_set_d(gmp_vars_db[i], variables[i]);
    mpf_init2(gmp_vars_mp[i], 256);
  }
  convertCMPToMPF(gmp_vars_mp, variables_cmp, array_size);

  // Compute simple absolute difference and average:
  mpf_t tmp, abs_val, avg_diff;
  mpf_init2(tmp, 256);
  mpf_init2(abs_val, 256);
  mpf_init2(avg_diff, 256);
  mpf_set_d(avg_diff, 0.0);
  for(int i = 0; i < array_size; i++) 
  {
    mpf_sub(tmp, gmp_vars_db[i], gmp_vars_mp[i]);
    mpf_abs(abs_val, tmp);
    mpf_add(avg_diff, avg_diff, abs_val);
  }
  mpf_div_ui(avg_diff, avg_diff, array_size);
  gmp_printf("Average difference for array [%d] \t\t= %.8Fe\n", array_size, avg_diff);
}





/*
 * Element-based Cell-centered FVM solver functions
 */
double ff_variable[NVAR];
double3 ff_flux_contribution_momentum_x;
double3 ff_flux_contribution_momentum_y;
double3 ff_flux_contribution_momentum_z;
double3 ff_flux_contribution_density_energy;

void initialize_variables(int nelr, double* variables)
{
  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    for(int j = 0; j < NVAR; j++) variables[i*NVAR + j] = ff_variable[j];
  }
}

inline void compute_flux_contribution(double& density, double3& momentum, double& density_energy, double& pressure, double3& velocity, double3& fc_momentum_x, double3& fc_momentum_y, double3& fc_momentum_z, double3& fc_density_energy)
{
  fc_momentum_x.x = velocity.x*momentum.x + pressure;
  fc_momentum_x.y = velocity.x*momentum.y;
  fc_momentum_x.z = velocity.x*momentum.z;

  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y*momentum.y + pressure;
  fc_momentum_y.z = velocity.y*momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z*momentum.z + pressure;

  double de_p = density_energy+pressure;
  fc_density_energy.x = velocity.x*de_p;
  fc_density_energy.y = velocity.y*de_p;
  fc_density_energy.z = velocity.z*de_p;
}

inline void compute_velocity(double& density, double3& momentum, double3& velocity)
{
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}

inline double compute_speed_sqd(double3& velocity)
{
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline double compute_pressure(double& density, double& density_energy, double& speed_sqd)
{
  return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}

inline double compute_speed_of_sound(double& density, double& pressure)
{
  return std::sqrt(double(GAMMA)*pressure/density);
}



void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    double density = variables[NVAR*i + VAR_DENSITY];

    double3 momentum;
    momentum.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
    momentum.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
    momentum.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

    double density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
    double3 velocity;    compute_velocity(density, momentum, velocity);
    double speed_sqd      = compute_speed_sqd(velocity);
    double pressure       = compute_pressure(density, density_energy, speed_sqd);
    double speed_of_sound = compute_speed_of_sound(density, pressure);

    // dt = double(0.5) * std::sqrt(areas[i]) /  (||v|| + c).... 
    // but when we do time stepping, this later would need to be divided by the area, 
    // so we just do it all at once
    step_factors[i] = double(0.5) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
  }
}


/*
 *
 *
*/

void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
  const double smoothing_coefficient = double(0.2f);

  #pragma omp parallel for default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    int j, nb;
    double3 normal; double normal_len;
    double factor;

    double density_i = variables[NVAR*i + VAR_DENSITY];
    double3 momentum_i;
    momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
    momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
    momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

    double density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

    double3 velocity_i;                      compute_velocity(density_i, momentum_i, velocity_i);
    double speed_sqd_i                          = compute_speed_sqd(velocity_i);
    double speed_i                              = std::sqrt(speed_sqd_i);
    double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
    double3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
    double3 flux_contribution_i_density_energy;
    compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

    double flux_i_density = double(0.0);
    double3 flux_i_momentum;
    flux_i_momentum.x = double(0.0);
    flux_i_momentum.y = double(0.0);
    flux_i_momentum.z = double(0.0);
    double flux_i_density_energy = double(0.0);

    double3 velocity_nb;
    double density_nb, density_energy_nb;
    double3 momentum_nb;
    double3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
    double3 flux_contribution_nb_density_energy;
    double speed_sqd_nb, speed_of_sound_nb, pressure_nb;

    for(j = 0; j < NNB; j++)
    {
      nb = elements_surrounding_elements[i*NNB + j];
      normal.x = normals[(i*NNB + j)*NDIM + 0];
      normal.y = normals[(i*NNB + j)*NDIM + 1];
      normal.z = normals[(i*NNB + j)*NDIM + 2];
      normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

      if(nb >= 0)   // a legitimate neighbor
      {
        density_nb =        variables[nb*NVAR + VAR_DENSITY];
        momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
        momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
        momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
        density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
                          compute_velocity(density_nb, momentum_nb, velocity_nb);
        speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
        pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
        speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
                          compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

        // artificial viscosity
        factor = -normal_len*smoothing_coefficient*double(0.5)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
        flux_i_density += factor*(density_i-density_nb);
        flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
        flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
        flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
        flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

        // accumulate cell-centered fluxes
        factor = double(0.5)*normal.x;
        flux_i_density += factor*(momentum_nb.x+momentum_i.x);
        flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
        flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
        flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
        flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);

        factor = double(0.5)*normal.y;
        flux_i_density += factor*(momentum_nb.y+momentum_i.y);
        flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
        flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
        flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
        flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);

        factor = double(0.5)*normal.z;
        flux_i_density += factor*(momentum_nb.z+momentum_i.z);
        flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
        flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
        flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
        flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
      }
      else if(nb == -1) // a wing boundary
      {
        flux_i_momentum.x += normal.x*pressure_i;
        flux_i_momentum.y += normal.y*pressure_i;
        flux_i_momentum.z += normal.z*pressure_i;
      }
      else if(nb == -2) // a far field boundary
      {
        factor = double(0.5)*normal.x;
        flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
        flux_i_density_energy += factor*(ff_flux_contribution_density_energy.x+flux_contribution_i_density_energy.x);
        flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.x + flux_contribution_i_momentum_x.x);
        flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.x + flux_contribution_i_momentum_y.x);
        flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.x + flux_contribution_i_momentum_z.x);

        factor = double(0.5)*normal.y;
        flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
        flux_i_density_energy += factor*(ff_flux_contribution_density_energy.y+flux_contribution_i_density_energy.y);
        flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.y + flux_contribution_i_momentum_x.y);
        flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.y + flux_contribution_i_momentum_y.y);
        flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.y + flux_contribution_i_momentum_z.y);

        factor = double(0.5)*normal.z;
        flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
        flux_i_density_energy += factor*(ff_flux_contribution_density_energy.z+flux_contribution_i_density_energy.z);
        flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.z + flux_contribution_i_momentum_x.z);
        flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.z + flux_contribution_i_momentum_y.z);
        flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.z + flux_contribution_i_momentum_z.z);

      }
    }

    fluxes[i*NVAR + VAR_DENSITY] = flux_i_density;
    fluxes[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
    fluxes[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
    fluxes[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
    fluxes[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
  }
}

void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
  #pragma omp parallel for  default(shared) schedule(static)
  for(int i = 0; i < nelr; i++)
  {
    double factor = step_factors[i]/double(RK+1-j);

    variables[NVAR*i + VAR_DENSITY] = old_variables[NVAR*i + VAR_DENSITY] + factor*fluxes[NVAR*i + VAR_DENSITY];
    variables[NVAR*i + VAR_DENSITY_ENERGY] = old_variables[NVAR*i + VAR_DENSITY_ENERGY] + factor*fluxes[NVAR*i + VAR_DENSITY_ENERGY];
    variables[NVAR*i + (VAR_MOMENTUM+0)] = old_variables[NVAR*i + (VAR_MOMENTUM+0)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+0)];
    variables[NVAR*i + (VAR_MOMENTUM+1)] = old_variables[NVAR*i + (VAR_MOMENTUM+1)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+1)];
    variables[NVAR*i + (VAR_MOMENTUM+2)] = old_variables[NVAR*i + (VAR_MOMENTUM+2)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+2)];
  }
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

inline void compute_flux_contribution_outloop_cmp(
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

#ifdef TUNE2
  mul_CMPToD(&tmp, &velocity.x, &momentum.x);
#else
  certifMulExpans(tmp, velocity.x, momentum.x);
#endif
#ifdef TUNE3
  add_CMPToD(&fc_momentum_x.x, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_x.x, tmp, pressure);
#endif
#ifdef TUNE2  
  mul_CMPToD(&fc_momentum_x.y, &velocity.x, &momentum.y);
#else
  certifMulExpans(fc_momentum_x.y, velocity.x, momentum.y);
#endif
#ifdef TUNE2
  mul_CMPToD(&fc_momentum_x.z, &velocity.x, &momentum.z);
#else
  certifMulExpans(fc_momentum_x.z, velocity.x, momentum.z);
#endif

  fc_momentum_y.x = fc_momentum_x.y;
#ifdef TUNE2  
  mul_CMPToD(&tmp, &velocity.y, &momentum.y);
#else  
  certifMulExpans(tmp, velocity.y, momentum.y);
#endif
#ifdef TUNE3
  add_CMPToD(&fc_momentum_y.y, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_y.y, tmp, pressure);
#endif
#ifdef TUNE2
  mul_CMPToD(&fc_momentum_y.z, &velocity.y, &momentum.z);
#else
  certifMulExpans(fc_momentum_y.z, velocity.y, momentum.z);
#endif  

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
#ifdef TUNE2  
  mul_CMPToD(&tmp, &velocity.z, &momentum.z);
#else
  certifMulExpans(tmp, velocity.z, momentum.z);
#endif
#ifdef TUNE3
  add_CMPToD(&fc_momentum_z.z, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_z.z, tmp, pressure);
#endif

#ifdef TUNE3
  add_CMPToD(&de_p, &density_energy, &pressure);
#else
  certifAddExpans(de_p, density_energy, pressure);
#endif
#ifdef TUNE2  
  mul_CMPToD(&fc_density_energy.x, &velocity.x, &de_p);
#else
  certifMulExpans(fc_density_energy.x, velocity.x, de_p);
#endif
#ifdef TUNE2  
  mul_CMPToD(&fc_density_energy.y, &velocity.y, &de_p);
#else
  certifMulExpans(fc_density_energy.y, velocity.y, de_p);
#endif
#ifdef TUNE2  
  mul_CMPToD(&fc_density_energy.z, &velocity.z, &de_p);
#else
  certifMulExpans(fc_density_energy.z, velocity.z, de_p);  
#endif
}

inline void compute_flux_contribution_inloop_cmp(
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

#ifdef TUNE4
  mul_CMPToD(&tmp, &velocity.x, &momentum.x);
#else
  certifMulExpans(tmp, velocity.x, momentum.x);
#endif
#ifdef TUNE6
      add_CMPToD(&fc_momentum_x.x, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_x.x,  tmp,  pressure);
#endif
#ifdef TUNE4
  mul_CMPToD(&fc_momentum_x.y, &velocity.x, &momentum.y);
#else
  certifMulExpans(fc_momentum_x.y, velocity.x, momentum.y);
#endif
#ifdef TUNE4
  mul_CMPToD(&fc_momentum_x.z, &velocity.x, &momentum.z);
#else
  certifMulExpans(fc_momentum_x.z, velocity.x, momentum.z);
#endif

  fc_momentum_y.x = fc_momentum_x.y;
#ifdef TUNE4
  mul_CMPToD(&tmp, &velocity.y, &momentum.y);
#else  
  certifMulExpans(tmp, velocity.y, momentum.y);
#endif
#ifdef TUNE6
      add_CMPToD(&fc_momentum_y.y, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_y.y,  tmp,  pressure);
#endif
#ifdef TUNE4
  mul_CMPToD(&fc_momentum_y.z, &velocity.y, &momentum.z);
#else
  certifMulExpans(fc_momentum_y.z, velocity.y, momentum.z);
#endif  

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
#ifdef TUNE4  
  mul_CMPToD(&tmp, &velocity.z, &momentum.z);
#else
  certifMulExpans(tmp, velocity.z, momentum.z);
#endif
#ifdef TUNE6
      add_CMPToD(&fc_momentum_z.z, &tmp, &pressure);
#else
  certifAddExpans(fc_momentum_z.z,  tmp,  pressure);
#endif

#ifdef TUNE6
      add_CMPToD(&de_p, &density_energy, &pressure);
#else
  certifAddExpans(de_p,  density_energy,  pressure);
#endif
#ifdef TUNE4  
  mul_CMPToD(&fc_density_energy.x, &velocity.x, &de_p);
#else
  certifMulExpans(fc_density_energy.x, velocity.x, de_p);
#endif
#ifdef TUNE4  
  mul_CMPToD(&fc_density_energy.y, &velocity.y, &de_p);
#else
  certifMulExpans(fc_density_energy.y, velocity.y, de_p);
#endif
#ifdef TUNE4
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
#ifdef TUNE5
  div_CMPToD(&velocity.x, &momentum.x, &density);
#else
  divExpans(velocity.x, momentum.x, density);
#endif  
#ifdef TUNE5
  div_CMPToD(&velocity.y, &momentum.y, &density);
#else
  divExpans(velocity.y, momentum.y, density);
#endif  
#ifdef TUNE5  
  div_CMPToD(&velocity.z, &momentum.z, &density);
#else
  divExpans(velocity.z, momentum.z, density);  
#endif
}

inline multi_prec<CPR> compute_speed_sqd_cmp(campary3& velocity)
{
  multi_prec<CPR> result, tmp;
  initFromDouble(&result, 0.0);

#ifdef TUNE4
  mul_CMPToD(&tmp, &velocity.x, &velocity.x);
#else
  certifMulExpans(tmp, velocity.x, velocity.x);
#endif  
#ifdef TUNE6
     add_CMPToD(&result, &result, &tmp);
#else
  certifAddExpans(result, result,  tmp);
#endif
#ifdef TUNE4  
  mul_CMPToD(&tmp, &velocity.y, &velocity.y);
#else
  certifMulExpans(tmp, velocity.y, velocity.y);
#endif    
#ifdef TUNE6  
      add_CMPToD(&result,&result,&tmp);
#else
  certifAddExpans(result, result, tmp);
#endif
#ifdef TUNE4  
  mul_CMPToD(&tmp, &velocity.z, &velocity.z);
#else
  certifMulExpans(tmp, velocity.z, velocity.z);
#endif    
#ifdef TUNE6  
      add_CMPToD(&result,&result,&tmp);
#else
  certifAddExpans(result, result, tmp);
#endif

  return result;
}

inline multi_prec<CPR> compute_pressure_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& density_energy, 
  multi_prec<CPR>& speed_sqd)
{
  multi_prec<CPR> tmp1, tmp2, tmp3, result;

#ifdef TUNE4  
  mul_CMPToD(&tmp1, &half, &density);
#else
  certifMulExpans(tmp1, half, density);
#endif  
#ifdef TUNE4    
  mul_CMPToD(&tmp2, &tmp1, &speed_sqd);
#else
  certifMulExpans(tmp2, tmp1, speed_sqd);
#endif  
#ifdef TUNE6
  certifMulExpans(tmp3, tmp2, -one);
  add_CMPToD(&tmp1,&density_energy, &tmp3);
#else
  certifAddExpans(tmp1, density_energy, -tmp2);
#endif
#ifdef TUNE4  
  mul_CMPToD(&result, &gamma_minus_one, &tmp1);
#else
  certifMulExpans(result, gamma_minus_one, tmp1);
#endif

  return result;
}

inline multi_prec<CPR> compute_speed_of_sound_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& pressure)
{
  multi_prec<CPR> tmp1, tmp2, result;
#ifdef TUNE4  
  mul_CMPToD(&tmp1, &gamma_cmp, &pressure);
#else
  certifMulExpans(tmp1, gamma_cmp, pressure);
#endif
#ifdef TUNE5  
  div_CMPToD(&tmp2, &tmp1, &density);
#else
  divExpans(tmp2, tmp1, density);
#endif  

  // CAMPARY SQRT
#ifdef TUNE8
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
    
    // CAMPARY SQRT
#ifdef TUNE8
    sqrt_CMPToD(&tmp1, &speed_sqd);
#else
    sqrtNewtonExpans(tmp1, speed_sqd);
#endif

#ifdef TUNE6    
        add_CMPToD(&tmp2,&tmp1,&speed_of_sound);
#else
    certifAddExpans(tmp2, tmp1, speed_of_sound);
#endif    
    
    // CAMPARY SQRT
#ifdef TUNE8    
    sqrt_CMPToD(&tmp3, &areas[i]);
#else
    sqrtNewtonExpans(tmp3, areas[i]);
#endif

#ifdef TUNE4   
    mul_CMPToD(&denom, &tmp3, &tmp2);
#else
    certifMulExpans(denom, tmp3, tmp2);
#endif
#ifdef TUNE5
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

#ifdef TUNE4
  mul_CMPToD(&result, &normal.x, &normal.x);
#else
  certifMulExpans(result, normal.x, normal.x);
#endif  
#ifdef TUNE6  
      add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

#ifdef TUNE4
  mul_CMPToD(&result, &normal.y, &normal.y);
#else
  certifMulExpans(result, normal.y, normal.y);
#endif  
#ifdef TUNE6    
      add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

#ifdef TUNE4
  mul_CMPToD(&result, &normal.z, &normal.z);
#else
  certifMulExpans(result, normal.z, normal.z);
#endif  
#ifdef TUNE6    
      add_CMPToD(&tmp,&tmp,&result);
#else
  certifAddExpans(tmp, tmp, result);
#endif

  // CAMPARY SQRT
#ifdef TUNE8
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
#ifdef TUNE6
    certifMulExpans(tmp2, d, -one);
    add_CMPToD(&tmp1, &c, &tmp2);
#else
    certifAddExpans(tmp1, c, -d);
#endif    
  }
  else
  {
#ifdef TUNE6
    add_CMPToD(&tmp1, &c, &d);
#else
    certifAddExpans(tmp1, c, d);
#endif    
  }

#ifdef TUNE4
  mul_CMPToD(&tmp2, &b, &tmp1);
#else
  certifMulExpans(tmp2, b, tmp1);
#endif  
#ifdef TUNE6
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
    
    // CAMPARY SQRT
#ifdef TUNE8    
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

    compute_flux_contribution_inloop_cmp(
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

        compute_flux_contribution_inloop_cmp(
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
        
        // CAMPARY SQRT
#ifdef TUNE8
        sqrt_CMPToD(&tmp1, &speed_sqd_nb);
#else        
        sqrtNewtonExpans(tmp1, speed_sqd_nb);
#endif        

#ifdef TUNE6
        add_CMPToD(&tmp2, &speed_i, &tmp1);
#else
        certifAddExpans(tmp2, speed_i, tmp1);
#endif        
#ifdef TUNE6
        add_CMPToD(&tmp2, &tmp2, &speed_of_sound_i);
#else
        certifAddExpans(tmp2, tmp2, speed_of_sound_i);
#endif
#ifdef TUNE6     
        add_CMPToD(&tmp2, &tmp2, &speed_of_sound_nb);
#else
        certifAddExpans(tmp2, tmp2, speed_of_sound_nb);
#endif
#ifdef TUNE4
        mul_CMPToD(&tmp1, &normal_len, &smoothing_coefficient);
#else
        certifMulExpans(tmp1, normal_len, smoothing_coefficient);
#endif
#ifdef TUNE4
        mul_CMPToD(&tmp3, &tmp1, &half);
        tmp3 = -tmp3;
#else
        certifMulExpans(tmp3, tmp1, -half);
#endif
#ifdef TUNE4
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
#ifdef TUNE4
        mul_CMPToD(&factor, &half, &normal.x);
#else
        certifMulExpans(factor, half, normal.x);
#endif
        accumulate_cmp(flux_i_density, factor, momentum_nb.x, momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.x, flux_contribution_i_momentum_z.x);

#ifdef TUNE4
        mul_CMPToD(&factor, &half, &normal.y);
#else
        certifMulExpans(factor, half, normal.y);
#endif
        accumulate_cmp(flux_i_density, factor, momentum_nb.y, momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.y, flux_contribution_i_momentum_z.y);        

#ifdef TUNE4
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
#ifdef TUNE4
        mul_CMPToD(&tmp1, &pressure_i, &normal.x);
#else
        certifMulExpans(tmp1, pressure_i, normal.x);
#endif
#ifdef TUNE6
        add_CMPToD(&flux_i_momentum.x, &flux_i_momentum.x, &tmp1);
#else
        certifAddExpans(flux_i_momentum.x, flux_i_momentum.x, tmp1);
#endif        

#ifdef TUNE4
        mul_CMPToD(&tmp1, &pressure_i, &normal.y);
#else
        certifMulExpans(tmp1, pressure_i, normal.y);
#endif
#ifdef TUNE6
        add_CMPToD(&flux_i_momentum.y, &flux_i_momentum.y, &tmp1);
#else
        certifAddExpans(flux_i_momentum.y, flux_i_momentum.y, tmp1);
#endif

#ifdef TUNE4
        mul_CMPToD(&tmp1, &pressure_i, &normal.z);
#else
        certifMulExpans(tmp1, pressure_i, normal.z); 
#endif
#ifdef TUNE6
        add_CMPToD(&flux_i_momentum.z, &flux_i_momentum.z, &tmp1);
#else
        certifAddExpans(flux_i_momentum.z, flux_i_momentum.z, tmp1);
#endif
      }
      else if(nb == -2) // a far field boundary
      {
#ifdef TUNE4     
        mul_CMPToD(&factor, &half, &normal.x);
#else
        certifMulExpans(factor, half, normal.x);
#endif
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+0], momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.x, flux_contribution_i_momentum_z.x);        

#ifdef TUNE4
        mul_CMPToD(&factor, &half, &normal.y);
#else
        certifMulExpans(factor, half, normal.y);
#endif
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+1], momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.y, flux_contribution_i_momentum_z.y);        

#ifdef TUNE4
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
#ifdef TUNE5
    div_CMPToD(&factor, &step_factors[i], &tmp);
#else
    divExpans(factor, step_factors[i], tmp);
#endif    
    
#ifdef TUNE4  
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + VAR_DENSITY]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY]);
#endif    
#ifdef TUNE6
    add_CMPToD(&variables[NVAR*i + VAR_DENSITY], &old_variables[NVAR*i + VAR_DENSITY], &tmp);
#else
    certifAddExpans(variables[NVAR*i + VAR_DENSITY], old_variables[NVAR*i + VAR_DENSITY], tmp);
#endif    

#ifdef TUNE4
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
#endif
#ifdef TUNE6
    add_CMPToD(&variables[NVAR*i + VAR_DENSITY_ENERGY], &old_variables[NVAR*i + VAR_DENSITY_ENERGY], &tmp);
#else
    certifAddExpans(variables[NVAR*i + VAR_DENSITY_ENERGY], old_variables[NVAR*i + VAR_DENSITY_ENERGY], tmp);
#endif

#ifdef TUNE4
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
#endif
#ifdef TUNE6
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+0)], &old_variables[NVAR*i + (VAR_MOMENTUM+0)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+0)], old_variables[NVAR*i + (VAR_MOMENTUM+0)], tmp);
#endif

#ifdef TUNE4
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
#endif
#ifdef TUNE6
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+1)], &old_variables[NVAR*i + (VAR_MOMENTUM+1)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+1)], old_variables[NVAR*i + (VAR_MOMENTUM+1)], tmp);
#endif

#ifdef TUNE4
    mul_CMPToD(&tmp, &factor, &fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
#else
    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
#endif
#ifdef TUNE6
    add_CMPToD(&variables[NVAR*i + (VAR_MOMENTUM+2)], &old_variables[NVAR*i + (VAR_MOMENTUM+2)], &tmp);
#else
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+2)], old_variables[NVAR*i + (VAR_MOMENTUM+2)], tmp);
#endif
  }
}

void dump_gmp(mpf_t* variables, int nel, int nelr)
{
  {
    // std::ofstream file("density");
    FILE *file = fopen("density_cmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++) 
    {
      // file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY] << std::endl;
      gmp_fprintf(file, "%.80Fe\n", variables[i*NVAR + VAR_DENSITY]);
    }
    fclose(file);
  }

  {
    // std::ofstream file("momentum");
    FILE *file = fopen("momentum_cmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++) 
      {
        // file << std::fixed << std::setprecision(24) << variables[i*NVAR + (VAR_MOMENTUM+j)] << " ";
        gmp_fprintf(file, "%.80Fe \n", variables[i*NVAR + (VAR_MOMENTUM+j)]);
      }
      // file << std::endl;
      fprintf(file, "\n");
    }
    fclose(file);
  }

  {
    // std::ofstream file("density_energy");
    FILE *file = fopen("density_energy_cmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    for(int i = 0; i < nel; i++) 
    {
      // file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY_ENERGY] << std::endl;
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

  // Global constants for CAMPARY:
  initFromDouble(&gamma_cmp, double(GAMMA));
  initFromDouble(&one, double(1.0));
  initFromDouble(&half, 0.5);
#ifdef TUNE3
  one = -one;
  add_CMPToD(&gamma_minus_one, &gamma_cmp, &one);
  one = -one;
#else
  certifAddExpans(gamma_minus_one, gamma_cmp, -one);
#endif


  // CHECK TUNING ITERATION
#ifdef TUNE0
  std::cout << "No tuning!\n";
#endif
#ifdef TUNE1
  std::cout << "  Tuning iteration 1 enabled.\n";
#endif
#ifdef TUNE2
  std::cout << "  Tuning iteration 2 enabled.\n";
#endif
#ifdef TUNE3
  std::cout << "  Tuning iteration 3 enabled.\n";
#endif
#ifdef TUNE4
  std::cout << "  Tuning iteration 4 enabled.\n";
#endif
#ifdef TUNE5
  std::cout << "  Tuning iteration 5 enabled.\n";
#endif
#ifdef TUNE6
  std::cout << "  Tuning iteration 6 enabled.\n";
#endif
#ifdef TUNE7
  std::cout << "  Tuning iteration 7 enabled.\n";
#endif
#ifdef TUNE8
  std::cout << "  Tuning iteration 8 enabled.\n";
#endif
  
  
  // Set far field conditions:
  {

    // 1. double precision
    const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);

    ff_variable[VAR_DENSITY] = double(1.4);
    double ff_pressure = double(1.0);
    double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
    double ff_speed = double(ff_mach)*ff_speed_of_sound;

    double3 ff_velocity;
    // We use this if-else statement to avoid the FLOP needed to compute
    // the x-and y-direction velocity when the attack angle is near zero.
    double angle_tol = pow(10.0, -14);
    if ( angle_of_attack < angle_tol )
    {
      ff_velocity.x = ff_speed;
      ff_velocity.y = 0.0;
    }
    else
    {
      std::cout << "***Warning: cos and sin available in double-precision only!\n";
      ff_velocity.x = ff_speed*double(cos((double)angle_of_attack));
      ff_velocity.y = ff_speed*double(sin((double)angle_of_attack));
    }
    ff_velocity.z = 0.0;

    ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
    ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
    ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

    ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(double(0.5)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0));

    double3 ff_momentum;
    ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
    ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
    ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);

    compute_flux_contribution(
      ff_variable[VAR_DENSITY], 
      ff_momentum, 
      ff_variable[VAR_DENSITY_ENERGY], 
      ff_pressure, 
      ff_velocity, 
      ff_flux_contribution_momentum_x, 
      ff_flux_contribution_momentum_y, 
      ff_flux_contribution_momentum_z, 
      ff_flux_contribution_density_energy);

    
    // 2. CAMPARY multi_prec
    initFromDouble(&ff_variable_cmp[VAR_DENSITY], double(1.4));

    multi_prec<CPR> ff_pressure_cmp;
    initFromDouble(&ff_pressure_cmp, double(1.0));

    multi_prec<CPR> ff_speed_of_sound_cmp, tmp1, tmp2;
#ifdef TUNE2
    mul_CMPToD(&tmp1, &ff_pressure_cmp, &gamma_cmp);
#else
    certifMulExpans(tmp1, ff_pressure_cmp, gamma_cmp);
#endif
#ifdef TUNE1
    div_CMPToD(&tmp2, &tmp1, &ff_variable_cmp[VAR_DENSITY]);
#else
    divExpans(tmp2, tmp1, ff_variable_cmp[VAR_DENSITY]);
#endif

    // CAMPARY SQRT
#ifdef TUNE7
    sqrt_CMPToD(&ff_speed_of_sound_cmp, &tmp2);
#else
    sqrtNewtonExpans(ff_speed_of_sound_cmp, tmp2);
#endif

    multi_prec<CPR> ff_speed_cmp, ff_mach_cmp;
    initFromDouble(&ff_mach_cmp, double(ff_mach));
#ifdef TUNE2
    mul_CMPToD(&ff_speed_cmp, &ff_speed_of_sound_cmp, &ff_mach_cmp);
#else    
    certifMulExpans(ff_speed_cmp, ff_speed_of_sound_cmp, ff_mach_cmp);
#endif

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

#ifdef TUNE2
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+0], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.x);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+0], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.x);
#endif
#ifdef TUNE2    
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+1], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.y);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+1], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.y);
#endif
#ifdef TUNE2
    mul_CMPToD(&ff_variable_cmp[VAR_MOMENTUM+2], &ff_variable_cmp[VAR_DENSITY], &ff_velocity_cmp.z);
#else
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+2], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.z);
#endif    

#ifdef TUNE2
    mul_CMPToD(&tmp1, &ff_speed_cmp, &ff_speed_cmp);
#else
    certifMulExpans(tmp1, ff_speed_cmp, ff_speed_cmp);
#endif
#ifdef TUNE2    
    mul_CMPToD(&tmp2, &tmp1, &half);
#else
    certifMulExpans(tmp2, tmp1, half);
#endif
#ifdef TUNE2    
    mul_CMPToD(&tmp1, &ff_variable_cmp[VAR_DENSITY], &tmp2);
#else
    certifMulExpans(tmp1, ff_variable_cmp[VAR_DENSITY], tmp2);
#endif

#ifdef TUNE1
    div_CMPToD(&tmp2, &ff_pressure_cmp, &gamma_minus_one);
#else
    divExpans(tmp2, ff_pressure_cmp, gamma_minus_one);
#endif
#ifdef TUNE3
    add_CMPToD(&ff_variable_cmp[VAR_DENSITY_ENERGY], &tmp1, &tmp2);
#else
    certifAddExpans(ff_variable_cmp[VAR_DENSITY_ENERGY], tmp1, tmp2);
#endif
    
    campary3 ff_momentum_cmp;
    ff_momentum_cmp.x = *(ff_variable_cmp+VAR_MOMENTUM+0);
    ff_momentum_cmp.y = *(ff_variable_cmp+VAR_MOMENTUM+1);
    ff_momentum_cmp.z = *(ff_variable_cmp+VAR_MOMENTUM+2);

    compute_flux_contribution_outloop_cmp(
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


  // Read in domain geometry once:
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
  // DEBUG
  // std::cout << "\n"; compute_array_difference(areas, areas_cmp, nelr);
  // compute_array_difference(normals, normals_cmp, (NDIM*NNB*nelr));


  // [double] Create arrays and set initial conditions:
  double* variables = alloc<double>(nelr*NVAR);
  initialize_variables(nelr, variables);
  double* old_variables = alloc<double>(nelr*NVAR);
  double* fluxes = alloc<double>(nelr*NVAR);
  double* step_factors = alloc<double>(nelr);
  
  // [multi_prec] Create arrays and set initial conditions:
  multi_prec<CPR>* variables_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  initialize_variables_cmp(nelr, variables_cmp);
  multi_prec<CPR>* old_variables_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  multi_prec<CPR>* fluxes_cmp = alloc< multi_prec<CPR> >(nelr*NVAR);
  multi_prec<CPR>* step_factors_cmp = alloc< multi_prec<CPR> >(nelr);  
  // DEBUG
  // std::cout << "\n"; compute_difference(variables, variables_cmp, nel, nelr);
  // std::cout << "\n"; compute_array_difference(variables, variables_cmp, (nelr*NVAR));


  // double-precision simulation:
  double start, end;
  // std::cout << "\nStarting double-precision simulation..." << std::endl;
  // start = omp_get_wtime();
  // for(int i = 0; i < iterations; i++)
  // {
  //   // std::cout << "  begin iteration #" << i << "\n";
  //   copy<double>(old_variables, variables, nelr*NVAR);

  //   // for the first iteration we compute the time step
  //   compute_step_factor(nelr, variables, areas, step_factors);
  //   // DEBUG
  //   // compute_step_factor_cmp(nelr, variables_cmp, areas_cmp, step_factors_cmp);
  //   // std::cout << "   step_factors: "; 
  //   // compute_array_difference(step_factors, step_factors_cmp, nelr);

  //   for(int j = 0; j < RK; j++)
  //   {
  //     compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
  //     // DEBUG
  //     // compute_flux_cmp(nelr, elements_surrounding_elements, normals_cmp, variables_cmp, fluxes_cmp);
  //     // std::cout << "    compute_flux[" << j << "]: "; 
  //     // compute_array_difference(variables, variables_cmp, nelr);

  //     time_step(j, nelr, old_variables, variables, step_factors, fluxes);
  //     // DEBUG
  //     // time_step_cmp(j, nelr, old_variables_cmp, variables_cmp, step_factors_cmp, fluxes_cmp);
  //     // std::cout << "    time_step[" << j << "]   : "; 
  //     // compute_array_difference(variables, variables_cmp, nelr);
  //   }
  // }
  // end = omp_get_wtime();
  // std::cout << "Completed in " << (end-start)  / iterations << " seconds per iteration" << std::endl;
  // std::cout << "Completed in " << (end-start) << " seconds total\n" << std::endl;


  // CAMPARY multi_prec simulation:
  std::cout << "Starting CAMPARY multi_prec<" << CPR << "> simulation..." << std::endl;
  start = omp_get_wtime();
  for(int i = 0; i < iterations; i++)
  {
    std::cout << "  begin iteration #" << i << "\n";
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
  std::cout << "Completed in " << (end-start)  / iterations << " seconds per iteration" << std::endl;
  std::cout << "Completed in " << (end-start) << " seconds total\n" << std::endl;


  // Compare the double and multi_prec solutions: 
  compute_difference(variables, variables_cmp, nel, nelr);
  std::cout << "\nVisually check output..." << std::endl;
  int start_i = 10000;
  int end_i =   10010;
  // for (int i = start_i; i < end_i; ++i)
  // {
  //   std::cout << "  double:\t variables[" << i << "] \t = \t";
  //   std::cout << std::fixed << std::setprecision(64) << variables[i] << std::endl;
  // }
  // std::cout << "\n";
  mpf_t x;
  mpf_init2(x, 256);
  for (int i = start_i; i < end_i; ++i)
  {
    convertScalarCMPToMPF(x, variables_cmp[i]);
    std::cout << "  multi_prec:\t variables_cmp[" << i << "] \t = \t";
    gmp_printf("%.64Ff\n", x);
  }  
  std::cout << "Visual check completed..." << std::endl;

  std::cout << "Saving solution..." << std::endl;
  mpf_t* variables_gmp = alloc<mpf_t>(nelr*NVAR);
  init_gmp(variables_gmp, nelr*NVAR);
  convertCMPToMPF(variables_gmp, variables_cmp, nelr*NVAR);
  dump_gmp(variables_gmp, nel, nelr);
  std::cout << "Saved solution..." << std::endl;
  
  std::cout << "\nCleaning up..." << std::endl;
  dealloc<int>(elements_surrounding_elements);
  // double
  dealloc<double>(areas);
  dealloc<double>(normals);
  dealloc<double>(variables);
  dealloc<double>(old_variables);
  dealloc<double>(fluxes);
  dealloc<double>(step_factors);
  // multi_prec
  dealloc< multi_prec<CPR> >(areas_cmp);
  dealloc< multi_prec<CPR> >(normals_cmp);
  dealloc< multi_prec<CPR> >(variables_cmp);
  dealloc< multi_prec<CPR> >(old_variables_cmp);
  dealloc< multi_prec<CPR> >(fluxes_cmp);
  dealloc< multi_prec<CPR> >(step_factors_cmp);  

  // (end of main program)
  std::cout << "Done..." << std::endl;
  return 0;
}
