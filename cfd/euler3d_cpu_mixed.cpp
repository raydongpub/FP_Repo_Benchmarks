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
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


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
  gmp_printf("Average difference for array [%d] \t\t= %10.5Fe\n", array_size, avg_diff);
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

    // dt = double(0.5) * std::sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
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

  certifMulExpans(tmp, velocity.x, momentum.x);
  certifAddExpans(fc_momentum_x.x, tmp, pressure);
  certifMulExpans(fc_momentum_x.y, velocity.x, momentum.y);
  certifMulExpans(fc_momentum_x.z, velocity.x, momentum.z);

  fc_momentum_y.x = fc_momentum_x.y;
  certifMulExpans(tmp, velocity.y, momentum.y);
  certifAddExpans(fc_momentum_y.y, tmp, pressure);
  certifMulExpans(fc_momentum_y.z, velocity.y, momentum.z);

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  certifMulExpans(tmp, velocity.z, momentum.z);
  certifAddExpans(fc_momentum_z.z, tmp, pressure);

  certifAddExpans(de_p, density_energy, pressure);
  certifMulExpans(fc_density_energy.x, velocity.x, de_p);
  certifMulExpans(fc_density_energy.y, velocity.y, de_p);
  certifMulExpans(fc_density_energy.z, velocity.z, de_p);  
}

inline void compute_velocity_cmp(
  multi_prec<CPR>& density, 
  campary3& momentum, 
  campary3& velocity)
{
  divExpans(velocity.x, momentum.x, density);
  divExpans(velocity.y, momentum.y, density);
  divExpans(velocity.z, momentum.z, density);  
}

inline multi_prec<CPR> compute_speed_sqd_cmp(campary3& velocity)
{
  multi_prec<CPR> result, tmp;
  initFromDouble(&result, 0.0);

  certifMulExpans(tmp, velocity.x, velocity.x);
  certifAddExpans(result, result, tmp);
  certifMulExpans(tmp, velocity.y, velocity.y);
  certifAddExpans(result, result, tmp);
  certifMulExpans(tmp, velocity.z, velocity.z);
  certifAddExpans(result, result, tmp);

  return result;
}

inline multi_prec<CPR> compute_pressure_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& density_energy, 
  multi_prec<CPR>& speed_sqd)
{
  multi_prec<CPR> tmp1, tmp2, result;

  certifMulExpans(tmp1, half, density);
  certifMulExpans(tmp2, tmp1, speed_sqd);
  certifAddExpans(tmp1, density_energy, -tmp2);

  certifMulExpans(result, gamma_minus_one, tmp1);

  return result;
}

inline multi_prec<CPR> compute_speed_of_sound_cmp(
  multi_prec<CPR>& density, 
  multi_prec<CPR>& pressure)
{
  multi_prec<CPR> tmp1, tmp2, result;

  certifMulExpans(tmp1, gamma_cmp, pressure);
  divExpans(tmp2, tmp1, density);
  sqrtNewtonExpans(result, tmp2);

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

    multi_prec<CPR> tmp1, tmp2, tmp3, half, denom;
    sqrtNewtonExpans(tmp1, speed_sqd);
    certifAddExpans(tmp2, tmp1, speed_of_sound);
    sqrtNewtonExpans(tmp3, areas[i]);
    certifMulExpans(denom, tmp3, tmp2);
    initFromDouble(&half, 0.5);
    divExpans(step_factors[i], half, denom);
  }
}

inline multi_prec<CPR> compute_normal_len_cmp(campary3& normal)
{
  multi_prec<CPR> result, tmp;

  initFromDouble(&tmp, 0.0);
  initFromDouble(&result, 0.0);

  certifMulExpans(result, normal.x, normal.x);
  certifAddExpans(tmp, tmp, result);

  certifMulExpans(result, normal.y, normal.y);
  certifAddExpans(tmp, tmp, result);

  certifMulExpans(result, normal.z, normal.z);
  certifAddExpans(tmp, tmp, result);

  sqrtNewtonExpans(result, tmp);
  
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
    certifAddExpans(tmp1, c, -d);
  else
    certifAddExpans(tmp1, c, d);

  certifMulExpans(tmp2, b, tmp1);
  certifAddExpans(tmp1, a, tmp2);

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
    sqrtNewtonExpans(speed_i, speed_sqd_i);

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
        sqrtNewtonExpans(tmp1, speed_sqd_nb);
        certifAddExpans(tmp2, speed_i, tmp1);
        certifAddExpans(tmp2, tmp2, speed_of_sound_i);
        certifAddExpans(tmp2, tmp2, speed_of_sound_nb);
        certifMulExpans(tmp1, normal_len, smoothing_coefficient);
        certifMulExpans(tmp3, tmp1, -half);
        certifMulExpans(factor, tmp3, tmp2);

        accumulate_cmp(flux_i_density, factor, density_i, density_nb, 1);
        accumulate_cmp(flux_i_density_energy, factor, density_energy_i, density_energy_nb, 1);
        accumulate_cmp(flux_i_momentum.x, factor, momentum_i.x, momentum_nb.x, 1);
        accumulate_cmp(flux_i_momentum.y, factor, momentum_i.y, momentum_nb.y, 1);
        accumulate_cmp(flux_i_momentum.z, factor, momentum_i.z, momentum_nb.z, 1);

        // accumulate cell-centered fluxes
        certifMulExpans(factor, half, normal.x);
        accumulate_cmp(flux_i_density, factor, momentum_nb.x, momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.x, flux_contribution_i_momentum_z.x);

        certifMulExpans(factor, half, normal.y);
        accumulate_cmp(flux_i_density, factor, momentum_nb.y, momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.y, flux_contribution_i_momentum_z.y);        

        certifMulExpans(factor, half, normal.z);
        accumulate_cmp(flux_i_density, factor, momentum_nb.z, momentum_i.z);
        accumulate_cmp(flux_i_density_energy, factor, flux_contribution_nb_density_energy.z, flux_contribution_i_density_energy.z);
        accumulate_cmp(flux_i_momentum.x, factor, flux_contribution_nb_momentum_x.z, flux_contribution_i_momentum_x.z);
        accumulate_cmp(flux_i_momentum.y, factor, flux_contribution_nb_momentum_y.z, flux_contribution_i_momentum_y.z);
        accumulate_cmp(flux_i_momentum.z, factor, flux_contribution_nb_momentum_z.z, flux_contribution_i_momentum_z.z);
      }
      else if(nb == -1) // a wing boundary
      {
        certifMulExpans(tmp1, pressure_i, normal.x);
        certifAddExpans(flux_i_momentum.x, flux_i_momentum.x, tmp1);

        certifMulExpans(tmp1, pressure_i, normal.y);
        certifAddExpans(flux_i_momentum.y, flux_i_momentum.y, tmp1);

        certifMulExpans(tmp1, pressure_i, normal.z); 
        certifAddExpans(flux_i_momentum.z, flux_i_momentum.z, tmp1);
      }
      else if(nb == -2) // a far field boundary
      {
        certifMulExpans(factor, half, normal.x);
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+0], momentum_i.x);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.x, flux_contribution_i_density_energy.x);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.x, flux_contribution_i_momentum_x.x);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.x, flux_contribution_i_momentum_y.x);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.x, flux_contribution_i_momentum_z.x);        

        certifMulExpans(factor, half, normal.y);
        accumulate_cmp(flux_i_density, factor,        ff_variable_cmp[VAR_MOMENTUM+1], momentum_i.y);
        accumulate_cmp(flux_i_density_energy, factor, ff_flux_contribution_density_energy_cmp.y, flux_contribution_i_density_energy.y);
        accumulate_cmp(flux_i_momentum.x, factor,     ff_flux_contribution_momentum_x_cmp.y, flux_contribution_i_momentum_x.y);
        accumulate_cmp(flux_i_momentum.y, factor,     ff_flux_contribution_momentum_y_cmp.y, flux_contribution_i_momentum_y.y);
        accumulate_cmp(flux_i_momentum.z, factor,     ff_flux_contribution_momentum_z_cmp.y, flux_contribution_i_momentum_z.y);        

        certifMulExpans(factor, half, normal.z);
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
    divExpans(factor, step_factors[i], tmp);
    
    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY]);
    certifAddExpans(variables[NVAR*i + VAR_DENSITY], old_variables[NVAR*i + VAR_DENSITY], tmp);

    certifMulExpans(tmp, factor, fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
    certifAddExpans(variables[NVAR*i + VAR_DENSITY_ENERGY], old_variables[NVAR*i + VAR_DENSITY_ENERGY], tmp);

    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+0)], old_variables[NVAR*i + (VAR_MOMENTUM+0)], tmp);

    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+1)], old_variables[NVAR*i + (VAR_MOMENTUM+1)], tmp);

    certifMulExpans(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
    certifAddExpans(variables[NVAR*i + (VAR_MOMENTUM+2)], old_variables[NVAR*i + (VAR_MOMENTUM+2)], tmp);
  }
}



/*
 * Main function
 */
int main(int argc, char** argv)
{
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
  certifAddExpans(gamma_minus_one, gamma_cmp, -one);  

  
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
    certifMulExpans(tmp1, ff_pressure_cmp, gamma_cmp);
    divExpans(tmp2, tmp1, ff_variable_cmp[VAR_DENSITY]);
    sqrtNewtonExpans(ff_speed_of_sound_cmp, tmp2);

    multi_prec<CPR> ff_speed_cmp, ff_mach_cmp;
    initFromDouble(&ff_mach_cmp, double(ff_mach));
    certifMulExpans(ff_speed_cmp, ff_speed_of_sound_cmp, ff_mach_cmp);

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

    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+0], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.x);
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+1], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.y);
    certifMulExpans(ff_variable_cmp[VAR_MOMENTUM+2], ff_variable_cmp[VAR_DENSITY], ff_velocity_cmp.z);

    certifMulExpans(tmp1, ff_speed_cmp, ff_speed_cmp);
    certifMulExpans(tmp2, tmp1, half);
    certifMulExpans(tmp1, ff_variable_cmp[VAR_DENSITY], tmp2);    
    divExpans(tmp2, ff_pressure_cmp, gamma_minus_one);
    certifAddExpans(ff_variable_cmp[VAR_DENSITY_ENERGY], tmp1, tmp2);
    
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
  std::cout << "\nStarting double-precision simulation..." << std::endl;
  double start = omp_get_wtime();
  for(int i = 0; i < iterations; i++)
  {
    // std::cout << "  begin iteration #" << i << "\n";
    copy<double>(old_variables, variables, nelr*NVAR);

    // for the first iteration we compute the time step
    compute_step_factor(nelr, variables, areas, step_factors);
    // DEBUG
    // compute_step_factor_cmp(nelr, variables_cmp, areas_cmp, step_factors_cmp);
    // std::cout << "   step_factors: "; 
    // compute_array_difference(step_factors, step_factors_cmp, nelr);

    for(int j = 0; j < RK; j++)
    {
      compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
      // DEBUG
      // compute_flux_cmp(nelr, elements_surrounding_elements, normals_cmp, variables_cmp, fluxes_cmp);
      // std::cout << "    compute_flux[" << j << "]: "; 
      // compute_array_difference(variables, variables_cmp, nelr);

      time_step(j, nelr, old_variables, variables, step_factors, fluxes);
      // DEBUG
      // time_step_cmp(j, nelr, old_variables_cmp, variables_cmp, step_factors_cmp, fluxes_cmp);
      // std::cout << "    time_step[" << j << "]   : "; 
      // compute_array_difference(variables, variables_cmp, nelr);
    }
  }
  double end = omp_get_wtime();
  std::cout << "Completed in " << (end-start)  / iterations << " seconds per iteration" << std::endl;
  std::cout << "Completed in " << (end-start) << " seconds total\n" << std::endl;


  // CAMPARY multi_prec simulation:
  std::cout << "Starting CAMPARY multi_prec<" << CPR << "> simulation..." << std::endl;
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
  std::cout << "Completed in " << (end-start)  / iterations << " seconds per iteration" << std::endl;
  std::cout << "Completed in " << (end-start) << " seconds total\n" << std::endl;


  // Compare the double and multi_prec solutions: 
  compute_difference(variables, variables_cmp, nel, nelr);
  std::cout << "\nVisually check output..." << std::endl;
  int start_i = 10000;
  int end_i =   10010;
  for (int i = start_i; i < end_i; ++i)
  {
    std::cout << "  double:\t variables[" << i << "] \t = \t";
    std::cout << std::fixed << std::setprecision(24) << variables[i] << std::endl;
  }
  std::cout << "\n";
  mpf_t x;
  mpf_init2(x, 256);
  for (int i = start_i; i < end_i; ++i)
  {
    convertScalarCMPToMPF(x, variables_cmp[i]);
    std::cout << "  multi_prec:\t variables_cmp[" << i << "] \t = \t";
    gmp_printf("%.24Ff\n", x);
  }  
  std::cout << "Visual check completed..." << std::endl;

  // DEBUG
  // std::cout << "Saving solution..." << std::endl;
  // dump(variables, nel, nelr);
  // std::cout << "Saved solution..." << std::endl;

  
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
