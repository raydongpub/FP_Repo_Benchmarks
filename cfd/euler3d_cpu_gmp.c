// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpfr.h>

#include "read_file.h"

struct gmp3 { mpf_t x, y, z; };

#ifndef block_length
#error "you need to define block_length"
#endif

/*
 * Options
 *
 */
#define GMP 256
#define GAMMA 1.4
// #define iterations 2000

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

// GLOBAL VARIABLES =======================================
mpf_t *ff_variable_gmp;
struct gmp3 ff_flux_contribution_momentum_x_gmp;
struct gmp3 ff_flux_contribution_momentum_y_gmp;
struct gmp3 ff_flux_contribution_momentum_z_gmp;
struct gmp3 ff_flux_contribution_density_energy_gmp;
mpf_t gamma_gmp, gamma_minus_one;
mpf_t one, half, neg_one;
mpf_t tmp, result, factor;
// ========================================================

/*
 * Generic functions
 */
void init_gmp3(struct gmp3 *new_var) 
{
  mpf_init2(new_var->x, GMP);
  mpf_init2(new_var->y, GMP);
  mpf_init2(new_var->z, GMP);
}

void clear_gmp3(struct gmp3 *old_var)
{
  mpf_clear(old_var->x);
  mpf_clear(old_var->y);
  mpf_clear(old_var->z);
}

void init_gmp(mpf_t *A, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    mpf_init2(A[i], GMP);
}

void clear_gmp(mpf_t *A, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    mpf_clear(A[i]);
}

/*
 * Element-based Cell-centered FVM solver functions
 */
void compute_flux_contribution_gmp(
  mpf_t *de_p,
  mpf_t density, 
  struct gmp3 *momentum, 
  mpf_t density_energy, 
  mpf_t pressure, 
  struct gmp3 *velocity, 
  struct gmp3 *fc_momentum_x, 
  struct gmp3 *fc_momentum_y, 
  struct gmp3 *fc_momentum_z, 
  struct gmp3 *fc_density_energy)
{
  // mpf_t tmp;
  // mpf_init2(tmp, GMP);

  mpf_mul(tmp, velocity->x, momentum->x);
  mpf_add(fc_momentum_x->x, tmp, pressure);
  mpf_mul(fc_momentum_x->y, velocity->x, momentum->y);
  mpf_mul(fc_momentum_x->z, velocity->x, momentum->z);

  mpf_set(fc_momentum_y->x, fc_momentum_x->y);
  mpf_mul(tmp, velocity->y, momentum->y);
  mpf_add(fc_momentum_y->y, tmp, pressure);
  mpf_mul(fc_momentum_y->z, velocity->y, momentum->z);

  mpf_set(fc_momentum_z->x , fc_momentum_x->z);
  mpf_set(fc_momentum_z->y, fc_momentum_y->z);
  mpf_mul(tmp, velocity->z, momentum->z);
  mpf_add(fc_momentum_z->z, tmp, pressure);

  // mpf_t de_p;
  // mpf_init2(de_p, GMP);
  mpf_add(*de_p, density_energy, pressure);
  mpf_mul(fc_density_energy->x, velocity->x, *de_p);
  mpf_mul(fc_density_energy->y, velocity->y, *de_p);
  mpf_mul(fc_density_energy->z, velocity->z, *de_p);
}

void initialize_variables_gmp(
  int nelr, 
  mpf_t *variables)
{
  int i;
  // #pragma omp parallel for default(shared) schedule(static)
  for(i = 0; i < nelr; i++)
  {
    int j;
    for(j = 0; j < NVAR; j++) 
    {
      mpf_set(variables[i*NVAR + j], ff_variable_gmp[j]);
    }
  }
}

void copy_gmp(
  mpf_t *dst, 
  mpf_t *src, 
  int N)
{
  int i;
  // #pragma omp parallel for default(shared) schedule(static)
  for(i = 0; i < N; i++)
  {
    mpf_set(dst[i], src[i]);
  }
}

void compute_velocity_gmp(
        mpf_t *density, 
  struct gmp3 *momentum, 
  struct gmp3 *out_velocity)
{
  mpf_div(out_velocity->x, momentum->x, *density);
  mpf_div(out_velocity->y, momentum->y, *density);
  mpf_div(out_velocity->z, momentum->z, *density);
}

void compute_speed_sqd_gmp(
  struct gmp3 *velocity,
  mpf_t *out_speed_sqd)
{
  // mpf_t tmp;
  // mpf_t result;

  // mpf_init2(tmp, GMP);
  // mpf_init2(result, GMP);

  mpf_set_d(result, 0.0);
  mpf_mul(tmp, velocity->x, velocity->x);
  mpf_add(result, result, tmp);
  mpf_mul(tmp, velocity->y, velocity->y);
  mpf_add(result, result, tmp);
  mpf_mul(tmp, velocity->z, velocity->z);
  mpf_add(result, result, tmp);

  mpf_set(*out_speed_sqd, result);
}

void compute_pressure_gmp(
  mpf_t *density,
  mpf_t *density_energy, 
  mpf_t *speed_sqd,
  mpf_t *out_pressure)
{
  // mpf_t tmp1, tmp2, result;
  // mpf_init2(tmp1, GMP);
  // mpf_init2(tmp2, GMP);
  // mpf_init2(result, GMP);

  // mpf_mul(tmp1, half, *density);
  mpf_mul(tmp, half, *density);
  // mpf_mul(tmp2, tmp1, *speed_sqd);
  mpf_mul(tmp, tmp, *speed_sqd);
  // mpf_sub(tmp1, *density_energy, tmp2);
  mpf_sub(tmp, *density_energy, tmp);
  // mpf_mul(result, gamma_minus_one, tmp1);
  mpf_mul(result, gamma_minus_one, tmp);

  mpf_set(*out_pressure, result);
}

void compute_speed_of_sound_gmp(
  mpf_t *density, 
  mpf_t *pressure,
  mpf_t *out_sos)
{
  // mpf_t tmp1, tmp2, result;
  // mpf_init2(tmp1, GMP);
  // mpf_init2(tmp2, GMP);
  // mpf_init2(result, GMP);

  // mpf_mul(tmp1, gamma_gmp, *pressure);
  mpf_mul(tmp, gamma_gmp, *pressure);
  // mpf_div(tmp2, tmp1, *density);
  mpf_div(tmp, tmp, *density);
  // mpf_sqrt(result, tmp2);
  mpf_sqrt(result, tmp);

  mpf_set(*out_sos, result);
}

void compute_normal_len_gmp(
  struct gmp3 *normal,
  mpf_t *out_len)
{
  mpf_set_d(result, 0.0);
  mpf_set_d(tmp, 0.0);

  mpf_mul(result, normal->x, normal->x);
  mpf_add(tmp, tmp, result);

  mpf_mul(result, normal->y, normal->y);
  mpf_add(tmp, tmp, result);

  mpf_mul(result, normal->z, normal->z);
  mpf_add(tmp, tmp, result);

  mpf_sqrt(result, tmp);
  mpf_set(*out_len, result);
}

// These accumulate functions provided:
//      a += b * (c + d)  [default]
//  or  a += b * (c - d)  [subtract_flag = 1]
void accumulate_gmp(
  mpf_t *a,
  mpf_t *b,
  mpf_t *c,
  mpf_t *d)
{
  mpf_add(tmp, *c, *d);
  mpf_mul(tmp, *b, tmp);
  mpf_add(tmp, *a, tmp);
  mpf_set(*a, tmp);
}

void accumulate_sub_gmp(
  mpf_t *a,
  mpf_t *b,
  mpf_t *c,
  mpf_t *d)
{
  mpf_sub(tmp, *c, *d);
  mpf_mul(tmp, *b, tmp);
  mpf_add(tmp, *a, tmp);
  mpf_set(*a, tmp);
}

void compute_step_factor_gmp(
  int nelr,
  mpf_t *density,
  mpf_t *density_energy,
  mpf_t *speed_sqd,
  mpf_t *pressure,
  mpf_t *speed_of_sound,
  mpf_t *denom,
  struct gmp3 *momentum,
  struct gmp3 *velocity,
  mpf_t *variables, 
  mpf_t *areas, 
  mpf_t *step_factors)
{
  int i;
  // #pragma omp parallel for default(shared) schedule(static)
  for(i = 0; i < nelr; i++)
  {
    // mpf_t density;
    // mpf_init2(density, GMP);
    mpf_set(*density, variables[NVAR*i + VAR_DENSITY]);

    // struct gmp3 momentum;
    // init_gmp3(&momentum);
    mpf_set(momentum->x, variables[NVAR*i + (VAR_MOMENTUM+0)]);
    mpf_set(momentum->y, variables[NVAR*i + (VAR_MOMENTUM+1)]);
    mpf_set(momentum->z, variables[NVAR*i + (VAR_MOMENTUM+2)]);

    // double density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
    // mpf_t density_energy;
    // mpf_init2(density_energy, GMP);
    mpf_set(*density_energy, variables[NVAR*i + VAR_DENSITY_ENERGY]);

    // double3 velocity;    
    // struct gmp3 velocity;
    // init_gmp3(&velocity);
    compute_velocity_gmp(&(*density), &(*momentum), &(*velocity));

    // double speed_sqd      = compute_speed_sqd(velocity);
    // mpf_t speed_sqd;
    // mpf_init2(speed_sqd, GMP);
    compute_speed_sqd_gmp(&(*velocity), &(*speed_sqd));

    // double pressure       = compute_pressure(density, density_energy, speed_sqd);
    // mpf_t pressure;
    // mpf_init2(pressure, GMP);
    compute_pressure_gmp(&(*density), &(*density_energy), &(*speed_sqd), &(*pressure));

    // double speed_of_sound = compute_speed_of_sound(density, pressure);
    // mpf_t speed_of_sound;
    // mpf_init2(speed_of_sound, GMP);
    compute_speed_of_sound_gmp(&(*density), &(*pressure), &(*speed_of_sound));

    // step_factors[i] = double(0.5) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
    // mpf_t tmp1, tmp2, tmp3, denom;
    // mpf_init2(tmp1, GMP);
    // mpf_init2(tmp2, GMP);
    // mpf_init2(tmp3, GMP);
    // mpf_init2(denom, GMP);
    // mpf_sqrt(tmp1, *speed_sqd);
    mpf_sqrt(tmp, *speed_sqd);
    // mpf_add(tmp2, tmp1, *speed_of_sound);
    mpf_add(*denom, tmp, *speed_of_sound);
    // mpf_sqrt(tmp3, areas[i]);
    mpf_sqrt(tmp, areas[i]);
    // mpf_mul(denom, tmp3, tmp2);
    mpf_mul(*denom, tmp, *denom);
    
    // Store result:
    mpf_div(step_factors[i], half, *denom);
  }
}

void compute_flux_gmp(
  int    nelr, 
  int   *elements_surrounding_elements, 
  mpf_t *smoothing_coefficient, 
  mpf_t *normal_len, 
  mpf_t *density_i, 
  mpf_t *density_energy_i, 
  mpf_t *de_p, 
  mpf_t *speed_sqd_i, 
  mpf_t *speed_i,
  mpf_t *pressure_i,
  mpf_t *speed_of_sound_i,
  mpf_t *flux_i_density,
  mpf_t *flux_i_density_energy,
  mpf_t *density_nb,
  mpf_t *density_energy_nb,
  mpf_t *speed_sqd_nb,
  mpf_t *speed_of_sound_nb,
  mpf_t *pressure_nb,
  struct gmp3 *normal, 
  struct gmp3 *momentum_i, 
  struct gmp3 *velocity_i, 
  struct gmp3 *flux_contribution_i_momentum_x,
  struct gmp3 *flux_contribution_i_momentum_y,
  struct gmp3 *flux_contribution_i_momentum_z,
  struct gmp3 *flux_contribution_i_density_energy,
  struct gmp3 *flux_i_momentum,
  struct gmp3 *velocity_nb,
  struct gmp3 *momentum_nb,
  struct gmp3 *flux_contribution_nb_momentum_x,
  struct gmp3 *flux_contribution_nb_momentum_y,
  struct gmp3 *flux_contribution_nb_momentum_z,
  struct gmp3 *flux_contribution_nb_density_energy,
  mpf_t *normals, 
  mpf_t *variables, 
  mpf_t *fluxes)
{
  // const double smoothing_coefficient = double(0.2f);

  int i;
  // #pragma omp parallel for default(shared) schedule(static)
  for(i = 0; i < nelr; i++)
  {
    int j, nb;

    // double3 normal; 
    // double normal_len;
    // double factor;

    // gmp3 normal; 
    // init_gmp3(&normal);

    // mpf_t normal_len, factor;
    // mpf_init2(normal_len, GMP);
    // mpf_init2(factor, GMP);
    
    // mpf_t tmp1, tmp2, tmp3;
    // mpf_init2(tmp1, GMP);
    // mpf_init2(tmp2, GMP);
    // mpf_init2(tmp3, GMP);

    // double density_i = variables[NVAR*i + VAR_DENSITY];
    // mpf_t density_i;
    // mpf_init2(density_i, GMP);
    mpf_set(*density_i, variables[NVAR*i + VAR_DENSITY]);

    // double3 momentum_i;
    // momentum_i->x = variables[NVAR*i + (VAR_MOMENTUM+0)];
    // momentum_i->y = variables[NVAR*i + (VAR_MOMENTUM+1)];
    // momentum_i->z = variables[NVAR*i + (VAR_MOMENTUM+2)];
    // gmp3 momentum_i;
    // init_gmp3(&momentum_i);
    mpf_set(momentum_i->x, variables[NVAR*i + (VAR_MOMENTUM+0)]);
    mpf_set(momentum_i->y, variables[NVAR*i + (VAR_MOMENTUM+1)]);
    mpf_set(momentum_i->z, variables[NVAR*i + (VAR_MOMENTUM+2)]);

    // double density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];
    // mpf_t density_energy_i;
    // mpf_init2(density_energy_i, GMP);
    mpf_set(*density_energy_i, variables[NVAR*i + VAR_DENSITY_ENERGY]);

    // double3 velocity_i;
    // compute_velocity(density_i, momentum_i, velocity_i);
    // struct gmp3 velocity_i;
    // init_gmp3(&velocity_i);
    compute_velocity_gmp(&(*density_i), &(*momentum_i), &(*velocity_i));

    // double speed_sqd_i                          = compute_speed_sqd(velocity_i);
    // mpf_t speed_sqd_i;
    // mpf_init2(speed_sqd_i, GMP);
    compute_speed_sqd_gmp(&(*velocity_i), &(*speed_sqd_i));

    // double speed_i                              = std::sqrt(speed_sqd_i);
    // mpf_t speed_i;
    // mpf_init2(speed_i, GMP);
    mpf_sqrt(*speed_i, *speed_sqd_i);

    // double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    // mpf_t pressure_i;
    // mpf_init2(pressure_i, GMP);
    compute_pressure_gmp(&(*density_i), &(*density_energy_i), &(*speed_sqd_i), &(*pressure_i));

    // double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
    // mpf_t speed_of_sound_i;
    // mpf_init2(speed_of_sound_i, GMP);
    compute_speed_of_sound_gmp(&(*density_i), &(*pressure_i), &(*speed_of_sound_i));

    // double3 flux_contribution_i_momentum_x, 
    //         flux_contribution_i_momentum_y, 
    //         flux_contribution_i_momentum_z,
    //         flux_contribution_i_density_energy;
    // struct gmp3 flux_contribution_i_momentum_x, 
    //             flux_contribution_i_momentum_y, 
    //             flux_contribution_i_momentum_z,
    //             flux_contribution_i_density_energy;
    // init_gmp3(&flux_contribution_i_momentum_x);
    // init_gmp3(&flux_contribution_i_momentum_y);
    // init_gmp3(&flux_contribution_i_momentum_z);
    // init_gmp3(&flux_contribution_i_density_energy);

    // compute_flux_contribution_gmp(
    //   &de_p,
    //   ff_variable_gmp[VAR_DENSITY], 
    //   &ff_momentum_gmp, 
    //   ff_variable_gmp[VAR_DENSITY_ENERGY], 
    //   ff_pressure_gmp, 
    //   &ff_velocity_gmp, 
    //   &ff_flux_contribution_momentum_x_gmp, 
    //   &ff_flux_contribution_momentum_y_gmp, 
    //   &ff_flux_contribution_momentum_z_gmp, 
    //   &ff_flux_contribution_density_energy_gmp);
    compute_flux_contribution_gmp(
      &(*de_p),
      *density_i,
      &(*momentum_i), 
      *density_energy_i,
      *pressure_i,
      &(*velocity_i), 
      &(*flux_contribution_i_momentum_x), 
      &(*flux_contribution_i_momentum_y), 
      &(*flux_contribution_i_momentum_z), 
      &(*flux_contribution_i_density_energy));

    // double flux_i_density = double(0.0);
    // mpf_t flux_i_density;
    // mpf_init2(flux_i_density, GMP);
    mpf_set_d(*flux_i_density, 0.0);

    // double3 flux_i_momentum;
    // flux_i_momentum.x = double(0.0);
    // flux_i_momentum.y = double(0.0);
    // flux_i_momentum.z = double(0.0);
    // struct gmp3 flux_i_momentum;
    // init_gmp3(&flux_i_momentum);
    mpf_set_d(flux_i_momentum->x, 0.0);
    mpf_set_d(flux_i_momentum->y, 0.0);
    mpf_set_d(flux_i_momentum->z, 0.0);

    // double flux_i_density_energy = double(0.0);
    // mpf_t flux_i_density_energy;
    // mpf_init2(flux_i_density_energy, GMP);
    mpf_set_d(*flux_i_density_energy, 0.0);

    // // double3 velocity_nb;
    // struct gmp3 velocity_nb;
    // init_gmp3(&velocity_nb);

    // // double density_nb, density_energy_nb;
    // mpf_t density_nb, density_energy_nb;
    // mpf_init2(density_nb, GMP);
    // mpf_init2(density_energy_nb, GMP);

    // // double3 momentum_nb;
    // struct gmp3 momentum_nb;
    // init_gmp3(&momentum_nb);

    // // double3 flux_contribution_nb_momentum_x, 
    // //         flux_contribution_nb_momentum_y, 
    // //         flux_contribution_nb_momentum_z;
    // //         flux_contribution_nb_density_energy;
    // struct gmp3  flux_contribution_nb_momentum_x, 
    //       flux_contribution_nb_momentum_y, 
    //       flux_contribution_nb_momentum_z,
    //       flux_contribution_nb_density_energy;
    // init_gmp3(&flux_contribution_nb_momentum_x);
    // init_gmp3(&flux_contribution_nb_momentum_y);
    // init_gmp3(&flux_contribution_nb_momentum_z);
    // init_gmp3(&flux_contribution_nb_density_energy);

    // // double speed_sqd_nb, speed_of_sound_nb, pressure_nb;
    // mpf_t speed_sqd_nb, speed_of_sound_nb, pressure_nb;
    // mpf_init2(speed_sqd_nb, GMP);
    // mpf_init2(speed_of_sound_nb, GMP);
    // mpf_init2(pressure_nb, GMP);

    for(j = 0; j < NNB; j++)
    {
      nb = elements_surrounding_elements[i*NNB + j];
      // normal.x = normals[(i*NNB + j)*NDIM + 0];
      // normal.y = normals[(i*NNB + j)*NDIM + 1];
      // normal.z = normals[(i*NNB + j)*NDIM + 2];
      mpf_set(normal->x, normals[(i*NNB + j)*NDIM + 0]);
      mpf_set(normal->y, normals[(i*NNB + j)*NDIM + 1]);
      mpf_set(normal->z, normals[(i*NNB + j)*NDIM + 2]);

      // normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
      compute_normal_len_gmp(&(*normal), &(*normal_len));

      if(nb >= 0)   // a legitimate neighbor
      {
        mpf_set(*density_nb, variables[nb*NVAR + VAR_DENSITY]);
        mpf_set(momentum_nb->x, variables[nb*NVAR + (VAR_MOMENTUM+0)]);
        mpf_set(momentum_nb->y, variables[nb*NVAR + (VAR_MOMENTUM+1)]);
        mpf_set(momentum_nb->z, variables[nb*NVAR + (VAR_MOMENTUM+2)]);
        mpf_set(*density_energy_nb, variables[nb*NVAR + VAR_DENSITY_ENERGY]); 

        compute_velocity_gmp(&(*density_nb), &(*momentum_nb), &(*velocity_nb));
        compute_speed_sqd_gmp(&(*velocity_nb), &(*speed_sqd_nb));
        compute_pressure_gmp(density_nb, density_energy_nb, speed_sqd_nb, pressure_nb);
        compute_speed_of_sound_gmp(density_nb, pressure_nb, speed_of_sound_nb);
        
        compute_flux_contribution_gmp(
          &(*de_p),
          *density_nb, 
          &(*momentum_nb), 
          *density_energy_nb, 
          *pressure_nb, 
          &(*velocity_nb), 
          &(*flux_contribution_nb_momentum_x), 
          &(*flux_contribution_nb_momentum_y), 
          &(*flux_contribution_nb_momentum_z), 
          &(*flux_contribution_nb_density_energy));

        // artificial viscosity
        // factor = -normal_len*smoothing_coefficient*double(0.5)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
        
        // mpf_sqrt(tmp1, speed_sqd_nb);
        // mpf_add(tmp2, *speed_i, tmp1);
        // mpf_add(tmp2, tmp2, *speed_of_sound_i);
        // mpf_add(tmp2, tmp2, speed_of_sound_nb);
        // mpf_mul(tmp1, *normal_len, smoothing_coefficient);
        // mpf_mul(tmp3, tmp1, half);
        // mpf_mul(tmp1, tmp3, tmp2);
        // mpf_mul(factor, tmp1, neg_one);

        mpf_sqrt(tmp, *speed_sqd_nb);
        mpf_add(tmp, *speed_i, tmp);
        mpf_add(tmp, tmp, *speed_of_sound_i);
        mpf_add(tmp, tmp, *speed_of_sound_nb);
        mpf_mul(factor, *normal_len, *smoothing_coefficient);
        mpf_mul(factor, factor, half);
        mpf_mul(factor, factor, tmp);
        mpf_mul(factor, factor, neg_one);

        accumulate_sub_gmp(&(*flux_i_density), &factor, &(*density_i), &(*density_nb));
        accumulate_sub_gmp(&(*flux_i_density_energy), &factor, &(*density_energy_i), &(*density_energy_nb));
        accumulate_sub_gmp(&(flux_i_momentum->x), &factor, &(momentum_i->x), &(momentum_nb->x));
        accumulate_sub_gmp(&(flux_i_momentum->y), &factor, &(momentum_i->y), &(momentum_nb->y));
        accumulate_sub_gmp(&(flux_i_momentum->z), &factor, &(momentum_i->z), &(momentum_nb->z));

        // accumulate cell-centered fluxes
        mpf_mul(factor, half, normal->x);
        accumulate_gmp(&(*flux_i_density), &factor, &(momentum_nb->x), &(momentum_i->x));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(flux_contribution_nb_density_energy->x), &(flux_contribution_i_density_energy->x));
        accumulate_gmp(&(flux_i_momentum->x), &factor, &(flux_contribution_nb_momentum_x->x), &(flux_contribution_i_momentum_x->x));
        accumulate_gmp(&(flux_i_momentum->y), &factor, &(flux_contribution_nb_momentum_y->x), &(flux_contribution_i_momentum_y->x));
        accumulate_gmp(&(flux_i_momentum->z), &factor, &(flux_contribution_nb_momentum_z->x), &(flux_contribution_i_momentum_z->x));

        mpf_mul(factor, half, normal->y);
        accumulate_gmp(&(*flux_i_density), &factor, &(momentum_nb->y), &(momentum_i->y));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(flux_contribution_nb_density_energy->y), &(flux_contribution_i_density_energy->y));
        accumulate_gmp(&(flux_i_momentum->x), &factor, &(flux_contribution_nb_momentum_x->y), &(flux_contribution_i_momentum_x->y));
        accumulate_gmp(&(flux_i_momentum->y), &factor, &(flux_contribution_nb_momentum_y->y), &(flux_contribution_i_momentum_y->y));
        accumulate_gmp(&(flux_i_momentum->z), &factor, &(flux_contribution_nb_momentum_z->y), &(flux_contribution_i_momentum_z->y));

        mpf_mul(factor, half, normal->z);
        accumulate_gmp(&(*flux_i_density), &factor, &(momentum_nb->z), &(momentum_i->z));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(flux_contribution_nb_density_energy->z), &(flux_contribution_i_density_energy->z));
        accumulate_gmp(&(flux_i_momentum->x), &factor, &(flux_contribution_nb_momentum_x->z), &(flux_contribution_i_momentum_x->z));
        accumulate_gmp(&(flux_i_momentum->y), &factor, &(flux_contribution_nb_momentum_y->z), &(flux_contribution_i_momentum_y->z));
        accumulate_gmp(&(flux_i_momentum->z), &factor, &(flux_contribution_nb_momentum_z->z), &(flux_contribution_i_momentum_z->z));
      }
      else if(nb == -1) // a wing boundary
      {
        mpf_mul(tmp, *pressure_i, normal->x);
        mpf_add(flux_i_momentum->x, flux_i_momentum->x, tmp);

        mpf_mul(tmp, *pressure_i, normal->y);
        mpf_add(flux_i_momentum->y, flux_i_momentum->y, tmp);

        mpf_mul(tmp, *pressure_i, normal->z); 
        mpf_add(flux_i_momentum->z, flux_i_momentum->z, tmp);
      }
      else if(nb == -2) // a far field boundary
      {
        mpf_mul(factor, half, normal->x);
        accumulate_gmp(&(*flux_i_density), &factor,        &ff_variable_gmp[VAR_MOMENTUM+0], &(momentum_i->x));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(ff_flux_contribution_density_energy_gmp.x), &(flux_contribution_i_density_energy->x));
        accumulate_gmp(&(flux_i_momentum->x), &factor,     &(ff_flux_contribution_momentum_x_gmp.x), &(flux_contribution_i_momentum_x->x));
        accumulate_gmp(&(flux_i_momentum->y), &factor,     &(ff_flux_contribution_momentum_y_gmp.x), &(flux_contribution_i_momentum_y->x));
        accumulate_gmp(&(flux_i_momentum->z), &factor,     &(ff_flux_contribution_momentum_z_gmp.x), &(flux_contribution_i_momentum_z->x));        

        mpf_mul(factor, half, normal->y);
        accumulate_gmp(&(*flux_i_density), &factor,        &ff_variable_gmp[VAR_MOMENTUM+1], &(momentum_i->y));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(ff_flux_contribution_density_energy_gmp.y), &(flux_contribution_i_density_energy->y));
        accumulate_gmp(&(flux_i_momentum->x), &factor,     &(ff_flux_contribution_momentum_x_gmp.y), &(flux_contribution_i_momentum_x->y));
        accumulate_gmp(&(flux_i_momentum->y), &factor,     &(ff_flux_contribution_momentum_y_gmp.y), &(flux_contribution_i_momentum_y->y));
        accumulate_gmp(&(flux_i_momentum->z), &factor,     &(ff_flux_contribution_momentum_z_gmp.y), &(flux_contribution_i_momentum_z->y));        

        mpf_mul(factor, half, normal->z);
        accumulate_gmp(&(*flux_i_density), &factor,        &ff_variable_gmp[VAR_MOMENTUM+2], &(momentum_i->z));
        accumulate_gmp(&(*flux_i_density_energy), &factor, &(ff_flux_contribution_density_energy_gmp.z), &(flux_contribution_i_density_energy->z));
        accumulate_gmp(&(flux_i_momentum->x), &factor,     &(ff_flux_contribution_momentum_x_gmp.z), &(flux_contribution_i_momentum_x->z));
        accumulate_gmp(&(flux_i_momentum->y), &factor,     &(ff_flux_contribution_momentum_y_gmp.z), &(flux_contribution_i_momentum_y->z));
        accumulate_gmp(&(flux_i_momentum->z), &factor,     &(ff_flux_contribution_momentum_z_gmp.z), &(flux_contribution_i_momentum_z->z));
      }
    }
    mpf_set(fluxes[i*NVAR + VAR_DENSITY], *flux_i_density);
    mpf_set(fluxes[i*NVAR + (VAR_MOMENTUM+0)], flux_i_momentum->x);
    mpf_set(fluxes[i*NVAR + (VAR_MOMENTUM+1)], flux_i_momentum->y);
    mpf_set(fluxes[i*NVAR + (VAR_MOMENTUM+2)], flux_i_momentum->z);
    mpf_set(fluxes[i*NVAR + VAR_DENSITY_ENERGY], *flux_i_density_energy);
  }
}

void time_step_gmp(
  int j, 
  int nelr, 
  mpf_t *old_variables,
  mpf_t *variables, 
  mpf_t *step_factors, 
  mpf_t *fluxes)
{
  int i;
  // #pragma omp parallel for  default(shared) schedule(static)
  for(i = 0; i < nelr; i++)
  {
    // mpf_t factor;
    // mpf_init2(factor, GMP);

    double rk = (RK+1-j);
    mpf_set_d(tmp, rk);
    mpf_div(factor, step_factors[i], tmp);

    mpf_mul(tmp, factor, fluxes[NVAR*i + VAR_DENSITY]);
    mpf_add(variables[NVAR*i + VAR_DENSITY], old_variables[NVAR*i + VAR_DENSITY], tmp);

    mpf_mul(tmp, factor, fluxes[NVAR*i + VAR_DENSITY_ENERGY]);
    mpf_add(variables[NVAR*i + VAR_DENSITY_ENERGY], old_variables[NVAR*i + VAR_DENSITY_ENERGY], tmp);

    mpf_mul(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+0)]);
    mpf_add(variables[NVAR*i + (VAR_MOMENTUM+0)], old_variables[NVAR*i + (VAR_MOMENTUM+0)], tmp);

    mpf_mul(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+1)]);
    mpf_add(variables[NVAR*i + (VAR_MOMENTUM+1)], old_variables[NVAR*i + (VAR_MOMENTUM+1)], tmp);

    mpf_mul(tmp, factor, fluxes[NVAR*i + (VAR_MOMENTUM+2)]);
    mpf_add(variables[NVAR*i + (VAR_MOMENTUM+2)], old_variables[NVAR*i + (VAR_MOMENTUM+2)], tmp);
  }
}

void dump_gmp(
  mpf_t *variables, 
  int nel, 
  int nelr)
{
  {
    // std::ofstream file("density");
    FILE *file = fopen("density_gmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    int i;
    for(i = 0; i < nel; i++) 
    {
      // file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY] << std::endl;
      gmp_fprintf(file, "%.80Fe\n", variables[i*NVAR + VAR_DENSITY]);
    }
    fclose(file);
  }

  {
    // std::ofstream file("momentum");
    FILE *file = fopen("momentum_gmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    int i, j;
    for(i = 0; i < nel; i++)
    {
      for(j = 0; j != NDIM; j++) 
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
    FILE *file = fopen("density_energy_gmp.txt", "w");
    // file << nel << " " << nelr << std::endl;
    fprintf(file, "%d %d\n", nel, nelr);
    int i;
    for(i = 0; i < nel; i++) 
    {
      // file << std::fixed << std::setprecision(24) << variables[i*NVAR + VAR_DENSITY_ENERGY] << std::endl;
      gmp_fprintf(file, "%.80Fe\n", variables[i*NVAR + VAR_DENSITY_ENERGY]);
    }
    fclose(file);
  }
}





// template <typename T>
// T* alloc(int N)
// {
//   return new T[N];
// }

// template <typename T>
// void dealloc(T* array)
// {
//   delete[] array;
// }

// template <typename T>
// void copy(T* dst, T* src, int N)
// {
//   #pragma omp parallel for default(shared) schedule(static)
//   for(int i = 0; i < N; i++)
//   {
//     dst[i] = src[i];
//   }
// }

// void init_gmp3(struct gmp3 *new_struct)
// {
//   mpf_init2(new_struct->x, GMP);
//   mpf_init2(new_struct->y, GMP);
//   mpf_init2(new_struct->z, GMP);
// }

















/*
 * Main function
 */
int main(int argc, char** argv)
{
  // std::cout << "\nGNU MPFR SOLUTION: \n";
  printf("\nGNU MPFR SOLUTION: \n");
  // Manage command line input arguments
  if (argc < 2)
  {
    printf("specify data file name\n");
    printf("example usage: [progam.ex] [data_file] [iterations]\n");
    printf("$ ./euler3d_cpu_mixed.ex ./data/fvcorr.domn.097K 2000\n");
    return 1;
  }
  else if (argc < 3)
  {
    printf("specify number of iterations\n");
    return 1;
  }
  const char* data_file_name = argv[1];
  const int iterations = atoi(argv[2]);

  // Initialize the global variables:
  ff_variable_gmp = (mpf_t*)malloc(NVAR * sizeof(mpf_t));
  init_gmp(ff_variable_gmp, NVAR);
  init_gmp3(&ff_flux_contribution_momentum_x_gmp);
  init_gmp3(&ff_flux_contribution_momentum_y_gmp);
  init_gmp3(&ff_flux_contribution_momentum_z_gmp);
  init_gmp3(&ff_flux_contribution_density_energy_gmp);

  // DEBUG
  // mpf_set_d(ff_flux_contribution_momentum_x_gmp.x, 1.0);
  // mpf_set_d(ff_flux_contribution_momentum_x_gmp.y, 2.0);
  // mpf_set_d(ff_flux_contribution_momentum_x_gmp.z, 4.0);
  // gmp_printf("%.80Ff\n", ff_flux_contribution_momentum_x_gmp.x);
  // gmp_printf("%.80Ff\n", ff_flux_contribution_momentum_x_gmp.y);
  // gmp_printf("%.80Ff\n", ff_flux_contribution_momentum_x_gmp.z);

  // Initialize the global constants:
  mpf_init2(gamma_gmp, GMP);
  mpf_set_d(gamma_gmp, GAMMA);
  mpf_init2(one, GMP);
  mpf_set_d(one, 1.0);
  mpf_init2(neg_one, GMP);
  mpf_set_d(neg_one, -1.0);
  mpf_init2(half, GMP);
  mpf_set_d(half, 0.5);
  mpf_init2(gamma_minus_one, GMP);
  mpf_sub(gamma_minus_one, gamma_gmp, one);

  // Initialize new global variables: 
  mpf_init2(tmp, GMP);
  mpf_set_d(tmp, 0.0);
  mpf_init2(result, GMP);
  mpf_set_d(result, 0.0);
  mpf_init2(factor, GMP);
  mpf_set_d(factor, 0.0);

  // DEBUG
  // gmp_printf("half  = %.80Ff\n", half);
  // gmp_printf("one   = %.80Ff\n", one);
  // gmp_printf("-one  = %.80Ff\n", neg_one);
  // gmp_printf("gamma = %.80Ff\n", gamma_gmp);
  // gmp_printf("(g-1) = %.80Ff\n", gamma_minus_one);
  // printf("\n");

  // Set far field conditions:
  mpf_t tmp1, tmp2, de_p;
  mpf_init2(tmp1, GMP);
  mpf_init2(tmp2, GMP);
  mpf_init2(de_p, GMP);
  {
    const double angle_of_attack = (double)(3.1415926535897931 / 180.0) * (double)(deg_angle_of_attack);
    const double angle_tol = pow(10.0, -14);

    mpf_set_d(ff_variable_gmp[VAR_DENSITY], 1.4);

    mpf_t ff_pressure_gmp;
    mpf_init2(ff_pressure_gmp, GMP);
    mpf_set_d(ff_pressure_gmp, 1.0);

    mpf_t ff_speed_of_sound_gmp;
    mpf_init2(ff_speed_of_sound_gmp, GMP);
    mpf_mul(tmp1, gamma_gmp, ff_pressure_gmp);
    mpf_div(tmp2, tmp1, ff_variable_gmp[VAR_DENSITY]);
    mpf_sqrt(ff_speed_of_sound_gmp, tmp2);
    
    mpf_t ff_speed_gmp, ff_mach_gmp;
    mpf_init2(ff_mach_gmp, GMP);
    mpf_set_d(ff_mach_gmp, (double)ff_mach);
    mpf_init2(ff_speed_gmp, GMP);
    mpf_mul(ff_speed_gmp, ff_mach_gmp, ff_speed_of_sound_gmp);

    // We use this if-else statement to avoid the FLOP needed to compute
    // the x-and y-direction velocity when the attack angle is near zero.
    struct gmp3 ff_velocity_gmp;
    init_gmp3(&ff_velocity_gmp);
    if ( angle_of_attack < angle_tol )
    {
      mpf_set(ff_velocity_gmp.x, ff_speed_gmp);
      mpf_set_d(ff_velocity_gmp.y, 0.0);
    }
    else
    {
      printf("***Error: GNU MPFR solution does not support cos and sin at this time!\n");
      return 1;
    }
    mpf_set_d(ff_velocity_gmp.z, 0.0);

    mpf_mul(ff_variable_gmp[VAR_MOMENTUM+0], ff_variable_gmp[VAR_DENSITY], ff_velocity_gmp.x);
    mpf_mul(ff_variable_gmp[VAR_MOMENTUM+1], ff_variable_gmp[VAR_DENSITY], ff_velocity_gmp.y);
    mpf_mul(ff_variable_gmp[VAR_MOMENTUM+2], ff_variable_gmp[VAR_DENSITY], ff_velocity_gmp.z);
    
    mpf_mul(tmp1, ff_speed_gmp, ff_speed_gmp);
    mpf_mul(tmp2, tmp1, half);
    mpf_mul(tmp1, ff_variable_gmp[VAR_DENSITY], tmp2);
    mpf_div(tmp2, ff_pressure_gmp, gamma_minus_one);
    mpf_add(ff_variable_gmp[VAR_DENSITY_ENERGY], tmp1, tmp2);

    struct gmp3 ff_momentum_gmp;
    init_gmp3(&ff_momentum_gmp);
    mpf_set(ff_momentum_gmp.x, ff_variable_gmp[VAR_MOMENTUM+0]);
    mpf_set(ff_momentum_gmp.y, ff_variable_gmp[VAR_MOMENTUM+1]);
    mpf_set(ff_momentum_gmp.z, ff_variable_gmp[VAR_MOMENTUM+2]);

    compute_flux_contribution_gmp(
      &de_p,
      ff_variable_gmp[VAR_DENSITY], 
      &ff_momentum_gmp, 
      ff_variable_gmp[VAR_DENSITY_ENERGY], 
      ff_pressure_gmp, 
      &ff_velocity_gmp, 
      &ff_flux_contribution_momentum_x_gmp, 
      &ff_flux_contribution_momentum_y_gmp, 
      &ff_flux_contribution_momentum_z_gmp, 
      &ff_flux_contribution_density_energy_gmp);

    // Clear temporary variables:
    // mpf_clear(ff_pressure_gmp);
    // mpf_clear(ff_speed_of_sound_gmp);
    // mpf_clear(ff_speed_gmp);
    // mpf_clear(ff_mach_gmp);
    // clear_gmp3(&ff_velocity_gmp);
    // clear_gmp3(&ff_momentum_gmp);
  }

  // Declare new integers:
  int i, j;
  int nel, nelr;

  // DEBUG
  // for (i = 0; i < NVAR; ++i)
  // {
  //   gmp_printf("ff_var[%d] = %.80Ff\n", i, ff_variable_gmp[i]);
  // }

  // Read in domain geometry from data file:
  double* areas;
  int* elements_surrounding_elements;
  double* normals;
  const int BL = (int)block_length;
  read_data_from_file(
      data_file_name,
      BL,
      NDIM,
      NNB,
      &nel,
      &nelr,
      &areas,
      &elements_surrounding_elements,
      &normals);

  // Convert areas and normals to GMP types:
  mpf_t* areas_gmp;
  areas_gmp = (mpf_t*)malloc(nelr * sizeof(mpf_t));
  init_gmp(areas_gmp, nelr);
  for (i = 0; i < nelr; ++i)
  {
    mpf_set_d(areas_gmp[i], areas[i]);
  }
  mpf_t* normals_gmp;
  normals_gmp = (mpf_t*)malloc((NDIM*NNB*nelr) * sizeof(mpf_t));
  init_gmp(normals_gmp, NDIM*NNB*nelr);
  for (i = 0; i < NDIM*NNB*nelr; ++i)
  {
    mpf_set_d(normals_gmp[i], normals[i]);
  }
  free(areas);
  free(normals);

  // Create arrays and set initial conditions:
  mpf_t* variables_gmp;
  variables_gmp = (mpf_t*)malloc((nelr*NVAR) * sizeof(mpf_t));
  init_gmp(variables_gmp, nelr*NVAR);
  initialize_variables_gmp(nelr, variables_gmp);

  mpf_t* old_variables_gmp;
  old_variables_gmp = (mpf_t*)malloc((nelr*NVAR) * sizeof(mpf_t));
  init_gmp(old_variables_gmp, nelr*NVAR);

  mpf_t* fluxes_gmp;
  fluxes_gmp = (mpf_t*)malloc((nelr*NVAR) * sizeof(mpf_t));
  init_gmp(fluxes_gmp, nelr*NVAR);
  
  mpf_t* step_factors_gmp;
  step_factors_gmp = (mpf_t*)malloc((nelr) * sizeof(mpf_t));
  init_gmp(step_factors_gmp, nelr);


  // ================================================================
  // Temporary variables declared only ONCE outside loop:

    // 1. compute_step_factor_gmp
    mpf_t tmp_density, tmp_density_energy, tmp_speed_sqd;
    mpf_t tmp_pressure, tmp_speed_of_sound, tmp_denom;
    mpf_init2(tmp_density, GMP);
    mpf_init2(tmp_density_energy, GMP);
    mpf_init2(tmp_speed_sqd, GMP);
    mpf_init2(tmp_pressure, GMP);
    mpf_init2(tmp_speed_of_sound, GMP);
    mpf_init2(tmp_denom, GMP);
    struct gmp3 tmp_momentum;
    init_gmp3(&tmp_momentum);   
    struct gmp3 tmp_velocity;
    init_gmp3(&tmp_velocity);

    // 2. compute_flux_gmp
    mpf_t tmp_smooth_coef;
    mpf_init2(tmp_smooth_coef, GMP);
    mpf_set_d(tmp_smooth_coef, (double)0.2);
    
    mpf_t tmp_normal_len;
    mpf_init2(tmp_normal_len, GMP);

    mpf_t tmp_speed;
    mpf_init2(tmp_speed, GMP);    
    mpf_t tmp_flux_density;
    mpf_init2(tmp_flux_density, GMP);
    mpf_t tmp_flux_density_energy;
    mpf_init2(tmp_flux_density_energy, GMP);    

    struct gmp3 tmp_normal; 
    struct gmp3 tmp_flux_contribution_i_momentum_x;
    struct gmp3 tmp_flux_contribution_i_momentum_y;
    struct gmp3 tmp_flux_contribution_i_momentum_z;
    struct gmp3 tmp_flux_contribution_i_density_energy;    
    struct gmp3 tmp_flux_momentum;
    init_gmp3(&tmp_flux_momentum);
    init_gmp3(&tmp_normal);
    init_gmp3(&tmp_flux_contribution_i_momentum_x);
    init_gmp3(&tmp_flux_contribution_i_momentum_y);
    init_gmp3(&tmp_flux_contribution_i_momentum_z);
    init_gmp3(&tmp_flux_contribution_i_density_energy);

    mpf_t tmp_density_nb;
    mpf_t tmp_density_energy_nb;
    mpf_t tmp_speed_sqd_nb;
    mpf_t tmp_speed_of_sound_nb;
    mpf_t tmp_pressure_nb;
    mpf_init2(tmp_density_nb, GMP);
    mpf_init2(tmp_density_energy_nb, GMP);
    mpf_init2(tmp_speed_sqd_nb, GMP);
    mpf_init2(tmp_speed_of_sound_nb, GMP);
    mpf_init2(tmp_pressure_nb, GMP);

    struct gmp3 tmp_velocity_nb;
    struct gmp3 tmp_momentum_nb;
    struct gmp3 tmp_flux_contribution_nb_momentum_x;
    struct gmp3 tmp_flux_contribution_nb_momentum_y;
    struct gmp3 tmp_flux_contribution_nb_momentum_z;
    struct gmp3 tmp_flux_contribution_nb_density_energy;
    init_gmp3(&tmp_velocity_nb);
    init_gmp3(&tmp_momentum_nb);
    init_gmp3(&tmp_flux_contribution_nb_momentum_x);
    init_gmp3(&tmp_flux_contribution_nb_momentum_y);
    init_gmp3(&tmp_flux_contribution_nb_momentum_z);
    init_gmp3(&tmp_flux_contribution_nb_density_energy);

    // 3. time_step_gmp    
    //   (none)
  // ================================================================

  // Begin iterations
  printf("\nStarting GNU MPFR simulation...\n");
  double start = omp_get_wtime();
  for(i = 0; i < iterations; i++)
  {
    printf("  begin iteration #%d\n", i);
    copy_gmp(old_variables_gmp, variables_gmp, nelr*NVAR);

    // for the first iteration we compute the time step
    compute_step_factor_gmp(nelr,
                            &tmp_density,   &tmp_density_energy,  &tmp_speed_sqd, 
                            &tmp_pressure,  &tmp_speed_of_sound,  &tmp_denom,
                            &tmp_momentum,  &tmp_velocity,
                            variables_gmp,  areas_gmp,            step_factors_gmp);
    
    // DEBUG
    // for (int j = 0; j < nelr; j++)
    // {
    //   gmp_printf("sf[%d] = %.80Ff\n", j, step_factors_gmp[j]);
    // }

    for(j = 0; j < RK; j++)
    {
      compute_flux_gmp(nelr,     elements_surrounding_elements,
        &tmp_smooth_coef,       &tmp_normal_len,          &tmp_density,       
        &tmp_density_energy,    &de_p,                    &tmp_speed_sqd,
        &tmp_speed,             &tmp_pressure,            &tmp_speed_of_sound,
        &tmp_flux_density,      &tmp_flux_density_energy, &tmp_density_nb,
        &tmp_density_energy_nb, &tmp_speed_sqd_nb,        &tmp_speed_of_sound_nb,
        &tmp_pressure_nb,
        &tmp_normal,        
        &tmp_momentum,          
        &tmp_velocity,
        &tmp_flux_contribution_i_momentum_x,
        &tmp_flux_contribution_i_momentum_y,
        &tmp_flux_contribution_i_momentum_z,
        &tmp_flux_contribution_i_density_energy,   
        &tmp_flux_momentum, 
        &tmp_velocity_nb,       
        &tmp_momentum_nb,
        &tmp_flux_contribution_nb_momentum_x,
        &tmp_flux_contribution_nb_momentum_y,
        &tmp_flux_contribution_nb_momentum_z,
        &tmp_flux_contribution_nb_density_energy,
        normals_gmp,        variables_gmp,          fluxes_gmp);

      time_step_gmp(j, nelr, 
                    old_variables_gmp, variables_gmp, step_factors_gmp, fluxes_gmp);
    }
  }
  double end = omp_get_wtime();
  printf("Completed in %f seconds per iteration\n", ((end-start) / iterations));
  printf("Completed in %f seconds total\n\n", (end-start));


  printf("Visually check output...\n");
  for (i = 10000; i < 10010; ++i)
  {
    printf("  GNU MPFR:  variables[%d] = \t", i);
    gmp_printf("%.80Ff\n", variables_gmp[i]);
  }
  printf("Visual check completed...\n");

  printf("Saving solution...\n");
  dump_gmp(variables_gmp, nel, nelr);
  printf("Saved solution...\n");

  printf("\nCleaning up...\n");
  free(elements_surrounding_elements);

  clear_gmp(areas_gmp,          nelr );
  clear_gmp(normals_gmp,        (NDIM*NNB*nelr) );
  clear_gmp(variables_gmp,      (nelr*NVAR) );
  clear_gmp(old_variables_gmp,  (nelr*NVAR) );
  clear_gmp(fluxes_gmp,         (nelr*NVAR) );
  clear_gmp(step_factors_gmp,   nelr );

  free(areas_gmp);
  free(normals_gmp);
  free(variables_gmp);
  free(old_variables_gmp);
  free(fluxes_gmp);
  free(step_factors_gmp);  

  // (end of main program)
  printf("Done...\n");
  return 0;
}
