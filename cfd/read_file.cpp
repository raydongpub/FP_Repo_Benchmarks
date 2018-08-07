#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "read_file.h"

extern "C"  void read_data_from_file(
      const char     *data_file_name,
      const int       BL,
      const int       NDIM,
      const int       NNB,
            int      *out_nel,
            int      *out_nelr,
            double  **out_areas, 
            int     **out_elements,
            double  **out_normals)
{

  // Declare new integers:
  int i, j, k;
  int nel, nelr;

  // read in domain geometry
  double* areas;
  int* elements_surrounding_elements;
  double* normals;
  {
    std::ifstream file(data_file_name);

    file >> nel;
    nelr = BL*((nel / BL )+ std::min(1, nel % BL));

    areas = (double*)malloc(nelr * sizeof(double));
    elements_surrounding_elements = (int*)malloc((nelr*NNB) * sizeof(int));
    normals = (double*)malloc((NDIM*NNB*nelr) * sizeof(double));

    // read in data
    for(i = 0; i < nel; i++)
    {
      file >> areas[i];
      for(j = 0; j < NNB; j++)
      {
        file >> elements_surrounding_elements[i*NNB + j];
        if(elements_surrounding_elements[i*NNB+j] < 0) elements_surrounding_elements[i*NNB+j] = -1;
        elements_surrounding_elements[i*NNB + j]--; //it's coming in with Fortran numbering

        for(k = 0; k < NDIM; k++)
        {
          file >>  normals[(i*NNB + j)*NDIM + k];
          normals[(i*NNB + j)*NDIM + k] = -normals[(i*NNB + j)*NDIM + k];
        }
      }
    }

    // fill in remaining data
    int last = nel-1;
    for(i = nel; i < nelr; i++)
    {
      areas[i] = areas[last];
      for(j = 0; j < NNB; j++)
      {
        // duplicate the last element
        elements_surrounding_elements[i*NNB + j] = elements_surrounding_elements[last*NNB + j];
        for(k = 0; k < NDIM; k++) normals[(i*NNB + j)*NDIM + k] = normals[(last*NNB + j)*NDIM + k];
      }
    }
  }

  // define the output
  *out_nel = nel;
  *out_nelr = nelr;
  *out_areas = areas;
  *out_elements = elements_surrounding_elements;
  *out_normals = normals;

}
