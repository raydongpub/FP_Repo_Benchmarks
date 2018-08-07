#ifdef __cplusplus
extern "C"
#endif


void read_data_from_file(
      const char     *data_file_name,
      const int       BL,
      const int       NDIM,
      const int       NNB,
            int      *out_nel,
            int      *out_nelr,
            double  **out_areas, 
            int     **out_elements,
            double  **out_normals);
