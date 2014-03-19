/*
#define AUTOTUNE_GRID 4

struct autotune_s {
  float mkl_sgemm[AUTOTUNE_GRID];
  float cublas_sgemm[AUTOTUNE_GRID];
  float pci_bandwidth;
};

typedef struct autotune_s cj_Autotune;
*/
void cj_Autotune_init ();
