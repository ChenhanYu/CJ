/* 
 * test_nested_cpu.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cj.h>

int main (int argc, char* argv[]) {
  double *A11, *B11, *B21, *C11, *C21;
  double fone = 1.0, fmone = -1.0;
  int i, m = 6144;
	for (i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-M")) {
			sscanf(argv[i+1], "%d", &m);
    }
  }
  int ma = m, na = m;
  int mb = m, nb = m;
  int mc = m, nc = m;
  int info = 0;
  float time_ms;

#ifdef CJ_HAVE_CUDA
  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);
  cudaEventRecord(beg, 0);
#endif

  A11 = (double*) malloc(sizeof(double)*ma*na);
  B11 = (double*) malloc(sizeof(double)*mb*nb); B21 = (double*) malloc(sizeof(double)*mb*nb);
  C11 = (double*) malloc(sizeof(double)*mc*nc); C21 = (double*) malloc(sizeof(double)*mc*nc);

  /* Initial A11, B11 and C11 as SPD matrix. */
  for (i = 0; i < m; i++) {
    A11[i + m*i] = (double) m;
    B11[i + m*i] = (double) m;
    C11[i + m*i] = (double) m;
  }

  dpotrf_("L", &m, B11, &m, &info);
  dtrsm_("R", "L", "T", "N", &m, &m, &fone, B11, &m, B21, &m);
  dsyrk_("L", "N", &m, &m, &fone, B21, &m, &fmone, A11, &m);
  dpotrf_("L", &m, C11, &m, &info);
  dtrsm_("R", "L", "T", "N", &m, &m, &fone, C11, &m, C21, &m);
  dsyrk_("L", "N", &m, &m, &fone, C21, &m, &fmone, A11, &m);
  dpotrf_("L", &m, A11, &m, &info);

  free(A11);
  free(B11); free(B21);
  free(C11); free(C21);

#ifdef CJ_HAVE_CUDA
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
#endif
  fprintf(stderr, "cpu_nested(%d, %f);\n", m, time_ms/1000);	

  return 0;
}
