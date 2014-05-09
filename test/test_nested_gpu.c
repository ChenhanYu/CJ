/* 
 * test_nested_gpu.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

int main (int argc, char* argv[]) {

#ifdef CJ_HAVE_CUDA
  double *A11, *B11, *B21, *C11, *C21, *dA11, *dB11, *dB21, *dC11, *dC21;
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

  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  A11 = (double*) malloc(sizeof(double)*ma*na);
  B11 = (double*) malloc(sizeof(double)*mb*nb); B21 = (double*) malloc(sizeof(double)*mb*nb);
  C11 = (double*) malloc(sizeof(double)*mc*nc); C21 = (double*) malloc(sizeof(double)*mc*nc);

  cudaMalloc((void**)&dA11, sizeof(double)*ma*na);
  cudaMalloc((void**)&dB11, sizeof(double)*mb*nb); cudaMalloc((void**)&dB21, sizeof(double)*mb*nb);
  cudaMalloc((void**)&dC11, sizeof(double)*mc*nc); cudaMalloc((void**)&dC21, sizeof(double)*mc*nc);

  /* Initial A11, B11 and C11 as SPD matrix. */
  for (i = 0; i < m; i++) {
    A11[i + m*i] = (double) m;
    B11[i + m*i] = (double) m;
    C11[i + m*i] = (double) m;
  }
  cudaMemcpy(dA11, A11, sizeof(double)*ma*na, cudaMemcpyHostToDevice);
  cudaMemcpy(dB11, B11, sizeof(double)*mb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dB21, B21, sizeof(double)*mb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dC11, C11, sizeof(double)*mc*nc, cudaMemcpyHostToDevice);
  cudaMemcpy(dC21, C21, sizeof(double)*mc*nc, cudaMemcpyHostToDevice);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);
  cudaEventRecord(beg, 0);

  hybrid_dpotrf(&handle, m, dB11, m, &info);
  status = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
        m, m, &fone, dB11, m, dB21, m); 
  status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb, nb, &fmone, dB21, m, &fone, dA11, m);
  hybrid_dpotrf(&handle, m, dC11, m, &info);
  status = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
        m, m, &fone, dC11, m, dC21, m); 
  status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb, nb, &fmone, dC21, m, &fone, dA11, m);
  hybrid_dpotrf(&handle, m, dA11, m, &info);
/*
  cudaMemcpy(A11, dA11, sizeof(double)*ma*na, cudaMemcpyDeviceToHost);
  cudaMemcpy(B11, dB11, sizeof(double)*mb*nb, cudaMemcpyDeviceToHost);
  cudaMemcpy(B21, dB21, sizeof(double)*mb*nb, cudaMemcpyDeviceToHost);
  cudaMemcpy(C11, dC11, sizeof(double)*mc*nc, cudaMemcpyDeviceToHost);
  cudaMemcpy(C21, dC21, sizeof(double)*mc*nc, cudaMemcpyDeviceToHost);
*/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  fprintf(stderr, "gpu_nested(%d, %f);\n", m, time_ms/1000);

  free(A11);
  free(B11); free(B21);
  free(C11); free(C21);

  cudaFree(dA11);
  cudaFree(dB11); cudaFree(dB21);
  cudaFree(dC11); cudaFree(dC21);
#endif
  return 0;
}
