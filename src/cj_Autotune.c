/*
 * cj_Autotune.c
 * Chenhan D. Yu
 * Created: Mar 30, 1024
 *
 * Create the auto-tune information (the GEMM calculation time for a block) for the CPU and GPU
 * If auto-tuner is run first time, the CPU and GPU information will be stored as cj_autotune.bin under the ../test folder;
 * Otherwise, the CPU and GPU information will be directly read from the "../test/cj_autotune.bin", which will save some time.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include <CJ.h>
#include "cj_Blas.h"
#include "cj_Lapack.h"

static cj_Autotune *autotune;

/**
 *  @brief Report error messages. 
 *  @param *func_name function name pointer
 *  @param *msg_text error message pointer
 */
void cj_Autotune_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_AUTOTUNE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

/**
 *  @brief Automatically run a tuning script to generate the performance model 
 *         for CUBLAS library. This function will be executed only if the system
 *         supports NVIDIA CUDA.
 */
#ifdef CJ_HAVE_CUDA
void cj_Autotune_cublas () {
  int i, nb, ld;
  float time_ms = 0.0;
  float fone = 1.0, fmone = -1.0;
  double done = 1.0, dmone = -1.0;
  float *A, *B, *C;
  double *dA, *dB, *dC, *hA;
  int info;

  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);

  ld = BLOCK_SIZE*AUTOTUNE_GRID;

  cudaMalloc((void**)&A, ld*ld*sizeof(float));
  cudaMalloc((void**)&B, ld*ld*sizeof(float));
  cudaMalloc((void**)&C, ld*ld*sizeof(float));
  cudaMalloc((void**)&dA, ld*ld*sizeof(double));
  cudaMalloc((void**)&dB, ld*ld*sizeof(double));
  cudaMalloc((void**)&dC, ld*ld*sizeof(double));

  hA = (double*)malloc(sizeof(double)*ld*ld);
  for (i = 0; i < ld; i++) hA[i + ld*i] = (double) ld;  
  cudaMemcpy(dA, hA, sizeof(double)*ld*ld, cudaMemcpyHostToDevice);
  free(hA);

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    cudaEventRecord(beg, 0);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &fone, A, ld, B, ld, &fone, C, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasSgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_sgemm[i] = time_ms;

    cudaEventRecord(beg, 0);
    status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb, nb, &fmone, A, ld, &fone, C, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasSsyrk(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_ssyrk[i] = time_ms;

    cudaEventRecord(beg, 0);
    status = cublasStrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
        nb, nb, &fone, A, ld, B, ld); 
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasStrsm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_strsm[i] = time_ms;
  }

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    cudaEventRecord(beg, 0);
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &done, dA, ld, dB, ld, &done, dC, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasDgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_dgemm[i] = time_ms;

    cudaEventRecord(beg, 0);
    status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nb, nb, &dmone, dA, ld, &done, dC, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasDsyrk(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_dsyrk[i] = time_ms;

    cudaEventRecord(beg, 0);
    status = cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
        nb, nb, &done, dA, ld, dB, ld); 
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasDtrsm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_dtrsm[i] = time_ms;

    cudaEventRecord(beg, 0);
    hybrid_dpotrf (&handle, nb, dA, ld, &info);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  hybridDpotrf(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->hybrid_dpotrf[i] = time_ms;
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaEventDestroy(beg); cudaEventDestroy(end);
  cublasDestroy(handle);
}
#endif

/**
 *  @brief Automatically run a tuning script to generate the performance model 
 *         for MKL BLAS library.
 */
void cj_Autotune_mkl () {
  int i, nb, ld, info;
  float time_ms = 0.0;
  float fone = 1.0, fmone = -1.0;
  double done = 1.0, dmone = -1.0;;
  clock_t t;

  ld = BLOCK_SIZE*AUTOTUNE_GRID;

  float  *A  = malloc(ld*ld*sizeof(float));
  float  *B  = malloc(ld*ld*sizeof(float));
  float  *C  = malloc(ld*ld*sizeof(float));
  double *dA = malloc(ld*ld*sizeof(double));
  double *dB = malloc(ld*ld*sizeof(double));
  double *dC = malloc(ld*ld*sizeof(double));

  for (i = 0; i < ld; i++) {
    A[i + ld*i]  = (float)  ld;
    dA[i + ld*i] = (double) ld;
  }

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    t = clock();
    sgemm_("N", "N", &nb, &nb, &nb, &fmone, A, &ld, B, &ld, &fone, C, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Sgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_sgemm[i] = time_ms;
    t = clock();
    ssyrk_("L", "N", &nb, &nb, &fmone, A, &ld, &fone, C, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Ssyrk(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_ssyrk[i] = time_ms;
    t = clock();
    strsm_("R", "L", "T", "N", &nb, &nb, &fone, A, &ld, B, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Strsm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_strsm[i] = time_ms;
  }

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    t = clock();
    dgemm_("N", "N", &nb, &nb, &nb, &dmone, dA, &ld, dB, &ld, &done, dC, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Dgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_dgemm[i] = time_ms;
    t = clock();
    dsyrk_("L", "N", &nb, &nb, &dmone, dA, &ld, &done, dC, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Dsyrk(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_dsyrk[i] = time_ms;
    t = clock();
    dtrsm_("R", "L", "T", "N", &nb, &nb, &done, dA, &ld, dB, &ld);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Dtrsm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_dtrsm[i] = time_ms;
    t = clock();
    dpotrf_("L", &nb, dA, &ld, &info);
    time_ms = ((float) (clock() - t))/1000;
    fprintf(stderr, "  Dpotrf(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_dpotrf[i] = time_ms;
  }

  free(A);  free(B);  free(C);
  free(dA); free(dB); free(dC);
}

/**
 *  @brief Automatically run a tuning script to test the bandwidth of PCI-E bus.
 *         This function will only be executed if the system supports 
 *         NVIDIA CUDA.
 */
#ifdef CJ_HAVE_CUDA
void cj_Autotune_pci () {
  //float *dev_A, *hos_A;
  double *dev_A, *hos_A;
  float time_ms = 0.0;
  int n = BLOCK_SIZE;
  //cudaMallocHost((void**)&hos_A, n*n*sizeof(float));
  //cudaMalloc((void**)&dev_A, n*n*sizeof(float));
  cudaMallocHost((void**)&hos_A, n*n*sizeof(double));
  cudaMalloc((void**)&dev_A, n*n*sizeof(double));

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);

  cudaEventRecord(beg, 0);
  //cudaMemcpy(dev_A, hos_A, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A, hos_A, n*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  fprintf(stderr, "  bandwidth : %f ms\n", time_ms);	

  autotune->pci_bandwidth = time_ms;

  cudaEventDestroy(beg); cudaEventDestroy(end);
  cudaFreeHost(hos_A); cudaFree(dev_A);
}
#endif

/**
 *  @brief  Initial the whole autotuning scripts.
 */
void cj_Autotune_init () {
  FILE *pFile;
  autotune = (cj_Autotune*) malloc(sizeof(cj_Autotune));
  if (!autotune) cj_Autotune_error("init", "memory allocation failed.");

  pFile = fopen("cj_autotune.bin", "rb");

  if (pFile) {
    fread(autotune, sizeof(cj_Autotune), 1, pFile); 
    fclose(pFile);
  }
  else {
    cj_Autotune_mkl();
#ifdef CJ_HAVE_CUDA
    cj_Autotune_cublas();
    cj_Autotune_pci();
#endif
    pFile = fopen("cj_autotune.bin", "wb");
    fwrite(autotune, sizeof(cj_Autotune), 1, pFile); 
    fclose(pFile);
  }
}

/**
 *  @brief  Get the autotune structure pointer.
 *  @return structure pointer
 */
cj_Autotune *cj_Autotune_get_ptr () {
  return autotune;
}
