#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "cj_Object.h"
#include "cj_Autotune.h"

static cj_Autotune *autotune;

void cj_Autotune_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_AUTOTUNE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Autotune_cublas () {
  int i, nb, ld;
  float time_ms = 0.0;
  float fone = 1.0;
  float *A, *B, *C;

  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);

  ld = BLOCK_SIZE*AUTOTUNE_GRID;

  cudaMalloc((void**)&A, ld*ld*sizeof(float));
  cudaMalloc((void**)&B, ld*ld*sizeof(float));
  cudaMalloc((void**)&C, ld*ld*sizeof(float));

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    cudaEventRecord(beg, 0);
    //cublasSgemm('N', 'N', nb, nb, nb, 1.0, A, ld, B, ld, 1.0, C, ld);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &fone, A, ld, B, ld, &fone, C, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasSgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_sgemm[i] = time_ms;
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  cudaEventDestroy(beg); cudaEventDestroy(end);
  cublasDestroy(handle);
}

void cj_Autotune_mkl () {
  int i, nb, ld;
  float time_ms = 0.0;
  float fone = 1.0;

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);

  ld = BLOCK_SIZE*AUTOTUNE_GRID;

  float *A = malloc(ld*ld*sizeof(float));
  float *B = malloc(ld*ld*sizeof(float));
  float *C = malloc(ld*ld*sizeof(float));


  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    cudaEventRecord(beg, 0);
    sgemm_("N", "N", &nb, &nb, &nb, &fone, A, &ld, B, &ld, &fone, C, &ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  Sgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_sgemm[i] = time_ms;
  }

  free(A); free(B); free(C);
  cudaEventDestroy(beg); cudaEventDestroy(end);
}

void cj_Autotune_pci () {
  float *dev_A, *hos_A;
  float time_ms = 0.0;
  int n = BLOCK_SIZE;
  cudaMallocHost((void**)&hos_A, n*n*sizeof(float));
  cudaMalloc((void**)&dev_A, n*n*sizeof(float));

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);

  cudaEventRecord(beg, 0);
  cudaMemcpy(dev_A, hos_A, n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  fprintf(stderr, "  bandwidth : %f ms\n", time_ms);	

  autotune->pci_bandwidth = time_ms;

  cudaEventDestroy(beg); cudaEventDestroy(end);
  cudaFreeHost(hos_A); cudaFree(dev_A);
}

void cj_Autotune_init () {
  FILE *pFile;
  autotune = (cj_Autotune*) malloc(sizeof(cj_Autotune));
  if (!autotune) cj_Autotune_error("init", "memory allocation failed.");

  pFile = fopen("cj_autotune.bin", "rb");

  if (pFile) {
    //fread(autotune->cublas_sgemm, sizeof(float), AUTOTUNE_GRID, pFile); 
    //fread(autotune->mkl_sgemm, sizeof(float), AUTOTUNE_GRID, pFile); 
    //fread(&autotune->pci_bandwidth, sizeof(float), 1, pFile); 
    fread(autotune, sizeof(cj_Autotune), 1, pFile); 
    fclose(pFile);
  }
  else {
    cj_Autotune_cublas();
    cj_Autotune_mkl();
    cj_Autotune_pci();

    pFile = fopen("cj_autotune.bin", "wb");
    /*
    fwrite(autotune->cublas_sgemm, sizeof(float), AUTOTUNE_GRID, pFile); 
    fwrite(autotune->mkl_sgemm, sizeof(float), AUTOTUNE_GRID, pFile); 
    fwrite(&autotune->pci_bandwidth, sizeof(float), 1, pFile); 
    */
    fwrite(autotune, sizeof(cj_Autotune), 1, pFile); 
    fclose(pFile);
  }

}

