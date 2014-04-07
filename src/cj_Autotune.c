#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include <CJ.h>

static cj_Autotune *autotune;

void cj_Autotune_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_AUTOTUNE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

#ifdef CJ_HAVE_CUDA
void cj_Autotune_cublas () {
  int i, nb, ld;
  float time_ms = 0.0;
  float fone = 1.0;
  double done = 1.0;
  float *A, *B, *C;
  double *dA, *dB, *dC;

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

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    cudaEventRecord(beg, 0);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &fone, A, ld, B, ld, &fone, C, ld);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, beg, end);
    fprintf(stderr, "  cublasSgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->cublas_sgemm[i] = time_ms;
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
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaEventDestroy(beg); cudaEventDestroy(end);
  cublasDestroy(handle);
}
#endif

void cj_Autotune_mkl () {
  int i, nb, ld;
  float time_ms = 0.0;
  float fone = 1.0;
  double done = 1.0;
  clock_t t;

  ld = BLOCK_SIZE*AUTOTUNE_GRID;

  float *A = malloc(ld*ld*sizeof(float));
  float *B = malloc(ld*ld*sizeof(float));
  float *C = malloc(ld*ld*sizeof(float));
  double *dA = malloc(ld*ld*sizeof(double));
  double *dB = malloc(ld*ld*sizeof(double));
  double *dC = malloc(ld*ld*sizeof(double));

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    t = clock();
    sgemm_("N", "N", &nb, &nb, &nb, &fone, A, &ld, B, &ld, &fone, C, &ld);
    time_ms = ((float) (t - clock()))/1000;
    fprintf(stderr, "  Sgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_sgemm[i] = time_ms;
  }

  for (i = 0; i < AUTOTUNE_GRID; i++) {
    nb = (i + 1)*BLOCK_SIZE;
    t = clock();
    dgemm_("N", "N", &nb, &nb, &nb, &done, dA, &ld, dB, &ld, &done, dC, &ld);
    time_ms = ((float) (t - clock()))/1000;
    fprintf(stderr, "  Dgemm(%d, %d, %d) : %f ms\n", nb, nb, nb, time_ms);	
    autotune->mkl_dgemm[i] = time_ms;
  }

  free(A); free(B); free(C);
  free(dA); free(dB); free(dC);
}

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
    cj_Autotune_mkl();
#ifdef CJ_HAVE_CUDA
    cj_Autotune_cublas();
    cj_Autotune_pci();
#endif
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

cj_Autotune *cj_Autotune_get_ptr () {
  return autotune;
}
