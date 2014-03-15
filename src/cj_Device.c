/*

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "cj_Object.h"
#include "cj_Device.h"

uintptr_t cj_Device_malloc(size_t len, cj_devType devtype) {
  char *ptr = NULL;
  if (devtype == CJ_DEV_CUDA) {
    cudaMalloc((void**)&ptr, len);
  }
  
  return (uintptr_t) ptr;
}

void cj_Device_free(uintptr_t ptr, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    cudaFree((char *) ptr);
  }
}

*/
