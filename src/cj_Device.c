#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "cj_Object.h"
#include "cj_Device.h"

static int gpu_counter = 0;
static int mic_counter = 0;

void cj_Device_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_DEVICE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

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

void cj_Device_report(cj_Device *device) {
  fprintf(stderr, "  Name: %s\n", device->name);
}

cj_Device *cj_Device_new(cj_devType devtype) {
  int i;
  cj_Device *device = (cj_Device*) malloc(sizeof(cj_Device));
  if (!device) cj_Device_error("Device_new", "memory allocation failed.");
  if (devtype == CJ_DEV_CUDA) {
    cudaError_t error;
	  struct cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, gpu_counter);
	  device->name = prop.name;
	  gpu_counter ++;

	  /* Setup device cache */
	  device->cache.line_size = BLOCK_SIZE*BLOCK_SIZE*sizeof(double);
    for (i = 0; i < CACHE_LINE; i++) {
      device->cache.status[i] = CJ_CACHE_CLEAN;
      device->cache.last_use[i] = 0;
    }

    cj_Device_report(device);
  }

  return device;
}
