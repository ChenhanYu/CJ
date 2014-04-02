#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime_api.h>

/*
#include "cj_Object.h"
*/
#include <CJ.h>
#include "cj_Device.h"


static int gpu_counter = 0;
static int mic_counter = 0;

void cj_Device_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_DEVICE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

//void cj_Cache_read_in (cj_Cache *cache, char *ptr_h, int line_id, size_t len, cj_devType devtype) {
void cj_Cache_read_in (cj_Cache *cache, int line_id, cj_Object *target, cj_devType devtype) {
  uintptr_t ptr_d = cache->dev_ptr[line_id];
  char *ptr_h;
  if (devtype == CJ_DEV_CUDA) {
    if (target->objtype == CJ_MATRIX) {
      fprintf(stderr, "Cache_read_in (CJ_MATRIX) : \n");
      cj_Matrix *base = target->matrix->base;
      cj_Matrix *matrix = target->matrix;

      /* TODO : Check if the offset is correct. */
      ptr_h = base->buff + base->m*base->elelen*matrix->offn + matrix->offm*base->elelen;

      cj_Device_memcpy2d_h2d(ptr_d, BLOCK_SIZE*base->elelen, ptr_h, base->m*base->elelen,
          matrix->m, matrix->n*matrix->elelen, devtype);
    }
    else {
      /*
      if (len < cache->line_size) cj_Device_memcpy_h2d(ptr_d, ptr_h, len, devtype);
      else cj_Device_memcpy_h2d(ptr_d, ptr_h, cache->line_size, devtype);
      */
    }
  }
  cache->hos_ptr[line_id] = ptr_h;
  /* Update the distribution. */

  cache->status[line_id] = CJ_CACHE_CLEAN;
}

//void cj_Cache_write_back (cj_Cache *cache, int line_id, size_t len, cj_devType devtype) {
void cj_Cache_write_back (cj_Cache *cache, int line_id, cj_Object *target, cj_devType devtype) {
  uintptr_t ptr_d = cache->dev_ptr[line_id];
  char *ptr_h = cache->hos_ptr[line_id];

  if (devtype == CJ_DEV_CUDA) {
    if (target->objtype == CJ_MATRIX) {
      cj_Matrix *base = target->matrix->base;
      cj_Matrix *matrix = target->matrix;
      cj_Device_memcpy2d_d2h(ptr_h, base->m*base->elelen, ptr_d, BLOCK_SIZE*base->elelen,
          matrix->m, matrix->n*matrix->elelen, devtype);
    }
    else {
      /*
      if (len < cache->line_size) cj_Device_memcpy_d2h(ptr_h, ptr_d, len, devtype);
      ele cj_Device_memcpy_d2h(ptr_h, ptr_d, cache->line_size, devtype);
      */
    }
  }
  /* Update the distribution. */

  cache->status[line_id] = CJ_CACHE_CLEAN;
}

//int cj_Cache_fetch (cj_Cache *cache, char *ptr_h, size_t len, cj_devType devtype) {
int cj_Cache_fetch (cj_Cache *cache, cj_Object *target, cj_devType devtype) {
  int line_id = cache->fifo;

  /* Update fifo pointer */
  cache->fifo ++;
  if (cache->fifo >= CACHE_LINE) cache->fifo = 0;

  if (devtype == CJ_DEV_CUDA) {
    if (cache->status[line_id] == CJ_CACHE_DIRTY) {
      //cj_Cache_write_back(cache, line_id, len, devtype); 
      cj_Cache_write_back(cache, line_id, target, devtype);
      /* TODO : update distribution here. */ 
    }
    //cj_Cache_read_in(cache, ptr_h, line_id, len, devtype);
    cj_Cache_read_in(cache, line_id, target, devtype);
  }
  return line_id;
}

uintptr_t cj_Device_malloc (size_t len, cj_devType devtype) {
  char *ptr;
  if (devtype == CJ_DEV_CUDA) {
    cudaMalloc((void**)&ptr, len);
  }
  
  return (uintptr_t) ptr;
}

void cj_Device_free (uintptr_t ptr, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    cudaFree((char *) ptr);
  }
}

void cj_Device_memcpy_d2h (char *ptr_h, uintptr_t ptr_d, size_t len, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    cudaMemcpy(ptr_h, (char *) ptr_d, len, cudaMemcpyDeviceToHost);
  }
}

void cj_Device_memcpy2d_d2h (char *ptr_h, size_t pitch_h, uintptr_t ptr_d, size_t pitch_d, 
    size_t m, size_t nbytes, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    fprintf(stderr, "memcpy2d_d2h : %d, %d\n", (int) m, (int) nbytes);
    cudaMemcpy2D(ptr_h, pitch_h, (char *) ptr_d, pitch_d, m, nbytes, cudaMemcpyDeviceToHost);
  }
}

void cj_Device_memcpy_h2d (uintptr_t ptr_d, char *ptr_h, size_t len, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    cudaMemcpy((char *) ptr_d, ptr_h, len, cudaMemcpyHostToDevice);
  }
}

void cj_Device_memcpy2d_h2d (uintptr_t ptr_d, size_t pitch_d, char *ptr_h, size_t pitch_h, 
    size_t m, size_t nbytes, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
    fprintf(stderr, "memcpy2d_h2d : %d, %d\n", (int) m, (int) nbytes);
    cudaMemcpy2D((char *) ptr_d, pitch_d, ptr_h, pitch_h, m, nbytes, cudaMemcpyHostToDevice);
  }
}

void cj_Device_report(cj_Device *device) {
}

void cj_Device_bind(cj_Worker *worker, cj_Device *device) {
  worker->devtype = device->devtype;
  worker->device_id = device->id;
  device->bindid = worker->id;
}

cj_Device *cj_Device_new(cj_devType devtype, int device_id) {
  int i;

  cj_Device *device = (cj_Device*) malloc(sizeof(cj_Device));
  if (!device) cj_Device_error("Device_new", "memory allocation failed.");

  device->devtype = devtype;
  device->id = device_id;

  if (devtype == CJ_DEV_CUDA) {
    cudaError_t error;
	  struct cudaDeviceProp prop;
    cudaSetDevice(device_id);
    error = cudaGetDeviceProperties(&prop, gpu_counter);
	  device->name = prop.name;
	  gpu_counter ++;

	  /* Setup device cache */
	  device->cache.line_size = BLOCK_SIZE*BLOCK_SIZE*sizeof(double);
    for (i = 0; i < CACHE_LINE; i++) {
      device->cache.status[i] = CJ_CACHE_CLEAN;
      device->cache.last_use[i] = 0;
      device->cache.fifo = 0;
      device->cache.dev_ptr[i] = cj_Device_malloc(device->cache.line_size, CJ_DEV_CUDA);
      device->cache.hos_ptr[i] = NULL;
    }

    //cj_Device_report(device);
    fprintf(stderr, "  Name         : %s (%d.%d)\n", prop.name, prop.major, prop.minor);
    //fprintf(stderr, "  Device Memory: %d Mbytes\n", (int) ((prop.totalGlobalMem/1024)/1024));
    fprintf(stderr, "  Device Memory: %d Mbytes\n", (unsigned int) (prop.totalGlobalMem/1024)/1024);
  }

  return device;
}
