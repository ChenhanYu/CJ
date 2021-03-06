/*
 *  cj_Device.c
 *  Chenhan D. Yu
 *  Created: Mar 30, 2014
 *
 *  Implement the software cache for the GPU device.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cj.h>

static int gpu_counter = 0;
static int mic_counter = 0;

/**
 *  @brief  Report error messages. 
 *  @param  *func_name :function name pointer
 *  @param  *msg_text :error message pointer
 */
void cj_Device_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_DEVICE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

/**
 *  @brief  Read the target object into device cache.
 *  @param  *device :device structure pointer
 *  @param  line_id :cache line id
 *  @param  *target :target object pointer
 */
void cj_Cache_read_in (cj_Device *device, int line_id, cj_Object *target) {
  cj_Cache *cache = &device->cache;
  uintptr_t ptr_d = cache->dev_ptr[line_id];
  //fprintf(stderr, "inside cache_read_in\n");
  char *ptr_h;
  if (device->devtype == CJ_DEV_CUDA) {    
    if (target->objtype == CJ_MATRIX) {
      //fprintf(stderr, "Cache_read_in (CJ_MATRIX) : \n");
      cj_Matrix *base = target->matrix->base;
      cj_Matrix *matrix = target->matrix;

      ptr_h = base->buff + base->m*base->elelen*matrix->offn + matrix->offm*base->elelen;
      cj_Device_memcpy2d_h2d(ptr_d, BLOCK_SIZE*base->elelen, ptr_h, base->m*base->elelen,
          matrix->m*matrix->elelen, matrix->n, device);
    }
    else {
    }
  }
  cache->hos_ptr[line_id] = ptr_h;
  cache->status[line_id] = CJ_CACHE_CLEAN;
}

/**
 *  @brief  Write the target object from device cache back to main memory.
 *  @param  *device :device structure pointer
 *  @param  line_id :cache line id
 *  @param  *target :target object pointer
 */
void cj_Cache_write_back (cj_Device *device, int line_id, cj_Object *target) {
  cj_Cache *cache = &device->cache;
  uintptr_t ptr_d = cache->dev_ptr[line_id];
  char *ptr_h = cache->hos_ptr[line_id];
   
  if (device->devtype == CJ_DEV_CUDA) {
    if (target->objtype == CJ_MATRIX) {
      cj_Matrix *base = target->matrix->base;
      cj_Matrix *matrix = target->matrix;
      cj_Device_memcpy2d_d2h(ptr_h, base->m*base->elelen, ptr_d, BLOCK_SIZE*base->elelen,
          matrix->m*matrix->elelen, matrix->n, device);
    }
    else {
    }
  }
  cache->status[line_id] = CJ_CACHE_CLEAN;
}

/**
 *  @brief  Write the target object from device cache back to main memory 
 *          asynchronously. This functions can be synchronized by cj_Cache_sync.
 *  @param  *device :device structure pointer
 *  @param  line_id :cache line id
 *  @param  *target :target object pointer
 */
void cj_Cache_async_write_back (cj_Device *device, int line_id, cj_Object *target) {
  cj_Cache *cache = &device->cache;
  uintptr_t ptr_d = cache->dev_ptr[line_id];
  char *ptr_h = cache->hos_ptr[line_id];
   
  if (device->devtype == CJ_DEV_CUDA) {
    if (target->objtype == CJ_MATRIX) {
      cj_Matrix *base = target->matrix->base;
      cj_Matrix *matrix = target->matrix;
      cj_Device_async_memcpy2d_d2h(ptr_h, base->m*base->elelen, ptr_d, BLOCK_SIZE*base->elelen,
          matrix->m*matrix->elelen, matrix->n, device);
    }
  }
  //cache->status[line_id] = CJ_CACHE_CLEAN;
}

/**
 *  @brief  Fetch the target from main memory to device memory. This device cache 
 *          is implemented that can only replace clean cache line. If the cache 
 *          line is dirty, it will search for another one.
 *  @param  *device :device structure pointer
 *  @param  *target :target object pointer
 *  @return cache line id
 */
int cj_Cache_fetch (cj_Device *device, cj_Object *target) {
  cj_Cache *cache = &device->cache;
  int line_id = cache->fifo;

  /* Update fifo pointer */
  cache->fifo ++;
  if (cache->fifo >= CACHE_LINE) cache->fifo = 0;

  if (cache->status[line_id] == CJ_CACHE_DIRTY) {
    line_id = cj_Cache_fetch(device, target);
  }
  else {
    cj_Cache_read_in(device, line_id, target);
  }
  return line_id;
}

/**
 *  @brief  Synchronous barrier implemented by CUDA streams.
 *  @param  *device :device structure pointer
 * */
void cj_Cache_sync (cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaStreamSynchronize(device->stream[0]);
#endif
  }
}

/**
 *  @brief  Synchronous barrier implemented by CUDA streams.
 *  @param  *device :device structure pointer
 * */
void cj_Device_sync (cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaStreamSynchronize(device->stream[1]);
#endif
  }
}

/**
 *  @brief  Allocate device memory for device cache or other usage.
 *  @param  len :memory length in bytes
 *  @param  devtype :can be CUDA or MIC
 *  @return device memory pointer
 * */
uintptr_t cj_Device_malloc (size_t len, cj_devType devtype) {
  char *ptr;
  if (devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    error = cudaMalloc((void**)&ptr, len);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  } 
  return (uintptr_t) ptr;
}

/**
 *  @brief  Free the target device memory.
 *  @param  ptr :device memory pointer represented in unsigned long long.
 *  @param  devtype :can be CUDA or MIC
 * */
void cj_Device_free (uintptr_t ptr, cj_devType devtype) {
  if (devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    error = cudaFree((char *) ptr);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  }
}

/**
 *  @brief  Copy the object from device memory to main memory.
 *  @param  *ptr_h :the corresponding main memory pointer
 *  @param  ptr_d :device memory pointer represented in unsigned long long
 *  @param  len :memory length in bytes
 *  @param  *device :device structure pointer
 * */
void cj_Device_memcpy_d2h (char *ptr_h, uintptr_t ptr_d, size_t len, cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    cudaSetDevice(device->id);
    error = cudaMemcpy(ptr_h, (char *) ptr_d, len, cudaMemcpyDeviceToHost);
    //error = cudaMemcpyAsync(ptr_h, (char *) ptr_d, len, cudaMemcpyDeviceToHost, device->stream[0]);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  }
}

/**
 *  @brief  Copy the object (matrix) from device memory to main memory.
 *  @param  *ptr_h :the corresponding main memory pointer
 *  @param  pitch_h: 
 *  @param  ptr_d :device memory pointer represented in unsigned long long
 *  @param  pitch_d:
 *  @param  mbytes:
 *  @param  n:
 *  @param  *device :device structure pointer
 * */
void cj_Device_memcpy2d_d2h (char *ptr_h, size_t pitch_h, uintptr_t ptr_d, size_t pitch_d, 
    size_t mbytes, size_t n, cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    cudaSetDevice(device->id);
    /*
    fprintf(stderr, "memcpy2d_d2h : pitch_d:%d, pitch_h:%d, mbytes:%d, n:%d\n", 
        (int) pitch_d, (int) pitch_h, (int) mbytes, (int) n);
        */
    error = cudaMemcpy2D(ptr_h, pitch_h, (char *) ptr_d, pitch_d, mbytes, n, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  }
}

/**
 *  @brief  Copy the object (matrix) from main memory to device memory asynchronously.
 *  @param  *ptr_h :the corresponding main memory pointer
 *  @param  pitch_h: 
 *  @param  ptr_d :device memory pointer represented in unsigned long long
 *  @param  pitch_d:
 *  @param  mbytes:
 *  @param  n:
 *  @param  *device :device structure pointer
 * */
void cj_Device_async_memcpy2d_d2h (char *ptr_h, size_t pitch_h, uintptr_t ptr_d, size_t pitch_d, 
    size_t mbytes, size_t n, cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    cudaSetDevice(device->id);
    /*
    fprintf(stderr, "memcpy2d_d2h : pitch_d:%d, pitch_h:%d, mbytes:%d, n:%d\n", 
        (int) pitch_d, (int) pitch_h, (int) mbytes, (int) n);
        */
    error = cudaMemcpy2DAsync(ptr_h, pitch_h, (char *) ptr_d, pitch_d, mbytes, n, cudaMemcpyDeviceToHost, device->stream[0]);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  }
}

/**
 *  @brief  Copy the object from main memory to device memory.
 *  @param  ptr_d :device memory pointer represented in unsigned long long
 *  @param  *ptr_h :the corresponding main memory pointer
 *  @param  len :memory length in bytes
 *  @param  *device :device structure pointer
 * */
void cj_Device_memcpy_h2d (uintptr_t ptr_d, char *ptr_h, size_t len, cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    cudaSetDevice(device->id);
    //error = cudaMemcpy((char *) ptr_d, ptr_h, len, cudaMemcpyHostToDevice);
    error = cudaMemcpyAsync((char *) ptr_d, ptr_h, len, cudaMemcpyHostToDevice, device->stream[0]);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
  }
}

void cj_Device_memcpy2d_h2d (uintptr_t ptr_d, size_t pitch_d, char *ptr_h, size_t pitch_h, 
    size_t mbytes, size_t n, cj_Device *device) {
  if (device->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaError_t error; 
    cudaSetDevice(device->id);
    /*
    fprintf(stderr, "memcpy2d_h2d : pitch_d:%d, pitch_h:%d, mbytes:%d, n:%d\n", 
        (int) pitch_d, (int) pitch_h, (int) mbytes, (int) n);
        */
    //error = cudaMemcpy2D((char *) ptr_d, pitch_d, ptr_h, pitch_h, mbytes, n, cudaMemcpyHostToDevice);
    error = cudaMemcpy2DAsync((char *) ptr_d, pitch_d, ptr_h, pitch_h, mbytes, n, cudaMemcpyHostToDevice, device->stream[0]);
    if (error != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(error));
#endif
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
#ifdef CJ_HAVE_CUDA
    cudaError_t error;
	  struct cudaDeviceProp prop;
    cudaSetDevice(device_id);
    cudaDeviceReset();
    error = cudaGetDeviceProperties(&prop, gpu_counter);
    cudaStreamCreate(&(device->stream[0]));
    cudaStreamCreate(&(device->stream[1]));
    cublasCreate(&(device->handle));
    cublasSetStream(device->handle, device->stream[1]);
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

    fprintf(stderr, "  Name         : %s (%d.%d)\n", prop.name, prop.major, prop.minor);
    fprintf(stderr, "  Device Id    : %d \n", device_id);
    fprintf(stderr, "  Device Memory: %d Mbytes\n", (unsigned int) (prop.totalGlobalMem/1024)/1024);
#endif
  }

  return device;
}
