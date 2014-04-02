/*
typedef enum {CJ_DEV_CPU, CJ_DEV_CUDA, CJ_DEV_MIC} cj_devType;
typedef enum {CJ_CACHE_CLEAN, CJ_CACHE_DIRTY} cj_cacheStatus;

#define CACHE_LINE 32
#define MAX_GPU 4
#define MAX_MIC 4

struct cache_s {
  cj_cacheStatus status[CACHE_LINE];
  char *dev_ptr[CACHE_LINE];
  char *hos_ptr[CACHE_LINE];
  int last_use[CACHE_LINE];
  size_t line_size;
};

struct device_s {
  cj_devType devtype;
  char *name;
  struct cache_s cache;
  int bindid;
  int bandwidth;
};

typedef struct cache_s cj_Cache;
typedef struct device_s cj_Device;
*/


cj_Device *cj_Device_new(cj_devType, int);

void cj_Cache_read_in (cj_Cache*, int, cj_Object*, cj_devType);
void cj_Cache_write_back (cj_Cache*, int, cj_Object*, cj_devType);
int cj_Cache_fetch (cj_Cache*, cj_Object*, cj_devType);

void cj_Device_memcpy_d2h (char*, uintptr_t, size_t, cj_devType);
void cj_Device_memcpy_h2d (uintptr_t, char*, size_t, cj_devType);
void cj_Device_memcpy2d_d2h (char*, size_t, uintptr_t, size_t, size_t, size_t, cj_devType);
void cj_Device_memcpy2d_h2d (uintptr_t, size_t, char*, size_t, size_t, size_t, cj_devType);
