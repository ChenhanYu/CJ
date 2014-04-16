#include <stdarg.h>
#include <pthread.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#define AUTOTUNE_GRID 1
#define BLOCK_SIZE 2048
#define CACHE_LINE 32
#define MAX_WORKER 8
#define MAX_DEV 4
#define MAX_GPU 4
#define MAX_MIC 4

typedef enum {CJ_DISTRIBUTION, CJ_DQUEUE, CJ_TASK, CJ_VERTEX, CJ_EDGE, CJ_MATRIX, CJ_CONSTANT} cj_objType;
typedef enum {CJ_DOUBLE, CJ_SINGLE, CJ_COMPLEX, CJ_DCOMPLEX, CJ_INT32, CJ_INT64} cj_eleType;
typedef enum {CJ_W, CJ_R, CJ_RW} cj_rwType;
typedef enum {CJ_TOP, CJ_BOTTOM, CJ_LEFT, CJ_RIGHT} cj_Side;
typedef enum {CJ_TL, CJ_TR, CJ_BL, CJ_BR} cj_Quadrant;
typedef enum {ALLOCATED_ONLY, NOTREADY, QUEUED, RUNNING, DONE, CANCELLED} cj_taskStatus;
typedef enum {PRI_HIGH, PRI_LOW} cj_taskPriority;
typedef enum {TRUE, FALSE} cj_Bool;
typedef enum {WORKER_SLEEPING, WORKER_RUNNING} cj_workerStatus;
typedef enum {CJ_TASK_GEMM} cj_taskType;
typedef enum {CJ_DEV_CPU, CJ_DEV_CUDA, CJ_DEV_MIC} cj_devType;
typedef enum {CJ_CACHE_CLEAN, CJ_CACHE_DIRTY} cj_cacheStatus;

struct lock_s {
  pthread_mutex_t lock;
};

struct distribution_s {
  cj_Bool avail[MAX_DEV + 1];
  struct device_s *device[MAX_DEV + 1];
  int line[MAX_WORKER + 1];
  struct lock_s lock;
};

struct constant_s {
  cj_eleType eletype;
  double dconstant;
  float  sconstnat;
  /* TODO : decide what other type of constant should look like. */
};

struct matrix_s {
  cj_eleType eletype;
  size_t elelen;
  int m;
  int n;
  int mb;
  int nb;
  int offm;
  int offn;
  /* double array of dqueue */
  struct object_s ***rset;
  /* double array of dqueue */
  struct object_s ***wset;
  /* distribution */
  struct distribution_s ***dist;
  struct matrix_s *base;
  char *buff;
};

struct task_s {
  cj_taskType tasktype;
  struct worker_s *worker;
  char name[64];
  int id;
  struct lock_s tsk_lock;
  float cost;
  cj_taskPriority priority;
  /* Function ptr */
  void (*function) (void*);
  volatile cj_taskStatus status;
  volatile int num_dependencies_remaining;
  /* Dependency */
  struct object_s *in;
  struct object_s *out;
  /* Argument list */
  struct object_s *arg;
/*  
  struct object_s *arg_in;
  struct object_s *arg_out;
*/
  /* Destination */
};

struct dqueue_s {
  struct lock_s lock;
  int size;
  struct object_s *head;
  struct object_s *tail;
};

struct vertex_s {
  struct task_s *task;
};

struct edge_s {
  struct task_s *in;
  struct task_s *out;
};

struct object_s {
  void *key;
  cj_objType objtype;
  union {
    struct dqueue_s *dqueue;
    struct matrix_s *matrix;
    struct task_s   *task;
    struct vertex_s *vertex;
    struct edge_s   *edge;
    struct distribution_s *distribution;
  };
  struct object_s *prev;
  struct object_s *next;
  cj_rwType rwtype;
};


struct autotune_s {
  float mkl_sgemm[AUTOTUNE_GRID];
  float mkl_dgemm[AUTOTUNE_GRID];
  float cublas_sgemm[AUTOTUNE_GRID];
  float cublas_dgemm[AUTOTUNE_GRID];
  float pci_bandwidth;
};

struct worker_s {
  cj_devType devtype;
  int id;
  int device_id;
  pthread_t threadid;
  struct cj_s *cj_ptr;
  struct object_s *write_back;
};

struct schedule_s {
  struct object_s *ready_queue[MAX_WORKER];
  float time_remaining[MAX_WORKER];
  int task_remaining;
  struct lock_s run_lock[MAX_WORKER];
  struct lock_s ready_queue_lock[MAX_WORKER];
  struct lock_s war_lock;
  struct lock_s pci_lock;
  struct lock_s gpu_lock;
  struct lock_s mic_lock;
};

struct cj_s {
  int nworker;
  struct schedule_s schedule;
  struct worker_s **worker;
  int ngpu;
  int nmic;
  struct device_s *device[MAX_DEV];
  pthread_attr_t worker_attr;
};

struct cache_s {
  cj_cacheStatus status[CACHE_LINE];
  struct object_s *obj_ptr[CACHE_LINE];
  uintptr_t dev_ptr[CACHE_LINE];
  char *hos_ptr[CACHE_LINE];
  int last_use[CACHE_LINE];
  int fifo;
  size_t line_size;
};

struct device_s {
  cj_devType devtype;
  int id;
  char *name;
  struct cache_s cache;
  int bindid;
  int bandwidth;
#ifdef CJ_HAVE_CUDA
  cudaStream_t stream[2];
  cublasHandle_t handle;
#endif
};

struct graph_s {
  int nvisit;
  struct object_s *vertex;
  struct object_s *edge;
};

typedef struct graph_s cj_Graph;
typedef struct cache_s cj_Cache;
typedef struct device_s cj_Device;
typedef struct worker_s cj_Worker;
typedef struct schedule_s cj_Schedule;
typedef struct cj_s cj_t;
typedef struct autotune_s cj_Autotune;
typedef struct distribution_s cj_Distribution;
typedef struct dqueue_s cj_Dqueue;
typedef struct task_s   cj_Task;
typedef struct matrix_s cj_Matrix;
typedef struct object_s cj_Object;
typedef struct vertex_s cj_Vertex;
typedef struct edge_s   cj_Edge;
typedef struct lock_s   cj_Lock;
