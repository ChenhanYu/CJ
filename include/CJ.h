/*
 *  CJ.h
 *  Chenhan D. Yu
 *  Created: Mar 30, 2014
 *
 *  The global data structure for cj(General Linear Algebra Computing Environment).
 *
 */

#include <stdarg.h>
#include <pthread.h>

#ifdef CJ_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#define AUTOTUNE_GRID 1
#define BLOCK_SIZE 2048
#define CACHE_LINE 48
#define MAX_WORKER 8
#define MAX_DEV 4
#define MAX_GPU 4
#define MAX_MIC 4
#define GPU_NUM 3

typedef enum {CJ_RED, CJ_GREEN, CJ_BLUE, CJ_YELLOW, CJ_PURPLE, CJ_ORANGE, CJ_BLACK} cj_Color;

typedef enum {CJ_EVENT, CJ_DISTRIBUTION, CJ_DQUEUE, CJ_TASK, CJ_VERTEX, CJ_EDGE, CJ_MATRIX, CJ_CSC, CJ_SPARSE, CJ_CONSTANT} cj_objType;

typedef enum {CJ_DOUBLE, CJ_SINGLE, CJ_COMPLEX, CJ_DCOMPLEX, CJ_INT32, CJ_INT64} cj_eleType;

typedef enum {CJ_W, CJ_R, CJ_RW} cj_rwType;

typedef enum {CJ_TOP, CJ_BOTTOM, CJ_LEFT, CJ_RIGHT} cj_Side;

typedef enum {CJ_TL, CJ_TR, CJ_BL, CJ_BR} cj_Quadrant;

typedef enum {ALLOCATED_ONLY, NOTREADY, QUEUED, RUNNING, DONE, CANCELLED} cj_taskStatus;

typedef enum {PRI_HIGH, PRI_LOW} cj_taskPriority;

typedef enum {TRUE, FALSE} cj_Bool;

typedef enum {WORKER_SLEEPING, WORKER_RUNNING} cj_workerStatus;

//do we need to add CJ_TASK_SYRK?
typedef enum {CJ_TASK_GEMM, CJ_TASK_TRSM, CJ_TASK_SYRK, CJ_TASK_POTRF} cj_taskType;

typedef enum {CJ_DEV_CPU, CJ_DEV_CUDA, CJ_DEV_MIC} cj_devType;

//run_begin, run_end, fetch_begin, fetch_end, prefetch, wait_prefetch, init, terminate
typedef enum {CJ_EVENT_TASK_RUN_BEG, CJ_EVENT_TASK_RUN_END, CJ_EVENT_FETCH_BEG, CJ_EVENT_FETCH_END, 
  CJ_EVENT_PREFETCH, CJ_EVENT_WAIT_PREFETCH, CJ_EVENT_INIT, CJ_EVENT_TERM} cj_eveType;

typedef enum {CJ_CACHE_CLEAN, CJ_CACHE_DIRTY} cj_cacheStatus; 

/**
 *  Thread mutex
 */ 
struct lock_s {
  pthread_mutex_t lock;
};

/**
 *  Distribution is used to descripe the locality of an object in the 
 *  distributed memory environment.
 */
struct distribution_s {
  cj_Bool avail[MAX_DEV + 1];            /// available device and CPU
  struct device_s *device[MAX_DEV + 1];  /// device pointers array
  int line[MAX_DEV + 1];                 /// cache line id array
  struct lock_s lock;                    /// mutex for modifying the distribution
};

/**
 *  Constant is a an object type.   
 */
struct constant_s {
  cj_eleType eletype;                    /// constant type (double, float)
  double dconstant;                      /// double constant value
  float  sconstnat;                      /// float constant value
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
  /* double array of dqueue: read set*/
  struct object_s ***rset;
  /* double array of dqueue: write set*/
  struct object_s ***wset;
  /* distribution */
  struct distribution_s ***dist;
  /* the memory base for the matrix */
  struct matrix_s *base;
  char *buff;
};

struct csc_s {
  cj_eleType eletype;
  size_t elelen;
  int m;
  int n;
  int nnz;
  int *colptr;
  int *rowind;
  void *values;
};

struct sparse_s {
  int m;
  int n;
  int nnz;
  int offm;
  int offn;
  struct object_s *s00;
  struct object_s *s11;
  struct object_s *s20;
  struct object_s *s21;
  struct object_s *s22;
};

struct task_s {
  cj_taskType tasktype;
  struct worker_s *worker;
  char name[64];
  char label[64];
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
  cj_Color color;
};

struct edge_s {
  struct task_s *in;
  struct task_s *out;
  cj_Bool war;
};

struct event_s {
  cj_eveType evetype;
  float beg;
  float end;
  int task_id;
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
    struct csc_s    *csc;
    struct sparse_s *sparse;
    struct event_s  *event;
  };
  struct object_s *prev;
  struct object_s *next;
  cj_rwType rwtype;
};

struct profile_s {
  struct object_s *worker_timeline[MAX_WORKER];
};

/* */
struct autotune_s {
  /* mkl_sgemm... Do we need the mkl_ssyrk routine? */
  float mkl_sgemm[AUTOTUNE_GRID];
  float mkl_dgemm[AUTOTUNE_GRID];
  float cublas_sgemm[AUTOTUNE_GRID];
  float cublas_dgemm[AUTOTUNE_GRID];
  float mkl_ssyrk[AUTOTUNE_GRID];
  float mkl_dsyrk[AUTOTUNE_GRID];
  float cublas_ssyrk[AUTOTUNE_GRID];
  float cublas_dsyrk[AUTOTUNE_GRID];
  float mkl_strsm[AUTOTUNE_GRID];
  float mkl_dtrsm[AUTOTUNE_GRID];
  float cublas_strsm[AUTOTUNE_GRID];
  float cublas_dtrsm[AUTOTUNE_GRID];
  float mkl_spotrf[AUTOTUNE_GRID];
  float mkl_dpotrf[AUTOTUNE_GRID];
  float hybrid_spotrf[AUTOTUNE_GRID];
  float hybrid_dpotrf[AUTOTUNE_GRID];
  float pci_bandwidth;
};

struct worker_s {
  cj_devType devtype;
  int id;
  int device_id;
  pthread_t threadid;
  struct cj_s *cj_ptr;
  struct object_s *write_back;
  struct task_s *current_task;
};

struct schedule_s {
  /* Since every worker has a ready_queue, why don't we put it inside the data structure of worker? */
  struct object_s *ready_queue[MAX_WORKER];
  /* Remaining time for each worker. For performance model */
  float time_remaining[MAX_WORKER];
  /* It seems that we don't need this to record the remaing tasks */
  int ntask;
  struct lock_s run_lock[MAX_WORKER];
  struct lock_s ready_queue_lock[MAX_WORKER];
  struct lock_s ntask_lock;
  struct lock_s pci_lock;
  struct lock_s gpu_lock;
  struct lock_s mic_lock;
};

struct cj_s {
  int nworker;
  int ngpu;
  int nmic;
  struct schedule_s schedule;
  struct worker_s **worker;
  struct device_s *device[MAX_DEV];
  pthread_attr_t worker_attr;
  cj_Bool terminate;
};

struct cache_s {
  /* CJ_CACHE_CLEAN, CJ_CACHE_DIRTY} */
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
typedef struct event_s cj_Event;
typedef struct profile_s cj_Profile;
typedef struct dqueue_s cj_Dqueue;
typedef struct task_s   cj_Task;
typedef struct matrix_s cj_Matrix;
typedef struct csc_s    cj_Csc;
typedef struct sparse_s cj_Sparse;
typedef struct object_s cj_Object;
typedef struct vertex_s cj_Vertex;
typedef struct edge_s   cj_Edge;
typedef struct lock_s   cj_Lock;
