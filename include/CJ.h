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

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define NONE "\033[m"
#define RED "\033[0;32;31m"
#define LIGHT_RED "\033[1;31m"
#define GREEN "\033[0;32;32m"
#define LIGHT_GREEN "\033[1;32m"
#define BLUE "\033[0;32;34m"
#define LIGHT_BLUE "\033[1;34m"
#define DARY_GRAY "\033[1;30m"
#define CYAN "\033[0;36m"
#define LIGHT_CYAN "\033[1;36m"
#define PURPLE "\033[0;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN "\033[0;33m"
#define YELLOW "\033[1;33m"
#define LIGHT_GRAY "\033[0;37m"
#define WHITE "\033[1;37m"

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


/* cj API function prototypes */
void cj_Init (int);
void cj_Term ();
void cj_Queue_begin ();
void cj_Queue_end ();

/* cj_Distribution function prototypes */
cj_Distribution *cj_Distribution_new();

/* cj_Task function prototypes */
cj_Task *cj_Task_new ();
void cj_Task_set (cj_Task*, cj_taskType, void (*function)(void*));
void cj_Task_dependency_analysis (cj_Object*);
void cj_Task_dependency_add (cj_Object*, cj_Object*);
void cj_Task_dependencies_update (cj_Object*);

/* cj_Worker function prototypes */
float cj_Worker_estimate_cost (cj_Task*, cj_Worker*);
void cj_Worker_wait_prefetch (cj_Worker*, int, int);

void cj_Autotune_init ();
cj_Autotune *cj_Autotune_get_ptr ();


void cj_Gemm_nn_task_function (void*);
void cj_Gemm_nn (cj_Object*, cj_Object*, cj_Object*);
extern void sgemm_ (char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);
extern void dgemm_ (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void cj_Syrk_ln_task_function (void*);
void cj_Syrk_ln_task (cj_Object*, cj_Object*, cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var1 (cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var2 (cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var5 (cj_Object*, cj_Object*);
void cj_Syrk_ln (cj_Object*, cj_Object*);
extern void ssyrk_ (char*, char*, int*, int*, float*, float*, int*, float*, float*, int*);
extern void dsyrk_ (char*, char*, int*, int*, double*, double*, int*, double*, double*, int*);


void cj_Trsm_rlt (cj_Object*, cj_Object*);
void cj_Trsm_rlt_task_function (void*);
extern void strsm_ (char*, char*, char*, char*, int*, int*, float*, float*, int*, float*, int*);
extern void dtrsm_ (char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);


void cj_Cache_read_in (cj_Device*, int, cj_Object*);
void cj_Cache_write_back (cj_Device*, int, cj_Object*);
void cj_Cache_async_write_back (cj_Device*, int, cj_Object*);
int  cj_Cache_fetch (cj_Device*, cj_Object*);
void cj_Cache_sync (cj_Device*);
void cj_Device_sync (cj_Device*);

cj_Device *cj_Device_new (cj_devType, int);
void cj_Device_bind (cj_Worker*, cj_Device*);

/* memcpy from device to host */
void cj_Device_memcpy_d2h (char*, uintptr_t, size_t, cj_Device*);
/* memcpy from host to devide */
void cj_Device_memcpy_h2d (uintptr_t, char*, size_t, cj_Device*);
void cj_Device_memcpy2d_d2h (char*, size_t, uintptr_t, size_t, size_t, size_t, cj_Device*);
void cj_Device_async_memcpy2d_d2h (char*, size_t, uintptr_t, size_t, size_t, size_t, cj_Device*);
void cj_Device_memcpy2d_h2d (uintptr_t, size_t, char*, size_t, size_t, size_t, cj_Device*);


void cj_Graph_init ();
void cj_Graph_vertex_add (cj_Object*);
cj_Object *cj_Graph_vertex_get ();
void cj_Graph_edge_add (cj_Object*);
cj_Object *cj_Graph_edge_get ();
void cj_Graph_output_dot ();

/* cj_Vertex function prototypes */
void cj_Vertex_set (cj_Object*, cj_Object*);
cj_Vertex *cj_Vertex_new ();

/* cj_Edge function prototypes */
void cj_Edge_set (cj_Object*, cj_Object*, cj_Object*, cj_Bool);
cj_Edge *cj_Edge_new ();

void cj_Chol_l_task_function (void*);
void cj_Chol_l (cj_Object*);
#ifdef CJ_HAVE_CUDA
void hybrid_dpotrf (cublasHandle_t*, int, double*, int, int*);
#endif
extern void spotrf_ (char*, int*, float*, int*, int*);
extern void dpotrf_ (char*, int*, double*, int*, int*);


/* cj_Object function prototypes */
cj_Object *cj_Object_new (cj_objType);
cj_Object *cj_Object_append (cj_objType, void*);
void cj_Object_acquire (cj_Object*);

/* cj_Matrix function prototypes */
void cj_Matrix_duplicate (cj_Object*, cj_Object*);
void cj_Matrix_set (cj_Object*, int, int);
void cj_Matrix_set_identity (cj_Object*);
void cj_Matrix_set_lowertril_one (cj_Object*);
void cj_Matrix_set_special_chol (cj_Object*);

void cj_Matrix_print (cj_Object*);
void cj_Matrix_distribution_print (cj_Object*);
void cj_Matrix_part_2x1 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_1x2 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_2x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, int, cj_Quadrant);
void cj_Matrix_repart_2x1_to_3x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_repart_1x2_to_1x3 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_repart_2x2_to_3x3 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*,
                                                          cj_Object*, cj_Object*, cj_Object*,
                                  cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*,
                                                          int,        int,        cj_Quadrant);
void cj_Matrix_cont_with_3x1_to_2x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);
void cj_Matrix_cont_with_1x3_to_1x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);
void cj_Matrix_cont_with_3x3_to_2x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*,
                                                             cj_Object*, cj_Object*, cj_Object*,
                                     cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*,
                                                                                     cj_Quadrant);

#define cj_Matrix_get_distribution(a) a->base->dist[a->offm/BLOCK_SIZE][a->offn/BLOCKSIZE]

/* cj_Dqueue function prototypes */
int cj_Dqueue_get_size (cj_Object*);
void cj_Dqueue_push_head (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_head (cj_Object*);
void cj_Dqueue_push_tail (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_tail (cj_Object*);
void cj_Dqueue_clear (cj_Object*);

cj_Event *cj_Event_new ();
void cj_Profile_worker_record (cj_Worker*, cj_eveType);
void cj_Profile_init ();
void cj_Profile_output_timeline ();

cj_Csc *cj_Csc_new ();
cj_Sparse *cj_Sparse_new ();
