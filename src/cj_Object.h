#include <pthread.h>

#define BLOCK_SIZE 64

typedef enum {CJ_DISTRIBUTION, CJ_DQUEUE, CJ_TASK, CJ_VERTEX, CJ_EDGE, CJ_MATRIX, CJ_CONSTANT} cj_objType;
typedef enum {CJ_DOUBLE, CJ_SINGLE, CJ_COMPLEX, CJ_DCOMPLEX, CJ_INT32, CJ_INT64} cj_eleType;
typedef enum {CJ_TOP, CJ_BOTTOM, CJ_LEFT, CJ_RIGHT} cj_Side;
typedef enum {CJ_TL, CJ_TR, CJ_BL, CJ_BR} cj_Quadrant;
typedef enum {ALLOCATED_ONLY, NOTREADY, QUEUED, RUNNING, DONE, CANCELLED} cj_taskStatus;
typedef enum {PRI_HIGH, PRI_LOW} cj_taskPriority;
typedef enum {TRUE, FALSE} cj_Bool;
typedef enum {WORKER_SLEEPING, WORKER_RUNNING} cj_workerStatus;

//typedef enum {CJ_GEMM, CJ_TRSM, CJ_POTRF} cj_tskType;

struct distribution_s {
  int device_id;
  int cache_id;
};

struct lock_s {
  pthread_mutex_t lock;
};

struct constant_s {
  cj_eleType eletype;
  double dconstant;
  float  sconstnat;
  /* TODO : decide what other type of constant should look like. */
};

struct matrix_s {
  cj_eleType eletype;
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
  struct object_s ***dist;
  struct matrix_s *base;
};

struct task_s {
  char name[64];
  int id;
  struct lock_s tsk_lock;
  cj_taskPriority priority;
  /* Function ptr */
  void (*function) (void*);
  volatile cj_taskStatus status;
  volatile int num_dependencies_remaining;
  /* Dependency */
  struct object_s *in;
  struct object_s *out;
  /* Argument list */
  /* Destination */
};

struct dqueue_s {
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
};

typedef struct distribution_s cj_Distribution;
typedef struct dqueue_s cj_Dqueue;
typedef struct task_s   cj_Task;
typedef struct matrix_s cj_Matrix;
typedef struct object_s cj_Object;
typedef struct vertex_s cj_Vertex;
typedef struct edge_s   cj_Edge;
typedef struct lock_s   cj_Lock;

/* cj_Object function prototypes */
cj_Object *cj_Object_new (cj_objType);
cj_Object *cj_Object_append (cj_objType, void*);

/* cj_Lock function prototypes */
void cj_Lock_new (cj_Lock*);
void cj_Lock_delete (cj_Lock*);
void cj_Lock_acquire (cj_Lock*);
void cj_Lock_release (cj_Lock*);

/* cj_Matrix function prototypes */
void cj_Matrix_set (cj_Object*, int, int);
void cj_Matrix_part_2x1 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_1x2 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_2x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, int, cj_Quadrant);
void cj_Matrix_repart_2x1_to_3x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_repart_1x2_to_1x3 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_cont_with_3x1_to_2x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);
void cj_Matrix_cont_with_1x3_to_1x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);

/* cj_Dqueue function prototypes */
int cj_Dqueue_get_size (cj_Object*);
void cj_Dqueue_push_head (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_head (cj_Object*);
void cj_Dqueue_push_tail (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_tail (cj_Object*);
void cj_Dqueue_clear(cj_Object*);

/* cj_Task function prototypes */
void cj_Task_dependency_add(cj_Object*, cj_Object*);
void cj_Task_dependencies_update (cj_Object*);

/* cj_Vertex function prototypes */
void cj_Vertex_set (cj_Object*, cj_Object*);

/* cj_Edge function prototypes */
void cj_Edge_set (cj_Object*, cj_Object*, cj_Object*);
