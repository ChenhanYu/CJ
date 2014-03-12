#define BLOCK_SIZE 64

typedef enum {CJ_DQUEUE, CJ_TASK, CJ_MATRIX} cj_objType;
typedef enum {CJ_TOP, CJ_BOTTOM} cj_Side;
//typedef enum {CJ_GEMM, CJ_TRSM, CJ_POTRF} cj_tskType;

struct matrix_s {
  int m;
  int n;
  int mb;
  int nb;
  int offm;
  int offn;
  /* double array of dqueue */
  struct object_s **rset;
  /* double array of dqueue */
  struct object_s **wset;
  struct matrix_s *base;
};

struct task_s {
  char *name;
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

struct object_s {
  void *key;
  cj_objType objtype;
  union {
    struct dqueue_s *dqueue;
	struct matrix_s *matrix;
	struct task_s   *task;
  };
  struct object_s *prev;
  struct object_s *next;
};

typedef struct dqueue_s cj_Dqueue;
typedef struct task_s   cj_Task;
typedef struct matrix_s cj_Matrix;
typedef struct object_s cj_Object;


cj_Object *cj_Object_new (cj_objType);
void cj_Matrix_part_2x1 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_repart_2x1_to_3x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_cont_3x1_to_2x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);
