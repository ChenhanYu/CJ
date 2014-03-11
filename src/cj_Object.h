typedef enum {CJ_DQUEUE, CJ_MATRIX} cj_Type;

struct matrix_s {
  int m;
  int n;
};

struct dqueue_s {
  int size;
  struct object_s *head;
  struct object_s *tail;
};

struct object_s {
  void *key;
  cj_Type eletype;
  union {
    struct dqueue_s *dqueue;
	struct matrix_s *matrix;
  };
  struct object_s *prev;
  struct object_s *next;
};

typedef struct dqueue_s cj_Dqueue;
typedef struct matrix_s cj_Matrix;
typedef struct object_s cj_Object;
