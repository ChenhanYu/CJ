#include <stdarg.h>
#include <pthread.h>

struct worker_s {
  cj_devType devtype;
  int id;
  pthread_t threadid;
  struct cj_s *cj_ptr;
};

struct schedule_s {
  cj_Object *ready_queue;
  struct lock_s run_lock;
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
  cj_Device **device;
  pthread_attr_t worker_attr;
};

typedef struct worker_s cj_Worker;
typedef struct schedule_s cj_Schedule;
typedef struct cj_s cj_t;

void cj_Init(int);
void cj_Term();
void cj_Queue_begin();
void cj_Queue_end();

/* cj_Task function prototypes */
cj_Task *cj_Task_new ();
void cj_Task_dependency_add(cj_Object*, cj_Object*);
void cj_Task_dependencies_update (cj_Object*);
