#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <CJ.h>
#include "cj_Macro.h"
#include "cj_Device.h"
#include "cj_Graph.h"
#include "cj_Object.h"
#include "cj_Blas.h"
#include "cj_Autotune.h"
#include "cj.h"

static cj_t cj;
static int taskid = 0;
static cj_Bool cj_queue_enable = FALSE;

void cj_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Lock_new (cj_Lock *lock) {
  int ret = pthread_mutex_init(&(lock->lock), NULL);
  if (ret) cj_error("Lock_new", "Could not initial locks properly.");
}

void cj_Lock_delete (cj_Lock *lock) {
  int ret = pthread_mutex_destroy(&(lock->lock));
  if (ret) cj_error("Lock_delete", "Could not destroy locks properly.");
}

void cj_Lock_acquire (cj_Lock *lock) {
  int ret = pthread_mutex_lock(&(lock->lock));
  if (ret) cj_error("Lock_acquire", "Could not acquire locks properly.");
}

void cj_Lock_release (cj_Lock *lock) {
  int ret = pthread_mutex_unlock(&(lock->lock));
  if (ret) cj_error("Lock_release", "Could not release locks properly.");
}

cj_Task *cj_Task_new () {
  cj_Task *task = (cj_Task *) malloc(sizeof(cj_Task));
  if (!task) cj_error("Task_new", "memory allocation failed.");

  /* taskid is a monoton increasing global variable. */
  task->id       = taskid;
  taskid ++;

  task->status   = ALLOCATED_ONLY;
  task->function = NULL;
  task->num_dependencies_remaining = 0;
  cj_Lock_new(&task->tsk_lock);
  task->in       = cj_Object_new(CJ_DQUEUE);
  task->out      = cj_Object_new(CJ_DQUEUE);
  task->arg      = cj_Object_new(CJ_DQUEUE);

  if (!task->in || !task->out || !task->arg) {
    cj_error("Task_new", "memory allocation failed.");
  }

  task->status   = NOTREADY;
  return task;
}

void cj_Task_set (cj_Task *task, void (*function)(void*)) {
  task->function = function;
}

void cj_Task_dependency_analysis (cj_Object *task) {
  if (task->objtype != CJ_TASK) {
    cj_error("Task_dependency_analysis", "The object is not a task.");
  }

  /* Insert the task into the global dependency graph. */
  cj_Object *vertex = cj_Object_new(CJ_VERTEX);
  cj_Vertex_set(vertex, task);
  cj_Graph_vertex_add(vertex);

  /* Update read (input) dependencies. */
  cj_Object *now = task->task->arg->dqueue->head;
  cj_Object *set_r, *set_w;

  while (now) {
    if (now->objtype == CJ_MATRIX) {
      cj_Matrix *matrix = now->matrix;
      set_r = matrix->base->rset[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
      set_w = matrix->base->wset[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

      if (now->rwtype == CJ_R || now->rwtype == CJ_RW) {
        cj_Dqueue_push_tail(set_r, cj_Object_append(CJ_TASK, (void *) task->task));
        if (cj_Dqueue_get_size(set_w) > 0) {
          cj_Object *now_w = set_w->dqueue->head;
          while (now_w) {
            if (now_w->task->id != task->task->id) {
              cj_Object *edge = cj_Object_new(CJ_EDGE);
              cj_Edge_set(edge, now_w, task);
              cj_Graph_edge_add(edge);
              cj_Task_dependency_add(now_w, task);
              fprintf(stderr, "          %d->%d.\n", now_w->task->id, task->task->id);
            }
            now_w = now_w->next;
          }
        }
      }
      if (now->rwtype == CJ_W || now->rwtype == CJ_RW) {
        cj_Object *now_r = set_r->dqueue->head; 
        while (now_r) {
          if (now_r->task->id != task->task->id) {
            cj_Object *edge = cj_Object_new(CJ_EDGE);
            cj_Edge_set(edge, now_r, task);
            cj_Graph_edge_add(edge);
            cj_Task_dependency_add(now_r, task);
            fprintf(stderr, "          %d->%d. Anti-dependency.\n", now_r->task->id, task->task->id);
          }
          now_r = now_r->next;
        }
        cj_Dqueue_clear(set_w);
        cj_Dqueue_push_tail(set_w, cj_Object_append(CJ_TASK, (void *) task->task));
        cj_Dqueue_clear(set_r);
      }
    }
    now = now->next;
  }
}

void cj_Task_dependency_add (cj_Object *out, cj_Object *in) {
  if (out->objtype != CJ_TASK || in->objtype != CJ_TASK) {
    cj_error("Task_dependency_add", "The object is not a task.");
  }
  cj_Task *task_out = out->task;
  cj_Task *task_in  = in->task;

  /* Critical section : access shared memory */
  cj_Lock_acquire(&task_out->tsk_lock);
  {
    cj_Dqueue_push_tail(task_out->out, cj_Object_append(CJ_TASK, (void *) task_in));
  }
  cj_Lock_release(&task_out->tsk_lock);

  /* Critical section : access shared memory */
  cj_Lock_acquire(&task_in->tsk_lock);
  {
    cj_Dqueue_push_tail(task_in->in, cj_Object_append(CJ_TASK, (void *) task_out));
    if (task_out->status != DONE) task_in->num_dependencies_remaining ++;
  }
  cj_Lock_release(&task_in->tsk_lock);
}

void cj_Task_enqueue(cj_Object *target) {
  if (target->objtype != CJ_TASK) {
    cj_error("Task_enqueue", "The object is not a task.");
  }

  int i, dest = 1;
  float cost, min_time = -1.0;
  cj_Schedule *schedule = &cj.schedule;

  for (i = 1; i < cj.nworker; i++) {
    cost = cj_Worker_estimate_cost(target->task, cj.worker[i]);
    //fprintf(stderr, "  worker #%d, cost = %f\n", i, cost);
    if (min_time == -1.0 || schedule->time_remaining[i] + cost < min_time) {
      min_time = schedule->time_remaining[i] + cost;
      target->task->cost = cost;
      dest = i;
    }
    //fprintf(stderr, "  worker #%d, min_time = %f\n", i, min_time);
  }

  /* Critical section : push the task to worker[dest]'s ready_queue. */
  cj_Lock_acquire(&schedule->ready_queue_lock[dest]);
  {
    target->task->status = QUEUED;
    schedule->time_remaining[dest] += target->task->cost;
    cj_Dqueue_push_tail(schedule->ready_queue[dest], target);
    fprintf(stderr, "  Enqueue task<%d> to worker[%d]\n", target->task->id, dest);
  }
  cj_Lock_release(&schedule->ready_queue_lock[dest]);
}

void cj_Task_dependencies_update (cj_Object *target) {
  if (target->objtype != CJ_TASK) cj_error("Task_dependencies_update", "The object is not a task.");
  cj_Task *task = target->task;
  cj_Object *now = task->out->dqueue->head;

  while (now) {
    cj_Task *child = now->task;

    /* Critical section : */
    /* TODO : Check if this lock will lead to dead lock. */
    cj_Lock_acquire(&child->tsk_lock);
    {
      child->num_dependencies_remaining --;
      if (child->num_dependencies_remaining < 0) {
        cj_error("Task_dependencies_update", "Remaining dependencies can't be negative.");
      }
      if (child->num_dependencies_remaining == 0 && child->status == NOTREADY) {
        cj_Task_enqueue(cj_Object_append(CJ_TASK, (void*) child));
      }
    }
    cj_Lock_release(&child->tsk_lock);
    now = now->next;
  }
  task->status = DONE;
}

cj_Object *cj_Worker_wait_dqueue (cj_Worker *worker) {
  cj_Schedule *schedule = &cj.schedule;
  cj_Object *task = cj_Dqueue_pop_head(schedule->ready_queue[worker->id]);
  /*
     if (task) fprintf(stderr, YELLOW "  Worker_wait_dqueue (%d, %s): \n" NONE, task->task->id, task->task->name);
     else {
     fprintf(stderr, YELLOW "  Worker_wait_dqueue : ready_queue is empty.\n" NONE);
     sleep(1);
     }
     */
  return task;
}

/* This routine is going to gather all required memory. It will lock the
 * distribution all required object and will release them after the execution
 * is finished. */
void cj_Worker_fetch (cj_Task *task, cj_Worker *worker) {
  cj_Object *arg_I = task->arg->dqueue->head;

  while (arg_I) {
    if (arg_I->objtype == CJ_MATRIX) {
      cj_Matrix *matrix = arg_I->matrix;
      cj_Matrix *base   = matrix->base;
      cj_Object *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
      cj_Object *dist_I = dist->dqueue->head;
      cj_Distribution *distribution = NULL;
      cj_Object *dist_H = NULL;

      while (dist_I) {
        if (dist_I->distribution->device_id == worker->device_id) {
          distribution = dist_I->distribution;
          break;
        }
        if (dist_I->distribution->device_id == -1) dist_H = dist_I;
        dist_I = dist_I->next;
      }

      if (!distribution) {
        distribution = dist->dqueue->head->distribution;
        int device_id = distribution->device_id;
        
        if (!dist_H) {
          /* Sync Write Back */
          cj_Cache_write_back(cj.device[device_id], distribution->cache_id, arg_I);
          dist_H = cj_Object_new(CJ_DISTRIBUTION);

          /* Critical Section */
          cj_Lock_acquire(&dist->dqueue->lock);
          {
            cj_Dqueue_push_head(dist, dist_H);
          }
          cj_Lock_release(&dist->dqueue->lock);
        }
        if (worker->devtype != CJ_DEV_CPU) {
          cj_Object *dist_D = cj_Object_new(CJ_DISTRIBUTION);
          fprintf(stderr, "Cache fetch. device = %d\n", cj.device[worker->device_id]->id);
          int cache_id = cj_Cache_fetch(cj.device[worker->device_id], arg_I);
          cj_Distribution_set(dist_D, cj.device[worker->device_id], worker->device_id, cache_id);

          /* Critical Section */
          cj_Lock_acquire(&dist->dqueue->lock);
          {
            cj_Dqueue_push_head(dist, dist_D);
          }
          cj_Lock_release(&dist->dqueue->lock);
        }
      }
    }
    arg_I = arg_I->next;
  }
}

int cj_Worker_prefetch_d2h (cj_Worker *worker) {
  cj_Object *now = worker->write_back->dqueue->head;
  if (!now) return 0;

  if (now->objtype == CJ_MATRIX) {
    cj_Matrix *matrix = now->matrix;
    cj_Matrix *base   = matrix->base;
    cj_Object *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
    cj_Object *dist_I = dist->dqueue->head;

    cj_Distribution *distribution = NULL;
    cj_Object *dist_H = NULL;

    while (dist_I) {
      if (dist_I->distribution->device_id == worker->device_id) {
        distribution = dist_I->distribution;
      }
      if (dist_I->distribution->device_id == -1) dist_H = dist_I;
      dist_I = dist_I->next;
    }
    if (distribution && !dist_H) {
      cj_Cache_async_write_back(cj.device[distribution->device_id], distribution->cache_id, now);
      dist_H = cj_Object_new(CJ_DISTRIBUTION);

      /* Critial Section */
      cj_Lock_acquire(&dist->dqueue->lock);
      {
        cj_Dqueue_push_head(dist, dist_H);
      }
      cj_Lock_release(&dist->dqueue->lock);
    }
  }
  return 1;
}

int cj_Worker_prefetch_h2d (cj_Worker *worker) {
  cj_Object *arg_I = NULL, *dist_I = NULL;
  cj_Schedule *schedule = &cj.schedule;
  cj_Object *task = schedule->ready_queue[worker->id]->dqueue->head;

  if (worker->device_id == -1) return 0;
  if (task) arg_I = task->task->arg->dqueue->head;
  else return 0;

  while (arg_I) {
    if (arg_I->objtype == CJ_MATRIX) {
      cj_Matrix *matrix = arg_I->matrix;
      cj_Matrix *base   = matrix->base;
      cj_Object *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
      dist_I = dist->dqueue->head;
      cj_Distribution *distribution = NULL;
      cj_Object *dist_H = NULL;

      while (dist_I) {
        if (dist_I->distribution->device_id == worker->device_id) {
          distribution = dist_I->distribution;
          break;
        }
        if (dist_I->distribution->device_id == -1) dist_H = dist_I;
        dist_I = dist_I->next;
      }
      if (!distribution) {
        distribution = dist->dqueue->head->distribution;
        int device_id = distribution->device_id;
        
        if (dist_H && worker->devtype != CJ_DEV_CPU) {
          fprintf(stderr, GREEN "Cache prefetch. device = %d\n" NONE, cj.device[worker->device_id]->id);

          /* TODO : Check the object distribution being written back. */
          int cache_id = cj_Cache_fetch(cj.device[worker->device_id], arg_I);
          cj_Object *dist_D = cj_Object_new(CJ_DISTRIBUTION);
          cj_Distribution_set(dist_D, cj.device[worker->device_id], worker->device_id, cache_id);

          /* Critial Section */
          cj_Lock_acquire(&dist->dqueue->lock);
          {
            cj_Dqueue_push_head(dist, dist_D);
          }
          cj_Lock_release(&dist->dqueue->lock);
        }
      }
    }
    arg_I = arg_I->next;
  }
  return 1;
}

void cj_Worker_wait_prefetch (cj_Worker *worker, int h2d, int d2h) {
  if (h2d) {
    cj_Cache_sync(cj.device[worker->device_id]);
    
  }
  if (d2h) {
    cj_Cache_sync(cj.device[worker->device_id]);

  }
}

void cj_Worker_wait_execute (cj_Worker *worker) {
  if (worker->device_id != -1) {
    cj_Device_sync(cj.device[worker->device_id]);
  }
}

cj_Worker *cj_Worker_new (cj_devType devtype, int id) {
  fprintf(stderr, RED "  Worker_new (%d): \n" NONE, id); 
  fprintf(stderr, "  {\n"); 

  cj_Worker *worker = (cj_Worker *) malloc(sizeof(cj_Worker));
  if (!worker) {
    cj_error("Worker_new", "memory allocation failed.");
  }

  worker->devtype    = devtype;
  worker->id         = id;
  worker->device_id  = -1;
  worker->cj_ptr     = &cj;
  worker->write_back = cj_Object_new(CJ_DQUEUE); 

  fprintf(stderr, "  }\n"); 
  return worker;
}

float cj_Worker_estimate_cost (cj_Task *task, cj_Worker *worker) {
  cj_Autotune *model = cj_Autotune_get_ptr(); 
  float comp_cost = 0.0, comm_cost = 0.0, cost = 0.0;
  cj_Object *now, *dist_now;

  if (task->function == &cj_Gemm_nn_task_function) {
    if (worker->devtype == CJ_DEV_CUDA) {
      //comp_cost = model->cublas_sgemm[0];
      comp_cost = model->cublas_dgemm[0];
      /* Scan through all arguments. */
      now = task->arg->dqueue->head;
      while (now) {
        if (now->objtype == CJ_MATRIX) {
          comm_cost = model->pci_bandwidth;
          cj_Matrix *matrix = now->matrix;
          cj_Matrix *base = matrix->base;
          cj_Object *dist = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
          dist_now = dist->dqueue->head;
          while (dist_now) {
            if (dist_now->distribution->device_id == worker->device_id) {
              comm_cost = 0.0;
              break;
            }
            dist_now = dist_now->next;
          }
          cost += comm_cost;
        }
        now = now->next;
      }
    }
    else if (worker->devtype == CJ_DEV_CPU) {
      //comp_cost = model->mkl_sgemm[0];
      comp_cost = model->mkl_dgemm[0];
      now = task->arg->dqueue->head;
      while (now) {
        if (now->objtype == CJ_MATRIX) {
          comm_cost = model->pci_bandwidth;
          cj_Matrix *matrix = now->matrix;
          cj_Matrix *base = matrix->base;
          cj_Object *dist = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
          dist_now = dist->dqueue->head;
          while (dist_now) {
            if (dist_now->distribution->device_id == worker->device_id) {
              comm_cost = 0.0;
              break;
            }
            dist_now = dist_now->next;
          }
          cost += comm_cost;
        }
        now = now->next;
      }
    }
  }

  cost += comp_cost;

  return cost;
}

int cj_Worker_execute (cj_Task *task, cj_Worker *worker) {
  cj_Object *now, *dist_now;
  task->status = RUNNING;
  task->worker = worker;

  //fprintf(stderr, "%s\n", task->name);

  cj_Worker_fetch(task, worker);

  fprintf(stderr, "before wait prefetch\n");

  //cj_Worker_wait_prefetch(worker);

  fprintf(stderr, "before prefetch\n");

  cj_Worker_prefetch(worker);

  fprintf(stderr, "after prefetch\n");
  
  //usleep((unsigned int) task->cost);
  (*task->function)((void *) task);
  cj_Worker_wait_execute(worker);

  /* TODO : update output distribution. */
  now = task->arg->dqueue->head;
  while (now) {
    if (now->objtype == CJ_MATRIX) {
      cj_Matrix *matrix = now->matrix;
      cj_Matrix *base   = matrix->base;
      cj_Object *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
      cj_Object *dist_new = cj_Object_new(CJ_DISTRIBUTION);

      if (now->rwtype == CJ_W || now->rwtype == CJ_RW) {
        dist_now = dist->dqueue->head;
        while (dist_now) {
          if (dist_now->distribution->device_id == worker->device_id) {
            cj_Distribution_duplicate(dist_new, dist_now);
            break;
          }
          dist_now = dist_now->next;
        }
        cj_Dqueue_clear(dist);
        cj_Dqueue_push_tail(dist, dist_new);
        cj_Dqueue_push_tail(worker->write_back, cj_Object_append(CJ_MATRIX, matrix));
      }
    }
    now = now->next;
  }

  //cj_Worker_wait_prefetch(worker);

  return 1;
}

void *cj_Worker_entry_point (void *arg) {
  cj_Schedule *schedule = &cj.schedule;
  cj_Worker *me = (cj_Worker *) arg;
  int id, condition, committed;
  float cost;

  id = me->id;
  if (me->device_id != -1) {
    if (me->devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
      cudaSetDevice(me->device_id);
#endif      
      fprintf(stderr, YELLOW "  Worker_entry_point (%d): device(%d) \n" NONE, id, me->device_id);
    }
  }

  condition = 1;
  while (condition) {
    cj_Object *task;

    /* Critical section : access ready_queue. */
    cj_Lock_acquire(&schedule->ready_queue_lock[id]);
    {
      task = cj_Worker_wait_dqueue(me);
    }
    cj_Lock_release(&schedule->ready_queue_lock[id]);

    if (task) {
      committed = cj_Worker_execute(task->task, me);

      if (committed) {
        cj_Task_dependencies_update(task);
      }
    }
  }

  return NULL;
}

void cj_Queue_begin() {
  cj_queue_enable = TRUE;
  cj_Object *vertex, *now;
  cj_Schedule *schedule = &cj.schedule;

  /* Check all task node's status. */
  vertex = cj_Graph_vertex_get();
  now = vertex->dqueue->head;

  while (now) {
    cj_Task *now_task = now->vertex->task;
    cj_Lock_acquire(&now_task->tsk_lock);
    if (now_task->num_dependencies_remaining == 0 && now_task->status == NOTREADY) {
      fprintf(stderr, GREEN "  Sink Point (%d): \n" NONE, now_task->id);
      cj_Task_enqueue(cj_Object_append(CJ_TASK, now_task));
      //fprintf(stderr, GREEN "  ready_queue.size = %d: \n" NONE, schedule->ready_queue->dqueue->size);
    }
    cj_Lock_release(&now_task->tsk_lock);
    now = now->next;
  }
}

void cj_Queue_end() {
  cj_queue_enable = FALSE;
}

void cj_Schedule_init() {
  int i;
  cj_Schedule *schedule = &cj.schedule;
  for (i = 0; i < MAX_WORKER; i++) {
    schedule->ready_queue[i] = cj_Object_new(CJ_DQUEUE);
    schedule->time_remaining[i] = 0.0;
    cj_Lock_new(&schedule->run_lock[i]);
    cj_Lock_new(&schedule->ready_queue_lock[i]);
  }
  cj_Lock_new(&schedule->war_lock);
  cj_Lock_new(&schedule->pci_lock);
  cj_Lock_new(&schedule->gpu_lock);
  cj_Lock_new(&schedule->mic_lock);
}

void cj_Init(int nworker) {
  fprintf(stderr, RED "Init : \n" NONE); 
  fprintf(stderr, "{\n"); 

#ifdef CJ_HAVE_CUDA
  cudaDeviceReset();
#endif

  int i, ret;
  void *(*worker_entry_point)(void *);

  cj_Graph_init();
  cj_Schedule_init();
  cj_Autotune_init();

  if (nworker <= 0) cj_error("Init", "Worker number should at least be 1.");
  cj.nworker = nworker;
  cj.worker = (cj_Worker **) malloc(nworker*sizeof(cj_Worker *));
  if (!cj.worker) cj_error("Init", "memory allocation failed.");

  for (i = 0; i < nworker; i++) {
    cj.worker[i] = cj_Worker_new(CJ_DEV_CPU, i);
    if (!cj.worker[i]) cj_error("Init", "memory allocation failed.");
  }

  cj.ngpu = 2;
  cj.nmic = 0;
  for (i = 0; i < cj.ngpu; i++) {
    cj.device[i] = cj_Device_new(CJ_DEV_CUDA, i); 
    cj_Device_bind(cj.worker[i + 1], cj.device[i]);
  }
  for (i = cj.ngpu; i < cj.ngpu + cj.nmic; i++) {
    cj.device[i] = cj_Device_new(CJ_DEV_MIC, i);
    cj_Device_bind(cj.worker[i + 1], cj.device[i]);
  }


  /* Set up pthread_create parameters. */
  ret = pthread_attr_init(&cj.worker_attr);
  worker_entry_point = cj_Worker_entry_point;

  for (i = 1; i < cj.nworker; i++) {
    ret = pthread_create(&cj.worker[i]->threadid, &cj.worker_attr,
        worker_entry_point, (void *) cj.worker[i]);
    if (ret) cj_error("Init", "Could not create threads properly");
  }

  fprintf(stderr, "}\n"); 
}

void cj_Term() {
  fprintf(stderr, RED "Term : \n" NONE); 
  fprintf(stderr, "{\n"); 

  int i, ret;

  cj_Graph_output_dot();

  for (i = 1; i < cj.nworker; i++) {
    ret = pthread_join(cj.worker[i]->threadid, NULL);
    if (ret) cj_error("Term", "Could not join threads properly");
  }

  fprintf(stderr, "}\n"); 
}
