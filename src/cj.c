/*
 *  cj.h
 *  Chenhan D. Yu
 *  Created: Mar 30, 2014
 *
 * */

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
#include "cj_Lapack.h"
#include "cj_Autotune.h"
#include "cj_Profile.h"
#include "cj.h"

static cj_t cj;
static int taskid = 0;
static cj_Bool cj_queue_enable = FALSE;

/**
 *  @brief cj_error
 *
 *  @param *func_name function name pointer
 *  @param *msg_text error message pointer
 */

void cj_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}


/* ---------------------------------------------------------------------
 * cj_Lock
 * ---------------------------------------------------------------------
 * */

/**
 * @brief  Create a pthread mutex.
 * @param  *lock lock pointer 
 */
void cj_Lock_new (cj_Lock *lock) {
  int ret = pthread_mutex_init(&(lock->lock), NULL);
  if (ret) cj_error("Lock_new", "Could not initial locks properly.");
}

/**
 * @brief  Delete a pthread mutex.
 * @param  *lock lock pointer 
 */
void cj_Lock_delete (cj_Lock *lock) {
  int ret = pthread_mutex_destroy(&(lock->lock));
  if (ret) cj_error("Lock_delete", "Could not destroy locks properly.");
}

/**
 * @brief  This function will try to acquire the mutex. It will block until has 
 *         been acquired.
 * @param  *lock lock pointer 
 */
void cj_Lock_acquire (cj_Lock *lock) {
  int ret = pthread_mutex_lock(&(lock->lock));
  if (ret) cj_error("Lock_acquire", "Could not acquire locks properly.");
}

/**
 * @brief  This function will release the mutex.
 * @param  *lock lock pointer 
 */
void cj_Lock_release (cj_Lock *lock) {
  int ret = pthread_mutex_unlock(&(lock->lock));
  if (ret) cj_error("Lock_release", "Could not release locks properly.");
}


/* ---------------------------------------------------------------------
 * cj_Distribution
 * ---------------------------------------------------------------------
 * */

/**
 * @brief  Create a new distribution.
 * @return an uninitialized distribution 
 */
cj_Distribution *cj_Distribution_new () {
  cj_Distribution *dist = (cj_Distribution *) malloc(sizeof(cj_Distribution));
  if (!dist) cj_error("Distribution_new", "memory allocation failed.");

  int i;
  for (i = 0; i < MAX_DEV + 1; i++) {
    dist->avail[i] = FALSE;
    if (i > 0) {
      /* Be careful, here the index is different. */
      dist->device[i] = cj.device[i - 1];
    }
    dist->line[i]  = -1;
  }
  dist->avail[0] = TRUE;
  cj_Lock_new(&dist->lock);
  return dist;
}


/* ---------------------------------------------------------------------
 * cj_Task
 * ---------------------------------------------------------------------
 * */

/**
 * @brief  Create a new task.
 * @return an uninitialized task 
 */
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

/**
 * @brief  Set the type and function type for a task.
 * @param  *task :target task pointer
 * @param  tasktype :can be gemm, syrk, trsm ... etc
 * @param  function :function pointer 
 */
void cj_Task_set (cj_Task *task, cj_taskType tasktype, void (*function)(void*)) {
  task->tasktype = tasktype;
  task->function = function;
}

/**
 * @brief  Analysis dependencies and update the dependency graph according to
 *         the task's inputs, outputs and read/write features.
 * @param  *task target task pointer
 */
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
	  /* read set */
	  set_r = matrix->base->rset[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
	  /* write set */
	  set_w = matrix->base->wset[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

	  /* data dependency */
	  if (now->rwtype == CJ_R || now->rwtype == CJ_RW) {
		cj_Dqueue_push_tail(set_r, cj_Object_append(CJ_TASK, (void *) task->task));
		if (cj_Dqueue_get_size(set_w) > 0) {
		  cj_Object *now_w = set_w->dqueue->head;
		  while (now_w) {
			if (now_w->task->id != task->task->id) {
			  cj_Object *edge = cj_Object_new(CJ_EDGE);
			  cj_Edge_set(edge, now_w, task, FALSE);
			  cj_Graph_edge_add(edge);
			  cj_Task_dependency_add(now_w, task);
			  fprintf(stderr, "          %d->%d.\n", now_w->task->id, task->task->id);
			}
			now_w = now_w->next;
		  }
		}
	  }
	  /* anti dependency */
	  if (now->rwtype == CJ_W || now->rwtype == CJ_RW) {
        cj_Object *now_r = set_r->dqueue->head; 
        while (now_r) {
          if (now_r->task->id != task->task->id) {
            cj_Object *edge = cj_Object_new(CJ_EDGE);
            cj_Edge_set(edge, now_r, task, TRUE);
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

/**
 * @brief  Describes a dependency between two tasks.
 * @param  *out target task pointer depends on in
 * @param  *in 
 */
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

/**
 * @brief  Enqueue a task satisfying all dependencies to the ready queue.
 * @param  *target the task waiting for enqueuing
 */
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
    //fprintf(stderr, "  Enqueue task<%d> to worker[%d]\n", target->task->id, dest);
  }
  cj_Lock_release(&schedule->ready_queue_lock[dest]);
}

/**
 * @brief  Update dependencies relating the target task.
 * @param  *target the target task
 */
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


/* ---------------------------------------------------------------------
 * cj_Worker
 * ---------------------------------------------------------------------
 * */


/**
 * @brief  Fetch a task from the ready queue.
 * @param  *worker the target worker
 * @retval task 
 * @retval null if the queue is empty
 */
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
  int i;
  /* Iterate all the arguments of the task */
  while (arg_I) {
    if (arg_I->objtype == CJ_MATRIX) {
      cj_Matrix       *matrix = arg_I->matrix;
      cj_Matrix       *base   = matrix->base;
      cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];
      
      int dest = worker->device_id + 1;

      /* Acquire the distribution lock */
      cj_Lock_acquire(&dist->lock);
      {
        /* if the matrix has no latest copy on this worker */
        if (dist->avail[dest] == FALSE) {
          /* if there is no copy in the main memory of CPU */
          if (dist->avail[0] == FALSE) {
            for (i = 1; i < MAX_DEV + 1; i++) {
              if (dist->avail[i] == TRUE) {
                /* Sync Write Back */
                fprintf(stderr, RED "(%d) Cache_writeback: %d\n" NONE, worker->device_id, i - 1);
                /* pick a location from the distribution and write back */
                cj_Cache_write_back(dist->device[i], dist->line[i], arg_I);
                dist->avail[0] = TRUE;
                break;
              }
            }
          }
          if (worker->device_id != -1) {
            dist->line[dest]  = cj_Cache_fetch(dist->device[dest], arg_I);
            dist->avail[dest] = TRUE;
            fprintf(stderr, RED "(%d) Cache_fetch: %d\n" NONE, worker->device_id, dist->line[dest]);
          }
        }
        /* Lock the cache line here. */
      }
      cj_Lock_release(&dist->lock);
    }
    arg_I = arg_I->next;
  }
  if (worker->device_id != -1) {
    cj_Cache_sync(cj.device[worker->device_id]);
  }
}

int cj_Worker_prefetch_d2h (cj_Worker *worker) {
  cj_Object *object = worker->write_back->dqueue->head;
  if (!object) return 0;

  if (object->objtype == CJ_MATRIX) {
    cj_Matrix       *matrix = object->matrix;
    cj_Matrix       *base   = matrix->base;
    cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

    int dest = worker->device_id + 1;

    if (dist->avail[0] == TRUE) {
      object = cj_Dqueue_pop_head(worker->write_back);
      return 0;
    }
    if (dist->avail[dest] == FALSE) {
      object = cj_Dqueue_pop_head(worker->write_back);
      return 0;
      //cj_error("cj_Worker_prefetch_d2h", "missing distribution");
    }
    /* This is an async write back and will be sync later. */
    cj_Cache_async_write_back(dist->device[dest], dist->line[dest], object);
    fprintf(stderr, GREEN "(%d) Cache_prefetch_d2h: %d\n" NONE, worker->device_id, dist->line[dest]);
  }
  return 1;
}

//host to device
int cj_Worker_prefetch_h2d (cj_Worker *worker) {
  cj_Object *arg_I = NULL;
  cj_Schedule *schedule = &cj.schedule;
  cj_Object *task = schedule->ready_queue[worker->id]->dqueue->head;

  if (worker->device_id == -1) return 0;
  if (task) arg_I = task->task->arg->dqueue->head;
  else return 0;

  while (arg_I) {
    if (arg_I->objtype == CJ_MATRIX) {
      cj_Matrix       *matrix = arg_I->matrix;
      cj_Matrix       *base   = matrix->base;
      cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

      int dest = worker->device_id + 1;

      if (dist->avail[dest] == FALSE) {
        if (dist->avail[0] == TRUE) {
          /* This is an async fetch and will be sync later. */
          dist->line[dest]  = cj_Cache_fetch(dist->device[dest], arg_I);
          dist->avail[dest] = TRUE;
          fprintf(stderr, GREEN "(%d) Cache_prefetch_h2d: %d\n" NONE, worker->device_id, dist->line[dest]);
        }
      }
    }
    arg_I = arg_I->next;
  }
  return 1;
}

void cj_Worker_wait_prefetch (cj_Worker *worker, int h2d, int d2h) {
  if (h2d || d2h) {
    cj_Cache_sync(cj.device[worker->device_id]);    
  }
  if (d2h) {
    cj_Object *object = cj_Dqueue_pop_head(worker->write_back);
    if (!object) cj_error("cj_Worker_wait_prefetch", "missing object in write_back queue");
    if (object->objtype == CJ_MATRIX) {
      cj_Matrix       *matrix = object->matrix;
      cj_Matrix       *base   = matrix->base;
      cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

      int dest = worker->device_id + 1;
      cj_Cache        *cache  = &dist->device[dest]->cache;

      /* Critical Section */
      cj_Lock_acquire(&dist->lock);
      {
        dist->avail[0] = TRUE;
      }
      cj_Lock_release(&dist->lock);

      if (cache->status[dist->line[dest]] == CJ_CACHE_DIRTY) {
        cache->status[dist->line[dest]] = CJ_CACHE_CLEAN;
      }
    }
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

  worker->devtype      = devtype;
  worker->id           = id;
  worker->device_id    = -1;
  worker->cj_ptr       = &cj;
  worker->write_back   = cj_Object_new(CJ_DQUEUE); 
  worker->current_task = NULL;

  fprintf(stderr, "  }\n"); 
  return worker;
}

/* performance model. Estimate the time cost for both computation and communication of CPU and GPU */
float cj_Worker_estimate_cost (cj_Task *task, cj_Worker *worker) {
  /* Here it is very similar to a construct function in C++/Java. We implement in C */
  cj_Autotune *model = cj_Autotune_get_ptr(); 
  float comp_cost = 0.0, comm_cost = 0.0, cost = 0.0;

  if (worker->devtype == CJ_DEV_CUDA) {
    if (task->function == &cj_Gemm_nn_task_function)
      comp_cost = model->cublas_dgemm[0];
    if (task->function == &cj_Syrk_ln_task_function)
      comp_cost = model->cublas_dsyrk[0];
    if (task->function == &cj_Trsm_rlt_task_function)
      comp_cost = model->cublas_dtrsm[0];
    if (task->function == &cj_Chol_l_task_function)
      comp_cost = model->hybrid_dpotrf[0];
    /* Scan through all arguments. */
    cj_Object *arg_I = task->arg->dqueue->head;
    while (arg_I) {
      if (arg_I->objtype == CJ_MATRIX) {
        cj_Matrix       *matrix = arg_I->matrix;
        cj_Matrix       *base   = matrix->base;
        cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

        int dest = worker->device_id + 1;
        /* if the argument is not available on the device */
        if (dist->avail[dest] == FALSE) {
          comm_cost = model->pci_bandwidth;
          /* if the argument is not available on the host */
          if (dist->avail[0] == FALSE) {
            comm_cost = 2*(model->pci_bandwidth);
          }
        }
      }
      cost += comm_cost;
      arg_I = arg_I->next;
    }
  }
  else if (worker->devtype == CJ_DEV_CPU) {
    if (task->function == &cj_Gemm_nn_task_function)
      comp_cost = model->mkl_dgemm[0];
    if (task->function == &cj_Syrk_ln_task_function)
      comp_cost = model->mkl_dsyrk[0];
    if (task->function == &cj_Trsm_rlt_task_function)
      comp_cost = model->mkl_dtrsm[0];
    if (task->function == &cj_Chol_l_task_function)
      comp_cost = model->mkl_dpotrf[0];
    cj_Object *arg_I = task->arg->dqueue->head;
    while (arg_I) {
      if (arg_I->objtype == CJ_MATRIX) {
        cj_Matrix       *matrix  = arg_I->matrix;
        cj_Matrix       *base    = matrix->base;
        cj_Distribution *dist    = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

        int dest = worker->device_id + 1;
        /* if the argument is not available on the host */
        if (dist->avail[dest] == FALSE) {
          comm_cost = model->pci_bandwidth;
        }
      }
      cost += comm_cost;
      arg_I = arg_I->next;
    }
  }
  cost += comp_cost;

  return cost;
}

int cj_Worker_execute (cj_Task *task, cj_Worker *worker) {
  task->status = RUNNING;
  task->worker = worker;
  worker->current_task = task;
  int i;

  //fprintf(stderr, "%s\n", task->name);


  /* fetch... the core of execute algorithm */
  cj_Profile_worker_record(worker, CJ_EVENT_FETCH_BEG);
  cj_Worker_fetch(task, worker);
  cj_Profile_worker_record(worker, CJ_EVENT_FETCH_END);
  //fprintf(stderr, "before wait prefetch\n");
  
  /* prefetch.... */
  int d2h = cj_Worker_prefetch_d2h(worker);
  int h2d = cj_Worker_prefetch_h2d(worker);
  //fprintf(stderr, "after prefetch\n");
  //usleep((unsigned int) task->cost);
  //fprintf(stderr, "after fetch\n");
  
  /* Change cache status here. */
  /* Kernel function is here... */
  cj_Profile_worker_record(worker, CJ_EVENT_TASK_RUN_BEG);
  (*task->function)((void *) task);
  cj_Worker_wait_execute(worker);
  cj_Profile_worker_record(worker, CJ_EVENT_TASK_RUN_END);


  /* if did prefech or write back, then update the distribution of those prefectched objects*/
  cj_Object *arg_I = task->arg->dqueue->head;
  while (arg_I) {
    if (arg_I->objtype == CJ_MATRIX) {
      cj_Matrix       *matrix = arg_I->matrix;
      cj_Matrix       *base   = matrix->base;
      cj_Distribution *dist   = base->dist[matrix->offm/BLOCK_SIZE][matrix->offn/BLOCK_SIZE];

      int dest = worker->device_id + 1;

      if (arg_I->rwtype == CJ_W || arg_I->rwtype == CJ_RW) {

        /* Critical Section */
        cj_Lock_acquire(&dist->lock);
        {
          for (i = 0; i < MAX_DEV + 1; i++) {
            if (i != dest) {
              dist->avail[i] = FALSE;
              dist->line[i] = -1;
            }
          }
        }
        cj_Lock_release(&dist->lock);
        cj_Dqueue_push_tail(worker->write_back, cj_Object_append(CJ_MATRIX, (void *) matrix));
      }
    }
    arg_I = arg_I->next;
  }

  cj_Worker_wait_prefetch(worker, h2d, d2h);
  worker->current_task = NULL;

  return 1;
}

void *cj_Worker_entry_point (void *arg) {
  cj_Schedule *schedule = &cj.schedule;
  cj_Worker *me = (cj_Worker *) arg;
  int id,committed;
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

  while (1) {
    cj_Object *task;

    /* Critical section : access ready_queue. */
    cj_Lock_acquire(&schedule->ready_queue_lock[id]);
    {
      task = cj_Worker_wait_dqueue(me);
    }
    cj_Lock_release(&schedule->ready_queue_lock[id]);

    if (task) {
      committed = cj_Worker_execute(task->task, me);

	  /* if commit then update dependencies */
      if (committed) {
        cj_Task_dependencies_update(task);
        /* Critical section : access ntask. */
        cj_Lock_acquire(&schedule->ntask_lock);
        {
          schedule->ntask ++;
        }
        cj_Lock_release(&schedule->ntask_lock);
      }
    }
    int ntask = cj_Dqueue_get_size(cj_Graph_vertex_get());
    if (cj.terminate == TRUE && schedule->ntask == ntask) break;
  }

  return NULL;
}

/* ---------------------------------------------------------------------
 * cj_Queue
 * ---------------------------------------------------------------------
 * */

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

/* ---------------------------------------------------------------------
 * cj_Schedule
 * ---------------------------------------------------------------------
 *  */

void cj_Schedule_init() {
  int i;
  cj_Schedule *schedule = &cj.schedule;
  schedule->ntask = 0;
  for (i = 0; i < MAX_WORKER; i++) {
    schedule->ready_queue[i] = cj_Object_new(CJ_DQUEUE);
    schedule->time_remaining[i] = 0.0;
    cj_Lock_new(&schedule->run_lock[i]);
    cj_Lock_new(&schedule->ready_queue_lock[i]);
  }
  cj_Lock_new(&schedule->ntask_lock);
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
  cj.terminate = FALSE;
  cj_Profile_init();
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

  cj.ngpu = GPU_NUM;
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
  cj.terminate = TRUE;

  for (i = 1; i < cj.nworker; i++) {
    ret = pthread_join(cj.worker[i]->threadid, NULL);
    if (ret) cj_error("Term", "Could not join threads properly");
  }

  fprintf(stderr, "}\n"); 
}
