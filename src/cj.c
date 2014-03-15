#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "cj_Object.h"
#include "cj_Graph.h"
#include "cj_Device.h"
#include "cj_Macro.h"
#include "cj.h"

/* This is a dqueue. */

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
  cj_Task *task;
  task = (cj_Task *) malloc(sizeof(cj_Task));
  if (!task) cj_error("Task_new", "memory allocation failed.");
  task->id = taskid;
  taskid ++;
  task->status = ALLOCATED_ONLY;
  task->function = NULL;
  task->num_dependencies_remaining = 0;
  cj_Lock_new(&task->tsk_lock);
  task->in = cj_Object_new(CJ_DQUEUE);
  if (!task->in) cj_error("Task_new", "memory allocation failed.");
  task->out = cj_Object_new(CJ_DQUEUE);
  if (!task->out) cj_error("Task_new", "memory allocation failed.");
  task->status = NOTREADY;
  return task;
}

void cj_Task_dependency_add (cj_Object *out, cj_Object *in) {
  cj_Task *task_out, *task_in;
  if (out->objtype != CJ_TASK || in->objtype != CJ_TASK) 
    cj_error("Task_dependency_add", "The object is not a task.");
  task_out = out->task;
  task_in  = in->task;

  cj_Lock_acquire(&task_in->tsk_lock);
  cj_Dqueue_push_tail(task_out->out, cj_Object_append(CJ_TASK, (void *) task_in));
  cj_Lock_release(&task_in->tsk_lock);
  cj_Lock_acquire(&task_out->tsk_lock);
  cj_Dqueue_push_tail(task_in->in, cj_Object_append(CJ_TASK, (void *) task_out));
  cj_Lock_release(&task_out->tsk_lock);

  /* Increase the task's remaining denpendencies. */
  task_in->num_dependencies_remaining ++;
}

void cj_Task_enqueue(cj_Object *target) {
  if (target->objtype != CJ_TASK) cj_error("Task_enqueue", "The object is not a task.");
  cj_Schedule *schedule = &cj.schedule;
  cj_Lock_acquire(&schedule->run_lock);
  target->task->status = QUEUED;
  cj_Dqueue_push_tail(schedule->ready_queue, target);
  cj_Lock_release(&schedule->run_lock);
}

void cj_Task_dependencies_update (cj_Object *target) {
  cj_Task *task;
  if (target->objtype != CJ_TASK) cj_error("Task_dependencies_update", "The object is not a task.");
  task = target->task;
  cj_Object *now = task->out->dqueue->head;

  while (now) {
    cj_Task *child = now->task;
	cj_Lock_acquire(&child->tsk_lock);
	child->num_dependencies_remaining --;
	if (child->num_dependencies_remaining < 0) cj_error("Task_dependencies_update", "Remaining dependencies can't be negative.");
	if (child->num_dependencies_remaining == 0 && child->status == NOTREADY) cj_Task_enqueue(cj_Object_append(CJ_TASK, (void*) child));
	cj_Lock_release(&child->tsk_lock);
    now = now->next;
  }
}

cj_Object *cj_Worker_wait_dqueue () {
  cj_Schedule *schedule = &cj.schedule;
  cj_Object *task;
  task = cj_Dqueue_pop_head(schedule->ready_queue);
  /*
  if (task) fprintf(stderr, YELLOW "  Worker_wait_dqueue (%d, %s): \n" NONE, task->task->id, task->task->name);
  else {
    fprintf(stderr, YELLOW "  Worker_wait_dqueue : ready_queue is empty.\n" NONE);
	sleep(1);
  }
  */
  return task;
}

cj_Worker *cj_Worker_new (cj_devType devtype, int id) {
  fprintf(stderr, RED "  Worker_new (%d): \n" NONE, id); 
  fprintf(stderr, "  {\n"); 
  cj_Worker *worker;

  worker = (cj_Worker *) malloc(sizeof(cj_Worker));
  if (!worker) cj_error("Worker_new", "memory allocation failed.");

  worker->devtype = devtype;
  worker->id = id;
  worker->cj_ptr = &cj;

  fprintf(stderr, "  }\n"); 
  return worker;
}

int cj_Worker_execute (cj_Task *task) {
  task->status = RUNNING;
  sleep(1);
  fprintf(stderr, YELLOW "  Worker_execute (%d, %s): \n" NONE, task->id, task->name); 
  task->status = DONE;
  return 1;
}

void *cj_Worker_entry_point (void *arg) {
  cj_Schedule *schedule = &cj.schedule;
  cj_Worker *me;
  int id, condition, committed;

  me = (cj_Worker *) arg;
  id = me->id;
  fprintf(stderr, YELLOW "  Worker_entry_point (%d): \n" NONE, id);

  condition = 1;
  while (condition) {
	cj_Object *task;
	/* Access ready_queue, a critical section. */
    cj_Lock_acquire(&schedule->run_lock);
	task = cj_Worker_wait_dqueue();
    cj_Lock_release(&schedule->run_lock);

	if (task) {
      committed = cj_Worker_execute(task->task);
	  if (committed) cj_Task_dependencies_update(task);
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
    if (now_task->num_dependencies_remaining == 0 && now_task->status == NOTREADY) {
      fprintf(stderr, GREEN "  Sink Point (%d): \n" NONE, now_task->id);
	  cj_Task_enqueue(cj_Object_append(CJ_TASK, now_task));
      fprintf(stderr, GREEN "  ready_queue.size = %d: \n" NONE, schedule->ready_queue->dqueue->size);
	}
	now = now->next;
  }
}

void cj_Queue_end() {
  cj_queue_enable = FALSE;
}

void cj_Schedule_init() {
  cj_Schedule *schedule = &cj.schedule;
  schedule->ready_queue = cj_Object_new(CJ_DQUEUE);
  cj_Lock_new(&schedule->run_lock);
  cj_Lock_new(&schedule->pci_lock);
  cj_Lock_new(&schedule->gpu_lock);
  cj_Lock_new(&schedule->mic_lock);
}

void cj_Init(int nworker) {
  fprintf(stderr, RED "Init : \n" NONE); 
  fprintf(stderr, "{\n"); 

  int i, ret;
  void *(*worker_entry_point)(void *);

  cj_Graph_init();

  cj_Schedule_init();

  if (nworker <= 0) cj_error("Init", "Worker number should at least be 1.");
  cj.nworker = nworker;
  cj.worker = (cj_Worker **) malloc(nworker*sizeof(cj_Worker *));
  if (!cj.worker) cj_error("Init", "memory allocation failed.");

  for (i = 0; i < nworker; i++) {
    cj.worker[i] = cj_Worker_new(CJ_DEV_CPU, i);
    if (!cj.worker[i]) cj_error("Init", "memory allocation failed.");
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
