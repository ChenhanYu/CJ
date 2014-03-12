#include <stdio.h>
#include <stdlib.h>
#include "cj_Object.h"

void cj_Object_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_OBJECT_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

cj_Task *cj_Task_new (char *name) {
  cj_Task *task;
  task = (cj_Task *) malloc(sizeof(cj_Task));
  if (!task) cj_Object_error("Task_new", "memory allocation failed.");
  task->name = name;
  task->in = (cj_Object *) malloc(sizeof(cj_Object));
  if (!task->in) cj_Object_error("Task_new", "memory allocation failed.");
  task->out = (cj_Object *) malloc(sizeof(cj_Object));
  if (!task->out) cj_Object_error("Task_new", "memory allocation failed.");
  return task;
}

int cj_Dqueue_get_size (cj_Object *object) {
  if (object->objtype == CJ_DQUEUE) return (object->dqueue->size);
  else cj_Object_error("Dqueue_get_size", "The object is not a dqueue.");
}

cj_Dqueue *cj_Dqueue_new () {
  cj_Dqueue *dqueue;
  dqueue = (cj_Dqueue*) malloc(sizeof(cj_Dqueue));
  if (!dqueue) cj_Object_error("Dqueue_new", "memory allocation failed.");
  dqueue->head = NULL;
  dqueue->tail = NULL;	
  dqueue->size = 0;
  return dqueue;
}

cj_Object *cj_Dqueue_pop_head (cj_Object *object) {
  cj_Dqueue *dqueue;
  cj_Object *head;

  if (object->objtype == CJ_DQUEUE) {
    dqueue = object->dqueue;
    if (dqueue->size > 0) {
      head = dqueue->head;
      dqueue->head = head->next;
      dqueue->size --;
    }
    return head;
  }
  else cj_Object_error("Dqueue_pop_head", "The object is not a dqueue.");	
}

cj_Object *cj_Dqueue_pop_tail (cj_Object *object) {
  cj_Dqueue *dqueue;
  cj_Object *tail;

  if (object->objtype == CJ_DQUEUE) {
    dqueue = object->dqueue;
    if (dqueue->size > 0) {
      tail = dqueue->tail;
      dqueue->tail = tail->prev;
      dqueue->size --;
    }
    return tail;
  }
  else cj_Object_error("Dqueue_pop_tail", "The object is not a dqueue.");	
}

void cj_Dqueue_push_head (cj_Object *object, cj_Object *target) {
  cj_Dqueue *dqueue;
  cj_Object *head;
  if (object->objtype == CJ_DQUEUE) {
    dqueue = object->dqueue;
    if (dqueue->size == 0) {
      dqueue->head = target;
	  dqueue->tail = target;
	}
	else {
	  head = dqueue->head;
	  dqueue->head = target;
	  target->prev = NULL;
	  target->next = head;
	  head->prev = target; 
	}
	dqueue->size ++;
  }
  else cj_Object_error("Dqueue_push_head", "The object is not a dqueue.");
}

void cj_Dqueue_push_tail (cj_Object *object, cj_Object *target) {
  cj_Dqueue *dqueue;
  cj_Object *tail;
  if (object->objtype == CJ_DQUEUE) {
    dqueue = object->dqueue;
    if (dqueue->size == 0) {
      dqueue->head = target;
	  dqueue->tail = target;
	}
	else {
	  tail = dqueue->tail;
	  dqueue->tail = target;
	  target->prev = tail;
	  target->next = NULL;
	  tail->next = target; 
	}
	dqueue->size ++;
  }
  else cj_Object_error("Dqueue_push_tail", "The object is not a dqueue.");
}

void cj_Matrix_set (cj_Matrix *matrix, int m, int n) {
  int i;
  if (m <= 0 && n <= 0) cj_Object_error("Matrix_new", "m and n should at least be 1.");
  matrix->m = m;
  matrix->n = n;
  matrix->mb = (m - 1)/BLOCK_SIZE + 1;
  matrix->nb = (n - 1)/BLOCK_SIZE + 1;
  matrix->offm = 0;
  matrix->offn = 0;
  matrix->rset = (cj_Object **) malloc((matrix->mb)*sizeof(cj_Object*));
  if (!matrix->rset) cj_Object_error("Matrix_new", "memory allocation failed.");
  matrix->wset = (cj_Object **) malloc((matrix->mb)*sizeof(cj_Object*));
  if (!matrix->wset) cj_Object_error("Matrix_new", "memory allocation failed.");
  for (i = 0; i < matrix->mb; i++) {
     matrix->rset[i] = (cj_Object *) malloc((matrix->nb)*sizeof(cj_Object));
     if (!matrix->rset[i]) cj_Object_error("Matrix_new", "memory allocation failed.");
     matrix->wset[i] = (cj_Object *) malloc((matrix->nb)*sizeof(cj_Object));
     if (!matrix->wset[i]) cj_Object_error("Matrix_new", "memory allocation failed.");
  }
  matrix->base = matrix;
}

cj_Matrix *cj_Matrix_new () {
  cj_Matrix *matrix;
  matrix = (cj_Matrix *) malloc(sizeof(cj_Matrix));
  if (!matrix) cj_Object_error("Matrix_new", "memory allocation failed.");
  matrix->m = 0;
  matrix->n = 0;
  matrix->mb = 0;
  matrix->nb = 0;;
  matrix->offm = 0;
  matrix->offn = 0;
  matrix->rset = NULL;
  matrix->wset = NULL;
  matrix->base = NULL;
  return matrix; 
}

/* This routine is rewritten based on libflame. */
void cj_Matrix_part_2x1 (cj_Object *A, cj_Object *A1,
                                       cj_Object *A2,
						 int mb,       cj_Side side) {
  cj_Matrix *a, *a1, *a2;

  if (A->objtype != CJ_MATRIX || A1->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_part_2x1", "This is not a matrix.");
  a = A->matrix;
  a1 = A1->matrix;
  a2 = A2->matrix;
  if (mb > a->m) mb = a->m; 
  if (side == CJ_BOTTOM) mb = a->m - mb;

  a1->m    = mb;
  a1->n    = a->n;
  a1->offm = a->offm;
  a1->offn = a->offn;
  a1->base = a->base;

  a2->m    = a->m - mb;
  a2->n    = a->n;
  a2->offm = a->offm + mb;
  a2->offn = a->offn;
  a2->base = a->base;
}

void cj_Matrix_repart_2x1_to_3x1 (cj_Object *AT, cj_Object *A0,
                                                 cj_Object *A1,
                                  cj_Object *AB, cj_Object *A2,
								  int mb,        cj_Side side) {
  cj_Matrix *at, *a0, *a1, *ab, *a2;

  if (AT->objtype != CJ_MATRIX || A0->objtype != CJ_MATRIX || A1->objtype != CJ_MATRIX
      || AB->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_2x1_3x1", "This is not a matrix.");
  at = AT->matrix;
  a0 = A0->matrix;
  a1 = A1->matrix;
  ab = AB->matrix;
  a2 = A2->matrix;

  if (side == CJ_TOP) {
    cj_Matrix_part_2x1 (AT, A0, A1, mb, CJ_BOTTOM);
	a2->m = ab->m;
	a2->n = ab->n;
    a2->offm = ab->offm;
    a2->offn = ab->offn;
    a2->base = ab->base;
  }
  else {
	a0->m = at->m;
	a0->n = at->n;
    a0->offm = at->offm;
    a0->offn = at->offn;
    a0->base = at->base;
    cj_Matrix_part_2x1 (AB, A1, A2, mb, CJ_TOP);
  }
}

void cj_Matrix_cont_3x1_to_2x1 (cj_Object *AT, cj_Object *A0,
                                               cj_Object *A1,
                                cj_Object *AB, cj_Object *A2,
								               cj_Side side) {
  cj_Matrix *at, *a0, *a1, *ab, *a2;

  if (AT->objtype != CJ_MATRIX || A0->objtype != CJ_MATRIX || A1->objtype != CJ_MATRIX
      || AB->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_2x1_3x1", "This is not a matrix.");
  at = AT->matrix;
  a0 = A0->matrix;
  a1 = A1->matrix;
  ab = AB->matrix;
  a2 = A2->matrix;

  if (side == CJ_TOP) {
    at->m    = a0->m + a1->m;
	at->n    = a0->n;
	at->offm = a0->offm;
	at->offn = a0->offn;
	at->base = a0->base;

    ab->m    = a2->m;
	ab->n    = a2->n;
	ab->offm = a2->offm;
	ab->offn = a2->offn;
	ab->base = a2->base;
  }
  else {
    at->m    = a0->m;
	at->n    = a0->n;
	at->offm = a0->offm;
	at->offn = a0->offn;
	at->base = a0->base;

    ab->m    = a1->m + a2->m;
	ab->n    = a1->n;
	ab->offm = a1->offm;
	ab->offn = a1->offn;
	ab->base = a1->base;
  }
}

void cj_Matrix_delete (cj_Matrix *matrix) {

}

cj_Object *cj_Object_new (cj_objType type) {
  cj_Object *object;
  object = (cj_Object*) malloc(sizeof(cj_Object));
  if (!object) cj_Object_error("new", "memory allocation failed.");

  if (type == CJ_MATRIX) {
    object->objtype = CJ_MATRIX;
	object->matrix = cj_Matrix_new();
  }
  else if (type == CJ_DQUEUE) {
    object->objtype = CJ_DQUEUE;
    object->dqueue = cj_Dqueue_new();
  }
  else if (type == CJ_TASK) {
    object->objtype = CJ_TASK;
	/* Need to call the specific constructor. */
  }

  object->prev = NULL;
  object->next = NULL;

  return object;
}
