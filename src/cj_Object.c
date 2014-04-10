#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <CJ.h>
#include "cj_Macro.h"
#include "cj.h"
#include "cj_Graph.h"
#include "cj_Object.h"
#include "cj_Device.h"

void cj_Object_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_OBJECT_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Distribution_duplicate (cj_Object *target, cj_Object *object) {
  if (object->objtype != CJ_DISTRIBUTION || target->objtype != CJ_DISTRIBUTION) { 
    cj_Object_error("Distribution_set", "The object and target are not a distribution.");
  }
  cj_Distribution *dist_tar;
  cj_Distribution *dist_obj;
  dist_tar = target->distribution;
  dist_obj = object->distribution;

  dist_tar->device    = dist_obj->device;
  dist_tar->device_id = dist_obj->device_id;
  dist_tar->cache_id  = dist_obj->cache_id;
}

void cj_Distribution_set (cj_Object *object, cj_Device *device, int device_id, int cache_id) {
  if (object->objtype != CJ_DISTRIBUTION) {
    cj_Object_error("Distribution_set", "The object is not a distribution.");
  }
  cj_Distribution *dist = object->distribution;
  dist->device    = device;
  dist->device_id = device_id;
  dist->cache_id  = cache_id; 
}

cj_Distribution *cj_Distribution_new () {
  cj_Distribution *dist = (cj_Distribution *) malloc(sizeof(cj_Distribution));
  if (!dist) cj_Object_error("Distribution_new", "memory allocation failed.");
  /* -1 represent that this is distributed on CPU. */
  dist->device    = NULL;
  dist->device_id = -1;
  dist->cache_id  = -1;
  dist->len       = BLOCK_SIZE*BLOCK_SIZE*sizeof(double); 
  return dist;
}

int cj_Dqueue_get_size (cj_Object *object) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_get_size", "The object is not a dqueue.");
    return (object->dqueue->size);
  }
  return (object->dqueue->size);
}

cj_Dqueue *cj_Dqueue_new () {
  cj_Dqueue *dqueue = (cj_Dqueue*) malloc(sizeof(cj_Dqueue));
  if (!dqueue) {
    cj_Object_error("Dqueue_new", "memory allocation failed.");
  }
  dqueue->head = NULL;
  dqueue->tail = NULL;	
  dqueue->size = 0;
  return dqueue;
}

cj_Object *cj_Dqueue_pop_head (cj_Object *object) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_pop_head", "The object is not a dqueue.");	
  }
  cj_Dqueue *dqueue = object->dqueue;
  cj_Object *head = NULL;

  if (dqueue->size > 1) {
    head = dqueue->head;
    dqueue->head = head->next;
    dqueue->head->prev = NULL;
    dqueue->size --;
  }
  else if (dqueue->size == 1) {
    head = dqueue->head;
    dqueue->head = NULL;
    dqueue->tail = NULL;
    dqueue->size --;
  }
  if (head) {
    head->prev = NULL;
    head->next = NULL;
  }
  return head;
}

cj_Object *cj_Dqueue_pop_tail (cj_Object *object) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_pop_tail", "The object is not a dqueue.");	
  }
  cj_Dqueue *dqueue = object->dqueue;
  cj_Object *tail = NULL;

  if (dqueue->size > 1) {
    tail = dqueue->tail;
    dqueue->tail = tail->prev;
    dqueue->tail->next = NULL;
    dqueue->size --;
  }
  else if (dqueue->size == 1) {
    tail = dqueue->tail;
    dqueue->head = NULL;
    dqueue->tail = NULL;
    dqueue->size --;
  }
  if (tail) {
    tail->prev = NULL;
    tail->next = NULL;
  }
  return tail;
}

void cj_Dqueue_push_head (cj_Object *object, cj_Object *target) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_push_head", "The object is not a dqueue.");
  }
  if (!target) {
    cj_Object_error("Dqueue_push_head", "Target is empty.");
  }
  cj_Dqueue *dqueue = object->dqueue;
  cj_Object *head;

  target->prev = NULL;
  target->next = NULL;

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

void cj_Dqueue_push_tail (cj_Object *object, cj_Object *target) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_push_tail", "The object is not a dqueue.");
  }
  if (!target) {
    cj_Object_error("Dqueue_push_tail", "Target is empty.");
  }
  cj_Dqueue *dqueue = object->dqueue;
  cj_Object *tail;

  target->prev = NULL;
  target->next = NULL;

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

void cj_Dqueue_clear(cj_Object *object) {
  if (object->objtype != CJ_DQUEUE) {
    cj_Object_error("Dqueue_clear", "The object is not a dqueue.");
  }
  cj_Object *tmp;

  while (cj_Dqueue_get_size(object) != 0) {
    tmp = cj_Dqueue_pop_tail(object);
    free(tmp);
  }
}

void cj_Matrix_duplicate (cj_Object *object, cj_Object *target) {
  /* This routine only duplicate the basic info. No dynamic memory allocation. */
  if (object->objtype != CJ_MATRIX || target->objtype != CJ_MATRIX) {
    cj_Object_error("Matrix_duplicate", "The object or target is not a matrix.");
  }
  cj_Matrix *base = object->matrix;
  cj_Matrix *copy = target->matrix;

  copy->eletype = base->eletype;
  copy->elelen  = base->elelen;
  copy->m       = base->m;
  copy->n       = base->n;
  copy->mb      = base->mb;
  copy->nb      = base->nb;
  copy->offm    = base->offm;
  copy->offn    = base->offn;
  copy->base    = base->base;
  //copy->buff    = base->buff;
}

void cj_Matrix_set (cj_Object *object, int m, int n) {
  if (object->objtype != CJ_MATRIX) {
    cj_Object_error("Matrix_set", "The object is not a matrix.");
  }
  if (m <= 0 && n <= 0) {
    cj_Object_error("Matrix_set", "m and n should at least be 1.");
  }
  cj_Matrix *matrix = object->matrix;
  int i, j;
  size_t elelen = matrix->elelen;

  matrix->m     = m;
  matrix->n     = n;
  matrix->mb    = (m - 1)/BLOCK_SIZE + 1;
  matrix->nb    = (n - 1)/BLOCK_SIZE + 1;
  matrix->offm  = 0;
  matrix->offn  = 0;
  matrix->base  = object->matrix;
#ifdef CJ_HAVE_CUDA
  fprintf(stderr, "cudaMallocHost()\n");
  cudaMallocHost((void**)&(matrix->buff), m*n*elelen);
#else
  matrix->buff  = (char *) malloc(m*n*elelen);
#endif

  if (!matrix->buff) {
    cj_Object_error("Matrix_set", "memory allocation failed.");
  }
  matrix->rset = (cj_Object ***) malloc((matrix->mb)*sizeof(cj_Object**));
  matrix->wset = (cj_Object ***) malloc((matrix->mb)*sizeof(cj_Object**));
  matrix->dist = (cj_Object ***) malloc((matrix->mb)*sizeof(cj_Object**));

  if (!matrix->rset || !matrix->wset || !matrix->dist) {
    cj_Object_error("Matrix_set", "memory allocation failed.");
  }

  for (i = 0; i < matrix->mb; i++) {
    matrix->rset[i] = (cj_Object **) malloc(matrix->nb*sizeof(cj_Object*));
    matrix->wset[i] = (cj_Object **) malloc(matrix->nb*sizeof(cj_Object*));
    matrix->dist[i] = (cj_Object **) malloc(matrix->nb*sizeof(cj_Object*));

    if (!matrix->rset[i] || !matrix->wset[i] || !matrix->dist[i]) {
      cj_Object_error("Matrix_set", "memory allocation failed.");
    }

    for (j = 0; j < matrix->nb; j++) {
      matrix->rset[i][j] = cj_Object_new(CJ_DQUEUE);
      matrix->wset[i][j] = cj_Object_new(CJ_DQUEUE);
      matrix->dist[i][j] = cj_Object_new(CJ_DQUEUE);

      if (!matrix->rset[i][j] || !matrix->wset[i][j] || !matrix->dist[i][j]) {
        cj_Object_error("Matrix_set", "memory allocation failed.");
      }

      /* Setup the initial distribution. (only on CPU) */
      cj_Object *distribution = cj_Object_new(CJ_DISTRIBUTION);
      if (i == matrix->mb - 1) {
        if (j == matrix->nb - 1) {
          distribution->distribution->len = (m - i*BLOCK_SIZE)*(n - j*BLOCK_SIZE)*elelen;
        }
        else {
          distribution->distribution->len = (m - i*BLOCK_SIZE)*(BLOCK_SIZE)*elelen;
        }
      }
      else {
        if (j == matrix->nb - 1) {
          distribution->distribution->len = (BLOCK_SIZE)*(n - j*BLOCK_SIZE)*elelen;
        }
        else {
          distribution->distribution->len = (BLOCK_SIZE)*(BLOCK_SIZE)*elelen;
        }
      }

      cj_Dqueue_push_tail(matrix->dist[i][j], distribution);
    }
  }
}

cj_Matrix *cj_Matrix_new () {
  cj_Matrix *matrix;
  matrix = (cj_Matrix *) malloc(sizeof(cj_Matrix));
  if (!matrix) cj_Object_error("Matrix_new", "memory allocation failed.");

  matrix->eletype = CJ_DOUBLE;
  matrix->elelen = sizeof(double);

  matrix->m = 0;
  matrix->n = 0;
  matrix->mb = 0;
  matrix->nb = 0;;
  matrix->offm = 0;
  matrix->offn = 0;
  matrix->rset = NULL;
  matrix->wset = NULL;
  matrix->dist = NULL;
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

void cj_Matrix_part_1x2 (cj_Object *A, cj_Object *A1, cj_Object *A2,
    int nb,        cj_Side side) {
  cj_Matrix *a, *a1, *a2;

  if (A->objtype != CJ_MATRIX || A1->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_part_1x2", "This is not a matrix.");
  a = A->matrix;
  a1 = A1->matrix;
  a2 = A2->matrix;
  if (nb > a->n) nb = a->n; 
  if (side == CJ_RIGHT) nb = a->n - nb;

  a1->m    = a->m;
  a1->n    = nb;
  a1->offm = a->offm;
  a1->offn = a->offn;
  a1->base = a->base;

  a2->m    = a->m;
  a2->n    = a->n - nb;
  a2->offm = a->offm;
  a2->offn = a->offn + nb;
  a2->base = a->base;
}

void cj_Matrix_part_2x2 (cj_Object *A, cj_Object *A11, cj_Object *A12,
    cj_Object *A21, cj_Object *A22,
    int mb,       int nb,         cj_Quadrant quadrant) {
  cj_Matrix *a, *a11, *a12, *a21, *a22;

  if (A->objtype != CJ_MATRIX || A11->objtype != CJ_MATRIX || A12->objtype != CJ_MATRIX
      || A21->objtype != CJ_MATRIX || A22->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_part_2x2", "This is not a matrix.");
  a   = A->matrix;
  a11 = A11->matrix;
  a12 = A12->matrix;
  a21 = A21->matrix;
  a22 = A22->matrix;

  if (mb > a->m) mb = a->m;
  if (nb > a->n) nb = a->n;

  if (quadrant == CJ_BL || quadrant == CJ_BR) mb = a->m - mb;
  if (quadrant == CJ_TR || quadrant == CJ_BR) nb = a->n - nb;

  a11->m    = mb;
  a11->n    = nb;
  a11->offm = a->offm;
  a11->offn = a->offn;
  a11->base = a->base;

  a21->m    = a->m - mb;
  a21->n    = nb;
  a21->offm = a->offm + mb;
  a21->offn = a->offn;
  a21->base = a->base;

  a12->m    = mb;
  a12->n    = a->n - nb;
  a12->offm = a->offm;
  a12->offn = a->offn + nb;
  a12->base = a->base;

  a22->m    = a->m - mb;
  a22->n    = a->n - nb;
  a22->offm = a->offm + mb;
  a22->offn = a->offn + nb;
  a22->base = a->base;
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
    cj_Matrix_part_2x1(AT, A0, A1, mb, CJ_BOTTOM);
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
    cj_Matrix_part_2x1(AB, A1, A2, mb, CJ_TOP);
  }
}

void cj_Matrix_repart_1x2_to_1x3 (cj_Object *AL,                cj_Object *AR,
    cj_Object *A0, cj_Object *A1, cj_Object *A2,
    int nb,        cj_Side side) {
  cj_Matrix *al, *ar, *a0, *a1, *a2;
  if (AL->objtype != CJ_MATRIX || AR->objtype != CJ_MATRIX || A0->objtype != CJ_MATRIX
      || A1->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_1x2_1x3", "This is not a matrix.");
  al = AL->matrix;
  ar = AR->matrix;
  a0 = A0->matrix;
  a1 = A1->matrix;
  a2 = A2->matrix;

  if (side == CJ_LEFT) {
    cj_Matrix_part_1x2(AL, A0, A1, nb, CJ_RIGHT);
    a2->m = ar->m;
    a2->n = ar->n;
    a2->offm = ar->offm;
    a2->offn = ar->offn;
    a2->base = ar->base;
  }
  else {
    a0->m = al->m;
    a0->n = al->n;
    a0->offm = al->offm;
    a0->offn = al->offn;
    a0->base = al->base;
    cj_Matrix_part_1x2(AR, A1, A2, nb, CJ_LEFT);
  }
}

void cj_Matrix_repart_2x2_to_3x3 (cj_Object *ATL, cj_Object *ATR, cj_Object *A00, cj_Object *A01, cj_Object *A02,
    cj_Object *A10, cj_Object *A11, cj_Object *A12,
    cj_Object *ABL, cj_Object *ABR, cj_Object *A20, cj_Object *A21, cj_Object *A22,
    int mb,         int nb,         cj_Quadrant quadrant) {
  cj_Matrix *atl, *atr, *a00, *a01, *a02, *a10, *a11, *a12, *abl, *abr, *a20, *a21, *a22;

  if (ATL->objtype != CJ_MATRIX || ATR->objtype != CJ_MATRIX || 
      A00->objtype != CJ_MATRIX || A01->objtype != CJ_MATRIX || A02->objtype != CJ_MATRIX ||
      A10->objtype != CJ_MATRIX || A11->objtype != CJ_MATRIX || A12->objtype != CJ_MATRIX ||
      A20->objtype != CJ_MATRIX || A21->objtype != CJ_MATRIX || A22->objtype != CJ_MATRIX ||
      ABL->objtype != CJ_MATRIX || ABR->objtype != CJ_MATRIX || 
      A20->objtype != CJ_MATRIX || A21->objtype != CJ_MATRIX || A22->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_2x2_3x3", "This is not a matrix.");

  atl = ATL->matrix; atr = ATR->matrix; a00 = A00->matrix; a01 = A01->matrix; a02 = A02->matrix;
  a10 = A10->matrix; a11 = A11->matrix; a12 = A12->matrix;
  abl = ABL->matrix; abr = ABR->matrix; a20 = A20->matrix; a21 = A21->matrix; a22 = A22->matrix;

  if (quadrant == CJ_BR) {
    cj_Matrix_part_2x2(ABR, A11, A12,
        A21, A22, mb, nb, CJ_TL);
    cj_Matrix_part_1x2(ATR, A01, A02, nb    , CJ_LEFT);
    cj_Matrix_part_2x1(ABL, A10,
        A20,      mb,     CJ_TOP);
    a00->m    = atl->m;
    a00->n    = atl->n;
    a00->offm = atl->offm;
    a00->offn = atl->offn;
    a00->base = atl->base;
  }
  else if (quadrant == CJ_BL) {
    cj_Matrix_part_2x2(ABL, A10, A11,
        A20, A21, mb, nb, CJ_TR);
    cj_Matrix_part_1x2(ATL, A00, A01, nb    , CJ_RIGHT);
    cj_Matrix_part_2x1(ABR, A12,
        A22,      mb,     CJ_TOP);
    a02->m    = atr->m;
    a02->n    = atr->n;
    a02->offm = atr->offm;
    a02->offn = atr->offn;
    a02->base = atr->base;
  }
  else if (quadrant == CJ_TL) {
    cj_Matrix_part_2x2(ATL, A00, A01,
        A10, A11, mb, nb, CJ_BR);
    cj_Matrix_part_1x2(ABL, A20, A21, nb    , CJ_RIGHT);
    cj_Matrix_part_2x1(ATR, A02,
        A12,      mb,     CJ_BOTTOM);
    a22->m    = abr->m;
    a22->n    = abr->n;
    a22->offm = abr->offm;
    a22->offn = abr->offn;
    a22->base = abr->base;
  }
  else if (quadrant == CJ_TR) {
    cj_Matrix_part_2x2(ATR, A01, A02,
        A11, A12, mb, nb, CJ_BL);
    cj_Matrix_part_1x2(ABR, A21, A22, nb    , CJ_LEFT);
    cj_Matrix_part_2x1(ATL, A00,
        A10,      mb,     CJ_BOTTOM);
    a20->m    = abl->m;
    a20->n    = abl->n;
    a20->offm = abl->offm;
    a20->offn = abl->offn;
    a20->base = abl->base;
  }

}

void cj_Matrix_cont_with_3x1_to_2x1 (cj_Object *AT, cj_Object *A0,
    cj_Object *A1,
    cj_Object *AB, cj_Object *A2,
    cj_Side side) {
  cj_Matrix *at, *a0, *a1, *ab, *a2;

  if (AT->objtype != CJ_MATRIX || A0->objtype != CJ_MATRIX || A1->objtype != CJ_MATRIX
      || AB->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_2x1_3x1", "This is not a matrix.");
  at = AT->matrix; a0 = A0->matrix;
  a1 = A1->matrix;
  ab = AB->matrix; a2 = A2->matrix;

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

void cj_Matrix_cont_with_1x3_to_1x2 (cj_Object *AL,                cj_Object *AR,
    cj_Object *A0, cj_Object *A1, cj_Object *A2,
    cj_Side side) {
  cj_Matrix *al, *ar, *a0, *a1, *a2;
  if (AL->objtype != CJ_MATRIX || AR->objtype != CJ_MATRIX || A0->objtype != CJ_MATRIX
      || A1->objtype != CJ_MATRIX || A2->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_1x2_1x3", "This is not a matrix.");
  al = AL->matrix;
  ar = AR->matrix;
  a0 = A0->matrix;
  a1 = A1->matrix;
  a2 = A2->matrix;

  if (side == CJ_LEFT) {
    al->m    = a0->m;
    al->n    = a0->n + a1->n;
    al->offm = a0->offm;
    al->offn = a0->offn;
    al->base = a0->base;

    ar->m    = a2->m;
    ar->n    = a2->n;
    ar->offm = a2->offm;
    ar->offn = a2->offn;
    ar->base = a2->base;
  }
  else {
    al->m    = a0->m;
    al->n    = a0->n;
    al->offm = a0->offm;
    al->offn = a0->offn;
    al->base = a0->base;

    ar->m    = a1->m;
    ar->n    = a1->n + a2->n;
    ar->offm = a1->offm;
    ar->offn = a1->offn;
    ar->base = a1->base;
  }
}

void cj_Matrix_cont_with_2x2_to_3x3 (cj_Object *ATL, cj_Object *ATR, cj_Object *A00, cj_Object *A01, cj_Object *A02,
    cj_Object *A10, cj_Object *A11, cj_Object *A12,
    cj_Object *ABL, cj_Object *ABR, cj_Object *A20, cj_Object *A21, cj_Object *A22,
    cj_Quadrant quadrant) {
  cj_Matrix *atl, *atr, *a00, *a01, *a02, *a10, *a11, *a12, *abl, *abr, *a20, *a21, *a22;

  if (ATL->objtype != CJ_MATRIX || ATR->objtype != CJ_MATRIX || 
      A00->objtype != CJ_MATRIX || A01->objtype != CJ_MATRIX || A02->objtype != CJ_MATRIX ||
      A10->objtype != CJ_MATRIX || A11->objtype != CJ_MATRIX || A12->objtype != CJ_MATRIX ||
      A20->objtype != CJ_MATRIX || A21->objtype != CJ_MATRIX || A22->objtype != CJ_MATRIX ||
      ABL->objtype != CJ_MATRIX || ABR->objtype != CJ_MATRIX || 
      A20->objtype != CJ_MATRIX || A21->objtype != CJ_MATRIX || A22->objtype != CJ_MATRIX) 
    cj_Object_error("matrix_repart_2x2_3x3", "This is not a matrix.");

  atl = ATL->matrix; atr = ATR->matrix; a00 = A00->matrix; a01 = A01->matrix; a02 = A02->matrix;
  a10 = A10->matrix; a11 = A11->matrix; a12 = A12->matrix;
  abl = ABL->matrix; abr = ABR->matrix; a20 = A20->matrix; a21 = A21->matrix; a22 = A22->matrix;

  if (quadrant == CJ_TL) {
    atl->m    = a00->m + a10->m;
    atl->n    = a00->n + a10->n;
    atl->offm = a00->offm;
    atl->offn = a00->offn;
    atl->base = a00->base;

    atr->m    = a02->m + a12->m;
    atr->n    = a02->n;
    atr->offm = a02->offm;
    atr->offn = a02->offn;
    atr->base = a02->base;

    abl->m    = a20->m;
    abl->n    = a20->n + a21->n;
    abl->offm = a20->offm;
    abl->offn = a20->offn;
    abl->base = a20->base;

    abr->m    = a22->m;
    abr->n    = a22->n;
    abr->offm = a22->offm;
    abr->offn = a22->offn;
    abr->base = a22->base;
  }
  else if (quadrant == CJ_TR) {
    atl->m    = a00->m + a10->m;
    atl->n    = a00->n;
    atl->offm = a00->offm;
    atl->offn = a00->offn;
    atl->base = a00->base;

    atr->m    = a01->m + a11->m;
    atr->n    = a01->n + a02->n;
    atr->offm = a01->offm;
    atr->offn = a01->offn;
    atr->base = a01->base;

    abl->m    = a20->m;
    abl->n    = a20->n;
    abl->offm = a20->offm;
    abl->offn = a20->offn;
    abl->base = a20->base;

    abr->m    = a21->m;
    abr->n    = a21->n + a22->n;
    abr->offm = a21->offm;
    abr->offn = a21->offn;
    abr->base = a21->base;
  }
  else if (quadrant == CJ_BR) {
    atl->m    = a00->m;
    atl->n    = a00->n;
    atl->offm = a00->offm;
    atl->offn = a00->offn;
    atl->base = a00->base;

    atr->m    = a01->m;
    atr->n    = a01->n + a02->n;
    atr->offm = a01->offm;
    atr->offn = a01->offn;
    atr->base = a01->base;

    abl->m    = a10->m + a20->m;
    abl->n    = a10->n;
    abl->offm = a10->offm;
    abl->offn = a10->offn;
    abl->base = a10->base;

    abr->m    = a11->m + a21->m;
    abr->n    = a11->n + a12->n;
    abr->offm = a11->offm;
    abr->offn = a11->offn;
    abr->base = a11->base;
  }
  else if (quadrant == CJ_BL) {
    atl->m    = a00->m;
    atl->n    = a00->n + a01->n;
    atl->offm = a00->offm;
    atl->offn = a00->offn;
    atl->base = a00->base;

    atr->m    = a02->m;
    atr->n    = a02->n;
    atr->offm = a02->offm;
    atr->offn = a02->offn;
    atr->base = a02->base;

    abl->m    = a10->m + a20->m;
    abl->n    = a10->n + a11->n;
    abl->offm = a10->offm;
    abl->offn = a10->offn;
    abl->base = a10->base;

    abr->m    = a12->m + a22->m;
    abr->n    = a12->n;
    abr->offm = a12->offm;
    abr->offn = a12->offn;
    abr->base = a12->base;
  }

}

void cj_Matrix_delete (cj_Matrix *matrix) {

}

void cj_Matrix_set_identity (cj_Object *object) {
  if (object->objtype != CJ_MATRIX) cj_Object_error("Matrix_set", "The object is not a matrix.");
  if (!object->matrix) cj_Object_error("Matrix_set_identity", "The matrix hasn't been initialized yet.");
  cj_Matrix *matrix = object->matrix;
  if (matrix->m != matrix->n) cj_Object_error("Matrix_set_identity", "The matrix is not a sqaure matrix.");

  int i, j;
  
  for (j = 0; j < matrix->n; j++) {
    for (i = 0; i < matrix->m; i++) {
      if (matrix->eletype == CJ_SINGLE) {
        ((float *) (matrix->buff))[i + j*matrix->m] = 0.0;
        if (i == j) ((float *) (matrix->buff))[i + j*matrix->m] = 1.0;
      }
      else {
        ((double *) (matrix->buff))[i + j*matrix->m] = 0.0;
        if (i == j) ((double *) (matrix->buff))[i + j*matrix->m] = 1.0;
      }
    }
  }

}

void cj_Matrix_print (cj_Object *object) {
  if (object->objtype != CJ_MATRIX) cj_Object_error("Matrix_set", "The object is not a matrix.");
  if (!object->matrix) cj_Object_error("Matrix_set_identity", "The matrix hasn't been initialized yet.");

  cj_Matrix *matrix = object->matrix;
  int i, j;

  if (matrix->n <= 32 && matrix->m <= 32) {
    fprintf(stderr, "     ");
    fprintf(stderr, "     ");
    for (j = 0; j < matrix->n; j++) fprintf(stderr, "%4d ", j);
    fprintf(stderr, "\n");
    for (i = 0; i < matrix->m; i++) {
      fprintf(stderr, "%4d ", i);
      fprintf(stderr, "     ");
      for (j = 0; j < matrix->n; j++) {
        if (matrix->eletype == CJ_SINGLE) {
          fprintf(stderr, "%4.2f ", ((float *) (matrix->buff))[i + j*matrix->m]);
        }
        else {
          fprintf(stderr, "%4.2lf ", ((double *) (matrix->buff))[i + j*matrix->m]);
        }
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  } 
}

void cj_Matrix_distribution_print (cj_Object *object) {
  if (object->objtype != CJ_MATRIX) cj_Object_error("Matrix_set", "The object is not a matrix.");
  if (!object->matrix) cj_Object_error("Matrix_set_identity", "The matrix hasn't been initialized yet.");

  cj_Matrix *matrix = object->matrix;
  cj_Object ***dist = matrix->dist;
  int i, j, k;

  if (matrix->nb <= 4 && matrix->mb <= 4) {
    fprintf(stderr, "     ");
    fprintf(stderr, "     ");
    for (j = 0; j < matrix->nb; j++) fprintf(stderr, "%9d ", j);
    fprintf(stderr, "\n");
    for (i = 0; i < matrix->mb; i++) {
      fprintf(stderr, "%4d ", i);
      fprintf(stderr, "     ");
      for (j = 0; j < matrix->nb; j++) {
        cj_Object *now = dist[i][j]->dqueue->head;
        int remain = 5;
        while (now) {
          fprintf(stderr, "%1d,", now->distribution->device_id);
          now = now->next;
          remain --;
        }
        for (k = 0; k < remain; k++) fprintf(stderr, "  ");
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

void cj_Object_acquire (cj_Object *object) {
  if (object->objtype == CJ_MATRIX) {
    cj_Matrix *matrix = object->matrix;
    cj_Matrix *base = matrix->base;
    cj_Object *view_obj = cj_Object_new(CJ_MATRIX);
    cj_Matrix *view = view_obj->matrix;
    /* Set the base for this matrix view. */
    view->base = matrix->base;

    cj_Object *dist;
    /* Check the distribution. */
    int i, j;
    for (i = 0; i < matrix->m/BLOCK_SIZE; i ++) {
      for (j = 0; j < matrix->n/BLOCK_SIZE; j ++) {
        dist = base->dist[matrix->offm/BLOCK_SIZE + i][matrix->offn/BLOCK_SIZE + j];
        cj_Object *dist_cpu = NULL;
        cj_Object *dist_I = dist->dqueue->head;

        view->offm = i*BLOCK_SIZE;
        view->offn = j*BLOCK_SIZE;
        view->m = min(BLOCK_SIZE, matrix->m - i*BLOCK_SIZE);
        view->n = min(BLOCK_SIZE, matrix->n - j*BLOCK_SIZE);

        //fprintf(stderr, "offm : %d, offn : %d\n", view->offm, view->offn);

        while (dist_I) {
          if (dist_I->distribution->device_id == -1) {
            dist_cpu = dist_I;
            break;
          }
          dist_I = dist_I->next;
        }
        if (!dist_cpu) {
          dist_I = dist->dqueue->head;
          cj_Device *device = dist_I->distribution->device;
          int cache_id = dist_I->distribution->cache_id;
          //fprintf(stderr, "device_type: %d, %s\n", device->devtype, device->name);
          cj_Cache_write_back(device, cache_id, view_obj);
        }
      }
    }
    /* Need to free "view" here. */
  }
}

cj_Object *cj_Object_append (cj_objType type, void *ptr) {
  cj_Object *object;
  object = (cj_Object*) malloc(sizeof(cj_Object));
  if (!object) cj_Object_error("new", "memory allocation failed.");
  if (type == CJ_MATRIX) {
    object->objtype = CJ_MATRIX;
    object->matrix = (cj_Matrix *) ptr;
  }
  else if (type == CJ_DQUEUE) {
    object->objtype = CJ_DQUEUE;
    object->dqueue = (cj_Dqueue *) ptr;
  }
  else if (type == CJ_TASK) {
    object->objtype = CJ_TASK;
    object->task = (cj_Task *) ptr;
    /* Need to call the specific constructor. */
  }
  else if (type == CJ_VERTEX) {
    object->objtype = CJ_VERTEX;
    object->vertex = (cj_Vertex *) ptr;
  }
  else if (type == CJ_EDGE) {
    object->objtype = CJ_EDGE;
    object->edge = (cj_Edge *) ptr;
  }
  else if (type == CJ_DISTRIBUTION) {
    object->objtype = CJ_DISTRIBUTION;
    object->distribution = (cj_Distribution *) ptr;
  }

  object->prev = NULL;
  object->next = NULL;

  return object;
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
    object->task = cj_Task_new();
    /* Need to call the specific constructor. */
  }
  else if (type == CJ_VERTEX) {
    object->objtype = CJ_VERTEX;
    object->vertex = cj_Vertex_new();
  }
  else if (type == CJ_EDGE) {
    object->objtype = CJ_EDGE;
    object->edge = cj_Edge_new();
  }
  else if (type == CJ_DISTRIBUTION) {
    object->objtype = CJ_DISTRIBUTION;
    object->distribution = cj_Distribution_new();
  }

  object->prev = NULL;
  object->next = NULL;

  return object;
}
