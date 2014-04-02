#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cj_Macro.h"
/*
#include "cj_Device.h"
#include "cj_Object.h"
#include "cj_Graph.h"
#include "cj.h"
*/
#include <CJ.h>
#include "cj_Blas.h"
#include "cj_Graph.h"
#include "cj_Object.h"

#include <cublas.h>

void cj_Blas_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_BLAS_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Gemm_nn_task_function (void *task_ptr) {
  cj_Task *task = (cj_Task *) task_ptr;
  //cj_Cache = task->worker->cj->device
  cj_Object *A, *B, *C;
  cj_Matrix *a, *b, *c;
  A = task->arg_in->dqueue->head;
  B = A->next;
  C = B->next;
  if (C->next != NULL) fprintf(stderr, "arg_in has been tanted.");
  a = A->matrix;
  b = B->matrix;
  c = C->matrix;

  if (task->worker->device_id != -1 && task->worker->devtype == CJ_DEV_CUDA) {
    /*
    cj_Object *a_now = a->dist->dqueue->head;
    while (a_now->distribution-device_id != task->worker->device_id) a_now = a_now->next;
     */
  }
  else {
    if (a->eletype == CJ_SINGLE) {
      float f_one = 1.0;
      float *a_buff = (float *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;
      float *b_buff = (float *) (b->base->buff) + (b->base->m)*(b->offn) + b->offm;
      float *c_buff = (float *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
      sgemm_("N", "N", &(c->m), &(a->n), &(c->n), &f_one, a_buff, &(a->base->m), b_buff, &(b->base->m), &f_one, c_buff, &(c->base->m));
    }
    else {
      double f_one = 1.0;
      double *a_buff = (double *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;
      double *b_buff = (double *) (b->base->buff) + (b->base->m)*(b->offn) + b->offm;
      double *c_buff = (double *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
      dgemm_("N", "N", &(c->m), &(a->n), &(c->n), &f_one, a_buff, &(a->base->m), b_buff, &(b->base->m), &f_one, c_buff, &(c->base->m));
    }
  }

  fprintf(stderr, YELLOW "  Worker_execute %d (%d, %s), A(%d, %d), B(%d, %d), C(%d, %d): \n" NONE, 
      task->worker->id, task->id, task->name,
      a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE, 
      b->offm/BLOCK_SIZE, b->offn/BLOCK_SIZE,  
      c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE);  
}

void cj_Gemm_nn_task(cj_Object *alpha, cj_Object *A, cj_Object *B, cj_Object *beta, cj_Object *C) {
  /* Gemm will read A, B, C and write C. */
  cj_Object *A_copy, *B_copy, *C_copy;
  cj_Matrix *a, *b, *c;
  cj_Object *A_r, *B_r, *C_r, *A_w, *B_w, *C_w;
  cj_Object *task, *vertex;
  
  /* Generate copies of A, B and C. */
  A_copy = cj_Object_new(CJ_MATRIX);
  B_copy = cj_Object_new(CJ_MATRIX);
  C_copy = cj_Object_new(CJ_MATRIX);
  cj_Matrix_duplicate(A, A_copy);
  cj_Matrix_duplicate(B, B_copy);
  cj_Matrix_duplicate(C, C_copy);

  a = A_copy->matrix; b = B_copy->matrix; c = C_copy->matrix;

     fprintf(stderr, "          a->base : %d\n", (int)a->base);
     fprintf(stderr, "          b->base : %d\n", (int)b->base);
     fprintf(stderr, "          c->base : %d\n", (int)c->base);
     fprintf(stderr, "          a->offm : %d, a->offn : %d\n", a->offm, a->offn);
     fprintf(stderr, "          b->offm : %d, b->offn : %d\n", b->offm, b->offn);
     fprintf(stderr, "          c->offm : %d, c->offn : %d\n", c->offm, c->offn);
     
  A_r = a->base->rset[a->offm/BLOCK_SIZE][a->offn/BLOCK_SIZE];
  B_r = b->base->rset[b->offm/BLOCK_SIZE][b->offn/BLOCK_SIZE];
  C_r = c->base->rset[c->offm/BLOCK_SIZE][c->offn/BLOCK_SIZE];

  A_w = a->base->wset[a->offm/BLOCK_SIZE][a->offn/BLOCK_SIZE];
  B_w = b->base->wset[b->offm/BLOCK_SIZE][b->offn/BLOCK_SIZE];
  C_w = c->base->wset[c->offm/BLOCK_SIZE][c->offn/BLOCK_SIZE];

  task = cj_Object_new(CJ_TASK);
  cj_Task_set(task->task, &cj_Gemm_nn_task_function);

  /* Pushing input arguments. */
  cj_Dqueue_push_tail(task->task->arg_in, cj_Object_append(CJ_MATRIX, a));
  cj_Dqueue_push_tail(task->task->arg_in, cj_Object_append(CJ_MATRIX, b));
  cj_Dqueue_push_tail(task->task->arg_in, cj_Object_append(CJ_MATRIX, c));

  /* Exam the arguments. */



  /* Pushing output arguments. */
  cj_Dqueue_push_tail(task->task->arg_out, cj_Object_append(CJ_MATRIX, c));

  snprintf(task->task->name, 64, "Gemm_nn%d_A_%d_%d_B_%d_%d_C_%d_%d", 
      task->task->id,
      a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE,
      b->offm/BLOCK_SIZE, b->offn/BLOCK_SIZE,
      c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE );

  /* TODO : cj_Task_set() */
  vertex = cj_Object_new(CJ_VERTEX);
  cj_Vertex_set(vertex, task);
  cj_Graph_vertex_add(vertex);

  /*
     fprintf(stderr, "          A_r->objtype is CJ_DQUEUE = %d\n", A_r->objtype == CJ_DQUEUE);
     fprintf(stderr, "          B_r->objtype is CJ_DQUEUE = %d\n", B_r->objtype == CJ_DQUEUE);
     fprintf(stderr, "          C_r->objtype is CJ_DQUEUE = %d\n", C_r->objtype == CJ_DQUEUE);
     */

  cj_Dqueue_push_tail(A_r, cj_Object_append(CJ_TASK, (void *) task->task));

  if (cj_Dqueue_get_size(A_w) > 0) {
    cj_Object *now = A_w->dqueue->head;
    while (now) {
      if (now->task != task->task) {
        cj_Object *edge;
        edge = cj_Object_new(CJ_EDGE);
        cj_Edge_set(edge, now, task);
        cj_Graph_edge_add(edge);
        cj_Task_dependency_add(now, task);
        fprintf(stderr, "          %d->%d.\n", now->task->id, task->task->id);
      }
      now = now->next;
    }
  }

  cj_Dqueue_push_tail(B_r, cj_Object_append(CJ_TASK, (void *) task->task));
  if (cj_Dqueue_get_size(B_w) > 0) {
    cj_Object *now = B_w->dqueue->head;
    while (now) {
      if (now->task != task->task) {
        cj_Object *edge;
        edge = cj_Object_new(CJ_EDGE);
        cj_Edge_set(edge, now, task);
        cj_Graph_edge_add(edge);
        cj_Task_dependency_add(now, task);
        fprintf(stderr, "          %d->%d.\n", now->task->id, task->task->id);
      }
      now = now->next;
    }
  }

  cj_Dqueue_push_tail(C_r, cj_Object_append(CJ_TASK, (void *) task->task));

  if (cj_Dqueue_get_size(C_w) > 0) {
    cj_Object *now = C_w->dqueue->head;
    while (now) {
      if (now->task != task->task) {
        cj_Object *edge;
        edge = cj_Object_new(CJ_EDGE);
        cj_Edge_set(edge, now, task);
        cj_Graph_edge_add(edge);
        cj_Task_dependency_add(now, task);
        fprintf(stderr, "          %d->%d.\n", now->task->id, task->task->id);
      }
      now = now->next;
    }
  }

  if (cj_Dqueue_get_size(C_r) > 0) {
    cj_Object *now = C_r->dqueue->head; 
    while (now) {
      if (now->task->id != task->task->id) {
        cj_Object *edge;
        edge = cj_Object_new(CJ_EDGE);
        cj_Edge_set(edge, now, task);
        cj_Graph_edge_add(edge);
        cj_Task_dependency_add(now, task);
        fprintf(stderr, "          %d->%d. Anti-dependency.\n", now->task->id, now->task->id);
      }
      now = now->next;
    }
  }
  cj_Dqueue_clear(C_w);
  cj_Dqueue_push_tail(C_w, cj_Object_append(CJ_TASK, (void *) task->task));
  cj_Dqueue_clear(C_r);




}

void cj_Gebp_nn(cj_Object *A, cj_Object *B, cj_Object *C) {
  fprintf(stderr, RED "        Gebp_nn (A(%d, %d), B(%d, %d), C(%d, %d)): \n" NONE, 
      A->matrix->m, A->matrix->n,
      B->matrix->m, B->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "        {\n");
  cj_Object *alpha, *beta;
  alpha = cj_Object_new(CJ_CONSTANT);
  beta  = cj_Object_new(CJ_CONSTANT);
  /* TODO : Constant constructor */
  cj_Gemm_nn_task(alpha, A, B, beta, C);
  fprintf(stderr, "        }\n");
}

void cj_Gepp_nn_blk_var1(cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *AT, *A0, *AB, *A1, *A2, *CT, *C0, *CB, *C1, *C2;

  fprintf(stderr, "      Gepp_nn_blk_var1 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      B->matrix->m, B->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "      {\n");

  AT = cj_Object_new(CJ_MATRIX); A0 = cj_Object_new(CJ_MATRIX);
  AB = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX);
  A2 = cj_Object_new(CJ_MATRIX);

  CT = cj_Object_new(CJ_MATRIX); C0 = cj_Object_new(CJ_MATRIX);
  CB = cj_Object_new(CJ_MATRIX); C1 = cj_Object_new(CJ_MATRIX);
  C2 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_2x1(A,     AT,
      AB,     0,    CJ_TOP);
  cj_Matrix_part_2x1(C,     CT,
      CB,     0,    CJ_TOP);

  while (AT->matrix->m < A->matrix->m) {
    b = min(AB->matrix->m, BLOCK_SIZE);

    cj_Matrix_repart_2x1_to_3x1(AT,           A0,
        /* ** */      /* ** */
        A1,
        AB,           A2,      b,  CJ_BOTTOM);
    cj_Matrix_repart_2x1_to_3x1(CT,           C0,
        /* ** */      /* ** */
        C1,
        CB,           C2,      b,  CJ_BOTTOM);

    /* ------------------------------------------------------------------ */
    cj_Gebp_nn (A1, B, C1);
    /* ------------------------------------------------------------------ */

    cj_Matrix_cont_with_3x1_to_2x1(AT,           A0,
        A1,
        /* ** */      /* ** */
        AB,           A2,       CJ_TOP);
    cj_Matrix_cont_with_3x1_to_2x1(CT,           C0,
        C1,
        /* ** */      /* ** */
        CB,           C2,       CJ_TOP);
  }

  fprintf(stderr, "      }\n");
}

void cj_Gemm_nn_blk_var1 (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *AL, *AR, *A0, *A1, *A2, *BT, *B0, *BB, *B1, *B2;

  fprintf(stderr, "    Gemm_nn_blk_var1 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      B->matrix->m, B->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "    {\n");

  AL = cj_Object_new(CJ_MATRIX); AR = cj_Object_new(CJ_MATRIX);
  A0 = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX); A2 = cj_Object_new(CJ_MATRIX);

  BT = cj_Object_new(CJ_MATRIX); B0 = cj_Object_new(CJ_MATRIX);
  BB = cj_Object_new(CJ_MATRIX); B1 = cj_Object_new(CJ_MATRIX);
  B2 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_1x2(A, AL, AR, 0, CJ_LEFT);
  cj_Matrix_part_2x1(B, BT,
      BB,     0, CJ_TOP);
  while (AL->matrix->n < A->matrix->n) {
    b = min(AR->matrix->n, BLOCK_SIZE);

    cj_Matrix_repart_1x2_to_1x3(AL, /**/ AR,       A0, /**/ A1, A2,
        b, CJ_RIGHT);
    cj_Matrix_repart_2x1_to_3x1(BT,      B0,
        /* ** */ /* ** */
        B1,
        BB,      B2,       b, CJ_BOTTOM);

    /* ------------------------------------------------------------------ */
    //cj_Gepp_nn_blk_var1 (A1, B1, C);
    cj_Object *alpha, *beta;
    alpha = cj_Object_new(CJ_CONSTANT);
    beta  = cj_Object_new(CJ_CONSTANT);
    cj_Gemm_nn_task(alpha, A1, B1, beta, C);
    /* ------------------------------------------------------------------ */

    cj_Matrix_cont_with_1x3_to_1x2(AL, /**/ AR,       A0, A1, /**/ A2,
        CJ_LEFT);
    cj_Matrix_cont_with_3x1_to_2x1(BT,      B0,
        B1,
        /* ** */ /* ** */
        BB,      B2,       CJ_TOP);
  }
  fprintf(stderr, "    }\n");
}

void cj_Gemm_nn_blk_var3 (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *BL, *BR, *B0, *B1, *B2, *CL, *CR, *C0, *C1, *C2;

  fprintf(stderr, "  Gemm_nn_blk_var3 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      B->matrix->m, B->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "  {\n");

  BL = cj_Object_new(CJ_MATRIX); BR = cj_Object_new(CJ_MATRIX);
  B0 = cj_Object_new(CJ_MATRIX); B1 = cj_Object_new(CJ_MATRIX); B2 = cj_Object_new(CJ_MATRIX);

  CL = cj_Object_new(CJ_MATRIX); CR = cj_Object_new(CJ_MATRIX);
  C0 = cj_Object_new(CJ_MATRIX); C1 = cj_Object_new(CJ_MATRIX); C2 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_1x2(B, BL, BR, 0, CJ_LEFT);
  cj_Matrix_part_1x2(C, CL, CR, 0, CJ_LEFT);

  while (BL->matrix->n < B->matrix->n) {
    b = min(BR->matrix->n, BLOCK_SIZE);

    cj_Matrix_repart_1x2_to_1x3(BL, /**/ BR,     B0, /**/ B1, B2, 
        b, CJ_RIGHT);
    cj_Matrix_repart_1x2_to_1x3(CL, /**/ CR,     C0, /**/ C1, C2, 
        b, CJ_RIGHT);

    /* ------------------------------------------------------------------ */
    cj_Gemm_nn_blk_var1 (A, B1, C1);
    /* ------------------------------------------------------------------ */

    cj_Matrix_cont_with_1x3_to_1x2(BL, /**/ BR,     B0, /**/ B1, B2, 
        CJ_LEFT);
    cj_Matrix_cont_with_1x3_to_1x2(CL, /**/ CR,     C0, /**/ C1, C2, 
        CJ_LEFT);

  }
  fprintf(stderr, "  }\n");
}

void cj_Gemm_nn_blk_var5 (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *AT, *A0, *AB, *A1, *A2, *CT, *C0, *CB, *C1, *C2;

  fprintf(stderr, "Gemm_nn_blk_var5 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      B->matrix->m, B->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "{\n");

  AT = cj_Object_new(CJ_MATRIX); A0 = cj_Object_new(CJ_MATRIX);
  AB = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX);
  A2 = cj_Object_new(CJ_MATRIX);

  CT = cj_Object_new(CJ_MATRIX); C0 = cj_Object_new(CJ_MATRIX);
  CB = cj_Object_new(CJ_MATRIX); C1 = cj_Object_new(CJ_MATRIX);
  C2 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_2x1(A,     AT,
      AB,     0,    CJ_TOP);
  cj_Matrix_part_2x1(C,     CT,
      CB,     0,    CJ_TOP);

  while (AT->matrix->m < A->matrix->m) {
    b = min(AB->matrix->m, BLOCK_SIZE);

    cj_Matrix_repart_2x1_to_3x1(AT,           A0,
        /* ** */      /* ** */
        A1,
        AB,           A2,      b,  CJ_BOTTOM);
    cj_Matrix_repart_2x1_to_3x1(CT,           C0,
        /* ** */      /* ** */
        C1,
        CB,           C2,      b,  CJ_BOTTOM);

    /* ------------------------------------------------------------------ */
    cj_Gemm_nn_blk_var3 (A1, B, C1);
    /* ------------------------------------------------------------------ */

    cj_Matrix_cont_with_3x1_to_2x1(AT,           A0,
        A1,
        /* ** */      /* ** */
        AB,           A2,       CJ_TOP);
    cj_Matrix_cont_with_3x1_to_2x1(CT,           C0,
        C1,
        /* ** */      /* ** */
        CB,           C2,       CJ_TOP);
  }
  fprintf(stderr, "}\n");
}

void cj_Syrk_ln_blk_var5 (cj_Object *A, cj_Object *C) {

}

void cj_Gemm_nn (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Matrix *a, *b, *c;
  if (!A || !B || !C) 
    cj_Blas_error("gemm_nn", "matrices haven't been initialized yet.");
  if (!A->matrix || !B->matrix || !C->matrix)
    cj_Blas_error("gemm_nn", "Object types are not matrix type.");
  a = A->matrix;
  b = B->matrix;
  c = C->matrix;
  if ((c->m != a->m) || (c->n != b->n) || (a->n != b->m)) 
    cj_Blas_error("gemm_nn", "matrices dimension aren't matched.");

  cj_Queue_end();
  cj_Gemm_nn_blk_var5(A, B, C);
  cj_Queue_begin();
}

void cj_Syrk_ln (cj_Object *A, cj_Object *C) {
  cj_Matrix *a, *c;
  if (!A || !C) 
    cj_Blas_error("syrk_ln", "matrices haven't been initialized yet.");
  if (!A->matrix || !C->matrix)
    cj_Blas_error("syrk_ln", "Object types are not matrix type.");
  a = A->matrix;
  c = C->matrix;
  if ((a->m != a->n) || (c->m != c->n) || (c->m != a->m)) 
    cj_Blas_error("syrk_ln", "matrices dimension aren't matched.");

  cj_Syrk_ln_blk_var5(A, C);
}
