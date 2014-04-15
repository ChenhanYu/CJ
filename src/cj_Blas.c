#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cj_Macro.h"
#include <CJ.h>
#include "cj.h"
#include "cj_Blas.h"
#include "cj_Graph.h"
#include "cj_Object.h"

#ifdef CJ_HAVE_CUDA
#include <cublas_v2.h>
#endif

void cj_Blas_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_BLAS_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Gemm_nn_task_function (void *task_ptr) {
  cj_Task *task = (cj_Task *) task_ptr;
  cj_Worker *worker = task->worker;
  cj_devType devtype = worker->devtype;
  int device_id = worker->device_id;

  cj_Object *A, *B, *C;
  cj_Matrix *a, *b, *c;
  A = task->arg->dqueue->head;
  B = A->next;
  C = B->next;
  a = A->matrix;
  b = B->matrix;
  c = C->matrix;

  if (device_id != -1 && devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaSetDevice(device_id);
    cj_Device *device = worker->cj_ptr->device[device_id];
    cj_Cache *cache = &(device->cache);
    cublasHandle_t *handle = &(device->handle);
    cublasStatus_t status;
    cj_Object *now_a = a->base->dist[a->offm/BLOCK_SIZE][a->offn/BLOCK_SIZE]->dqueue->head;
    cj_Object *now_b = b->base->dist[b->offm/BLOCK_SIZE][b->offn/BLOCK_SIZE]->dqueue->head;
    cj_Object *now_c = c->base->dist[c->offm/BLOCK_SIZE][c->offn/BLOCK_SIZE]->dqueue->head;
    while (now_a->distribution->device_id != device_id && now_a != NULL) now_a = now_a->next;
    while (now_b->distribution->device_id != device_id && now_b != NULL) now_b = now_b->next;
    while (now_c->distribution->device_id != device_id && now_c != NULL) now_c = now_c->next;
    if (!now_a || !now_b || !now_c) 
      cj_Blas_error("cj_Gemm_nn_task_function", "No propriate distribution.");

    if (a->eletype == CJ_SINGLE) { 
      float f_one = 1.0;
      float *a_buff = (float *) cache->dev_ptr[now_a->distribution->cache_id];
      float *b_buff = (float *) cache->dev_ptr[now_b->distribution->cache_id];
      float *c_buff = (float *) cache->dev_ptr[now_c->distribution->cache_id];
      status = cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, c->m, a->n, c->n, &f_one, a_buff, BLOCK_SIZE, 
          b_buff, BLOCK_SIZE, &f_one, c_buff, BLOCK_SIZE);
    }
    else {
      double f_one = 1.0;
      double *a_buff = (double *) cache->dev_ptr[now_a->distribution->cache_id];
      double *b_buff = (double *) cache->dev_ptr[now_b->distribution->cache_id];
      double *c_buff = (double *) cache->dev_ptr[now_c->distribution->cache_id];
      //fprintf(stderr, "%d, %d, %d, %d\n", c->m, a->n, c->n, BLOCK_SIZE);
      //fprintf(stderr, "%d, %d, %d\n", (int) a_buff, (int) b_buff, (int) c_buff);
      status = cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, c->m, a->n, c->n, &f_one, a_buff, BLOCK_SIZE, 
          b_buff, BLOCK_SIZE, &f_one, c_buff, BLOCK_SIZE);
    }
    if (status != CUBLAS_STATUS_SUCCESS) cj_Blas_error("cj_Gemm_nn_task_function", "cublas failure");
#endif
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
  cj_Object *A_copy, *B_copy, *C_copy, *task;
  cj_Matrix *a, *b, *c;
  
  /* Generate copies of A, B and C. */
  A_copy = cj_Object_new(CJ_MATRIX);
  B_copy = cj_Object_new(CJ_MATRIX);
  C_copy = cj_Object_new(CJ_MATRIX);
  cj_Matrix_duplicate(A, A_copy);
  cj_Matrix_duplicate(B, B_copy);
  cj_Matrix_duplicate(C, C_copy);

  a = A_copy->matrix; b = B_copy->matrix; c = C_copy->matrix;
  /*
     fprintf(stderr, "          a->base : %d\n", (int)a->base);
     fprintf(stderr, "          b->base : %d\n", (int)b->base);
     fprintf(stderr, "          c->base : %d\n", (int)c->base);
     fprintf(stderr, "          a->offm : %d, a->offn : %d\n", a->offm, a->offn);
     fprintf(stderr, "          b->offm : %d, b->offn : %d\n", b->offm, b->offn);
     fprintf(stderr, "          c->offm : %d, c->offn : %d\n", c->offm, c->offn);
  */   

  task = cj_Object_new(CJ_TASK);
  cj_Task_set(task->task, &cj_Gemm_nn_task_function);

  /* Pushing arguments. */
  cj_Object *arg_A = cj_Object_append(CJ_MATRIX, a);
  cj_Object *arg_B = cj_Object_append(CJ_MATRIX, b);
  cj_Object *arg_C = cj_Object_append(CJ_MATRIX, c);
  arg_A->rwtype = CJ_R;
  arg_B->rwtype = CJ_R;
  arg_C->rwtype = CJ_RW;
  cj_Dqueue_push_tail(task->task->arg, arg_A);
  cj_Dqueue_push_tail(task->task->arg, arg_B);
  cj_Dqueue_push_tail(task->task->arg, arg_C);

  /* Setup task name. */
  snprintf(task->task->name, 64, "Gemm_nn%d_A_%d_%d_B_%d_%d_C_%d_%d", 
      task->task->id,
      a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE,
      b->offm/BLOCK_SIZE, b->offn/BLOCK_SIZE,
      c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE );

  cj_Task_dependency_analysis(task);
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

void cj_Gemm_nt_task_function (void *task_ptr) {
	cj_Task *task = (cj_Task *) task_ptr;
	cj_Worker *worker = task->worker;
	cj_devType devtype = worker->devtype;
	int device_id = worker->device_id;
	
	cj_Object *A, *B, *C;
	cj_Matrix *a, *b, *c;
	A = task->arg->dqueue->head;
	B = A->next;
	C = B->next;
	a = A->matrix;
	b = B->matrix;
	c = C->matrix;
	
	if (device_id != -1 && devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
		cudaSetDevice(device_id);
		cj_Device *device = worker->cj_ptr->device[device_id];
		cj_Cache *cache = &(device->cache);
		cublasHandle_t *handle = &(device->handle);
		cublasStatus_t status;
		cj_Object *now_a = a->base->dist[a->offm/BLOCK_SIZE][a->offn/BLOCK_SIZE]->dqueue->head;
		cj_Object *now_b = b->base->dist[b->offm/BLOCK_SIZE][b->offn/BLOCK_SIZE]->dqueue->head;
		cj_Object *now_c = c->base->dist[c->offm/BLOCK_SIZE][c->offn/BLOCK_SIZE]->dqueue->head;
		while (now_a->distribution->device_id != device_id && now_a != NULL) now_a = now_a->next;
		while (now_b->distribution->device_id != device_id && now_b != NULL) now_b = now_b->next;
		while (now_c->distribution->device_id != device_id && now_c != NULL) now_c = now_c->next;
		if (!now_a || !now_b || !now_c) 
			cj_Blas_error("cj_Gemm_nt_task_function", "No propriate distribution.");
		
		if (a->eletype == CJ_SINGLE) { 
			float f_one = 1.0;
			float *a_buff = (float *) cache->dev_ptr[now_a->distribution->cache_id];
			float *b_buff = (float *) cache->dev_ptr[now_b->distribution->cache_id];
			float *c_buff = (float *) cache->dev_ptr[now_c->distribution->cache_id];
			status = cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, c->m, a->n, c->n, &f_one, a_buff, BLOCK_SIZE, 
								 b_buff, BLOCK_SIZE, &f_one, c_buff, BLOCK_SIZE);
		}
		else {
			double f_one = 1.0;
			double *a_buff = (double *) cache->dev_ptr[now_a->distribution->cache_id];
			double *b_buff = (double *) cache->dev_ptr[now_b->distribution->cache_id];
			double *c_buff = (double *) cache->dev_ptr[now_c->distribution->cache_id];
			//fprintf(stderr, "%d, %d, %d, %d\n", c->m, a->n, c->n, BLOCK_SIZE);
			//fprintf(stderr, "%d, %d, %d\n", (int) a_buff, (int) b_buff, (int) c_buff);
			status = cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, c->m, a->n, c->n, &f_one, a_buff, BLOCK_SIZE, 
								 b_buff, BLOCK_SIZE, &f_one, c_buff, BLOCK_SIZE);
		}
		if (status != CUBLAS_STATUS_SUCCESS) cj_Blas_error("cj_Gemm_nt_task_function", "cublas failure");
#endif
	}
	else {
		if (a->eletype == CJ_SINGLE) {
			float f_one = 1.0;
			float *a_buff = (float *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;
			//DO we have proble here???B is a matrix to be transformed...
			float *b_buff = (float *) (b->base->buff) + (b->base->m)*(b->offn) + b->offm;
			float *c_buff = (float *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
			sgemm_("N", "T", &(c->m), &(a->n), &(c->n), &f_one, a_buff, &(a->base->m), b_buff, &(b->base->m), &f_one, c_buff, &(c->base->m));
		}
		else {
			double f_one = 1.0;
			double *a_buff = (double *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;
			double *b_buff = (double *) (b->base->buff) + (b->base->m)*(b->offn) + b->offm;
			double *c_buff = (double *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
			dgemm_("N", "T", &(c->m), &(a->n), &(c->n), &f_one, a_buff, &(a->base->m), b_buff, &(b->base->m), &f_one, c_buff, &(c->base->m));
		}
	}
	
	fprintf(stderr, YELLOW "  Worker_execute %d (%d, %s), A(%d, %d), B(%d, %d), C(%d, %d): \n" NONE, 
			task->worker->id, task->id, task->name,
			a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE, 
			b->offm/BLOCK_SIZE, b->offn/BLOCK_SIZE,  
			c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE);  
}

void cj_Gemm_nt_task(cj_Object *alpha, cj_Object *A, cj_Object *B, cj_Object *beta, cj_Object *C) {
	/* Gemm will read A, B, C and write C. */
	cj_Object *A_copy, *B_copy, *C_copy, *task;
	cj_Matrix *a, *b, *c;
	
	/* Generate copies of A, B and C. */
	A_copy = cj_Object_new(CJ_MATRIX);
	B_copy = cj_Object_new(CJ_MATRIX);
	C_copy = cj_Object_new(CJ_MATRIX);
	cj_Matrix_duplicate(A, A_copy);
	cj_Matrix_duplicate(B, B_copy);
	cj_Matrix_duplicate(C, C_copy);
	
	a = A_copy->matrix; b = B_copy->matrix; c = C_copy->matrix;
	/*
     fprintf(stderr, "          a->base : %d\n", (int)a->base);
     fprintf(stderr, "          b->base : %d\n", (int)b->base);
     fprintf(stderr, "          c->base : %d\n", (int)c->base);
     fprintf(stderr, "          a->offm : %d, a->offn : %d\n", a->offm, a->offn);
     fprintf(stderr, "          b->offm : %d, b->offn : %d\n", b->offm, b->offn);
     fprintf(stderr, "          c->offm : %d, c->offn : %d\n", c->offm, c->offn);
	 */   
	
	task = cj_Object_new(CJ_TASK);
	cj_Task_set(task->task, &cj_Gemm_nt_task_function);
	
	/* Pushing arguments. */
	cj_Object *arg_A = cj_Object_append(CJ_MATRIX, a);
	cj_Object *arg_B = cj_Object_append(CJ_MATRIX, b);
	cj_Object *arg_C = cj_Object_append(CJ_MATRIX, c);
	arg_A->rwtype = CJ_R;
	arg_B->rwtype = CJ_R;
	arg_C->rwtype = CJ_RW;
	cj_Dqueue_push_tail(task->task->arg, arg_A);
	cj_Dqueue_push_tail(task->task->arg, arg_B);
	cj_Dqueue_push_tail(task->task->arg, arg_C);
	
	/* Setup task name. */
	snprintf(task->task->name, 64, "Gemm_nt%d_A_%d_%d_B_%d_%d_C_%d_%d", 
			 task->task->id,
			 a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE,
			 b->offm/BLOCK_SIZE, b->offn/BLOCK_SIZE,
			 c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE );
	
	cj_Task_dependency_analysis(task);
}

void cj_Gemm_nt_blk_var5(cj_Object *A, cj_Object *B, cj_Object *C) {
	cj_Object *AL,    *AR,       *A0,  *A1,  *A2;	
	cj_Object *BL,    *BR,       *B0,  *B1,  *B2;
	
	fprintf(stderr, "      Gemm_nt_blk_var5 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
			A->matrix->m, A->matrix->n,
			B->matrix->m, B->matrix->n,
			C->matrix->m, C->matrix->n);
	fprintf(stderr, "      {\n");
	
	AL = cj_Object_new(CJ_MATRIX); AR = cj_Object_new(CJ_MATRIX);
	A0 = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX); A2 = cj_Object_new(CJ_MATRIX);
	
	BL = cj_Object_new(CJ_MATRIX); BR = cj_Object_new(CJ_MATRIX);
	B0 = cj_Object_new(CJ_MATRIX); B1 = cj_Object_new(CJ_MATRIX); B2 = cj_Object_new(CJ_MATRIX);
	
	int b;
	
	cj_Matrix_part_1x2( A,    AL,  AR,      0, CJ_LEFT );
	cj_Matrix_part_1x2( B,    BL,  BR,      0, CJ_LEFT );
	
	while ( AL->matrix->n < A->matrix->n){
		
		b = min(AR->matrix->n, BLOCK_SIZE);
		
		cj_Matrix_repart_1x2_to_1x3( AL,  /**/ AR,        A0, /**/ A1, A2,
									b, CJ_RIGHT );
		
		cj_Matrix_repart_1x2_to_1x3( BL,  /**/ BR,        B0, /**/ B1, B2,
									b, CJ_RIGHT );
		
		/*------------------------------------------------------------*/
		
		/* C = alpha * A1 * B1' + C; */
		//    FLA_Gemm_internal( FLA_NO_TRANSPOSE, FLA_TRANSPOSE,
		//                       alpha, A1, B1, FLA_ONE, C,
		//                       FLA_Cntl_sub_gemm( cntl ) );
		//                    
		
		cj_Object *alpha, *beta;
		alpha = cj_Object_new(CJ_CONSTANT);
		beta  = cj_Object_new(CJ_CONSTANT);
		cj_Gemm_nt_task(alpha, A1, B1, beta, C); 
		
		/*------------------------------------------------------------*/
		
		cj_Matrix_cont_with_1x3_to_1x2( AL,  /**/ AR,        A0, A1, /**/ A2,
									   CJ_LEFT );
		
		cj_Matrix_cont_with_1x3_to_1x2( BL,  /**/ BR,        B0, B1, /**/ B2,
									   CJ_LEFT );
		
	}
	fprintf(stderr, "      }\n");	
}

void cj_Gemm_nt_blk_var3 ( cj_Object *A, cj_Object *B, cj_Object *C) {
	cj_Object *BT,              *B0,
          	*BB,              *B1,
	                            *B2;
	
	cj_Object *CL,    *CR,       *C0,  *C1,  *C2;
	
	fprintf(stderr, "    Gemm_nt_blk_var3 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
			A->matrix->m, A->matrix->n,
			B->matrix->m, B->matrix->n,
			C->matrix->m, C->matrix->n);
	fprintf(stderr, "    {\n");
	
	BT = cj_Object_new(CJ_MATRIX); B0 = cj_Object_new(CJ_MATRIX);
	BB = cj_Object_new(CJ_MATRIX); B1 = cj_Object_new(CJ_MATRIX);
	B2 = cj_Object_new(CJ_MATRIX);
	
	CL = cj_Object_new(CJ_MATRIX); CR = cj_Object_new(CJ_MATRIX);
	C0 = cj_Object_new(CJ_MATRIX); C1 = cj_Object_new(CJ_MATRIX); C2 = cj_Object_new(CJ_MATRIX);
	
	int b;
	
	cj_Matrix_part_2x1( B,    BT, 
					   BB,            0, CJ_TOP );
	
	cj_Matrix_part_1x2( C,    CL,  CR,      0, CJ_LEFT );
	
	while ( BT->matrix->m < B->matrix->m ){
		
		b = min(BB->matrix->m, BLOCK_SIZE);
		
		cj_Matrix_repart_2x1_to_3x1( BT,                B0, 
									/* ** */            /* ** */
									B1, 
									BB,                B2,        b, CJ_BOTTOM );
		
		cj_Matrix_repart_1x2_to_1x3( CL,  /**/ CR,        C0, /**/ C1, C2,
									b, CJ_RIGHT );
		
		/*------------------------------------------------------------*/
		
		/* C1 = A * B1' + C1; */
		//    FLA_Gemm_internal( FLA_NO_TRANSPOSE, FLA_TRANSPOSE,
		//                       alpha, A, B1, beta, C1,
		//                       FLA_Cntl_sub_gemm( cntl ) );
		
		cj_Gemm_nt_blk_var5(A, B1, C1);
		
		/*------------------------------------------------------------*/
		
		cj_Matrix_cont_with_3x1_to_2x1( BT,                B0, 
									   B1, 
									   /* ** */           /* ** */
									   BB,                B2,     CJ_TOP );
		
		cj_Matrix_cont_with_1x3_to_1x2( CL,  /**/ CR,        C0, C1, /**/ C2,
									   CJ_LEFT );
		
	}
	fprintf(stderr, "    }\n");
}

void cj_Gemm_nt_blk_var1 (cj_Object *A, cj_Object *B, cj_Object *C) {
	cj_Object *AT,    *A0,
	  *AB,    *A1,
	*A2;
	
	cj_Object *CT,    *C0,
	*CB,    *C1,
	*C2;
	
	
	fprintf(stderr, "  Gemm_nt_blk_var1 (A(%d, %d), B(%d, %d), C(%d, %d)): \n", 
			A->matrix->m, A->matrix->n,
			B->matrix->m, B->matrix->n,
			C->matrix->m, C->matrix->n);
	fprintf(stderr, "  {\n");
	
	AT = cj_Object_new(CJ_MATRIX); A0 = cj_Object_new(CJ_MATRIX);
	AB = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX);
	A2 = cj_Object_new(CJ_MATRIX);
	
	CT = cj_Object_new(CJ_MATRIX); C0 = cj_Object_new(CJ_MATRIX);
	CB = cj_Object_new(CJ_MATRIX); C1 = cj_Object_new(CJ_MATRIX);
	C2 = cj_Object_new(CJ_MATRIX);
	
	
	int b;
	
	cj_Matrix_part_2x1( A,    AT, 
					   AB,            0, CJ_TOP );
	
	cj_Matrix_part_2x1( C,    CT, 
					   CB,            0, CJ_TOP );
	
	while ( AT->matrix->m < A->matrix->m ){
		
		b = min(AB->matrix->m, BLOCK_SIZE);
		
		cj_Matrix_repart_2x1_to_3x1( AT,                A0, 
									/* ** */            /* ** */
									A1, 
									AB,                 A2,        b, CJ_BOTTOM );
		
		cj_Matrix_repart_2x1_to_3x1( CT,                C0, 
									/* ** */            /* ** */
									C1, 
									CB,                C2,        b, CJ_BOTTOM );
		
		/*------------------------------------------------------------*/
		
		/* C1 = A1 * B' + C1; */
		//    FLA_Gemm_internal( FLA_NO_TRANSPOSE, FLA_TRANSPOSE,
		//                       alpha, A1, B, beta, C1,
		//                       FLA_Cntl_sub_gemm( cntl ) );
		
		cj_Gemm_nt_blk_var3 (A1, B, C1);
		
		/*------------------------------------------------------------*/
		
		cj_Matrix_cont_with_3x1_to_2x1( AT,                A0, 
									   A1, 
									   /* ** */           /* ** */
									   AB,                A2,     CJ_TOP );
		
		cj_Matrix_cont_with_3x1_to_2x1( CT,                C0, 
									   C1, 
									   /* ** */           /* ** */
									   CB,                C2,     CJ_TOP );
		
	}
	fprintf(stderr, "  }\n");
	
}

void cj_Syrk_ln_task_function (void *task_ptr) {
  fprintf(stderr, "Inside syrk_ln_task\n");

  cj_Task *task = (cj_Task *) task_ptr;
  cj_Worker *worker = task->worker;
  cj_devType devtype = worker->devtype;
  int device_id = worker->device_id;

  cj_Object *A, *C;
  cj_Matrix *a, *c;
  A = task->arg->dqueue->head;
  C = A->next;
  a = A->matrix;
  c = C->matrix;
  if (device_id != -1 && devtype == CJ_DEV_CUDA) {
#ifdef CJ_HAVE_CUDA
    cudaSetDevice(device_id);
    cj_Device *device = worker->cj_ptr->device[device_id];
    cj_Cache *cache = &(device->cache);
    cublasHandle_t *handle = &(device->handle);
    cublasStatus_t status;
    cj_Object *now_a = a->base->dist[a->offm/BLOCK_SIZE][a->offn/BLOCK_SIZE]->dqueue->head;
    cj_Object *now_c = c->base->dist[c->offm/BLOCK_SIZE][c->offn/BLOCK_SIZE]->dqueue->head;
    while (now_a->distribution->device_id != device_id && now_a != NULL) now_a = now_a->next;
    while (now_c->distribution->device_id != device_id && now_c != NULL) now_c = now_c->next;
    if (!now_a || !now_c) 
      cj_Blas_error("cj_Syrk_ln_task_function", "No propriate distribution.");

    if (a->eletype == CJ_SINGLE) { 
      float f_one = 1.0;
      float *a_buff = (float *) cache->dev_ptr[now_a->distribution->cache_id];
      float *c_buff = (float *) cache->dev_ptr[now_c->distribution->cache_id];
      status = cublasSsyrk(*handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, c->m, a->n, &f_one, a_buff, BLOCK_SIZE, 
          &f_one, c_buff, BLOCK_SIZE);
    }
    else {
      double f_one = 1.0;
      double *a_buff = (double *) cache->dev_ptr[now_a->distribution->cache_id];
      double *c_buff = (double *) cache->dev_ptr[now_c->distribution->cache_id];
      fprintf(stderr, "%d, %d\n", (int) a_buff, (int) c_buff);
      status = cublasDsyrk(*handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, c->m, a->n, &f_one, a_buff, BLOCK_SIZE, 
          &f_one, c_buff, BLOCK_SIZE);
    }
    if (status != CUBLAS_STATUS_SUCCESS) cj_Blas_error("cj_Syrk_ln_task_function", "cublas failure");
#endif
  }
  else {
    if (a->eletype == CJ_SINGLE) {
      float f_one = 1.0;
      float *a_buff = (float *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;  //Row-major order???
      float *c_buff = (float *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
      ssyrk_("L", "N", &(c->m), &(a->n), &f_one, a_buff, &(a->base->m), &f_one, c_buff, &(c->base->m)); //Column-major order??
    }
    else {
      double f_one = 1.0;
      double *a_buff = (double *) (a->base->buff) + (a->base->m)*(a->offn) + a->offm;
      double *c_buff = (double *) (c->base->buff) + (c->base->m)*(c->offn) + c->offm;
      dsyrk_("L", "N", &(c->m), &(a->n), &f_one, a_buff, &(a->base->m), &f_one, c_buff, &(c->base->m));
    }
  }

  fprintf(stderr, YELLOW "  Worker_execute %d (%d, %s), A(%d, %d), C(%d, %d): \n" NONE, 
      task->worker->id, task->id, task->name,
      a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE, 
      c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE);  
}

void cj_Syrk_ln_task(cj_Object *alpha, cj_Object *A, cj_Object *beta, cj_Object *C) {
  /* Syrk will read A, C and write C. */
  cj_Object *A_copy, *C_copy, *task;
  cj_Matrix *a, *b, *c;

  /* Generate copies of A and C. */
  A_copy = cj_Object_new(CJ_MATRIX);
  C_copy = cj_Object_new(CJ_MATRIX);
  cj_Matrix_duplicate(A, A_copy);
  cj_Matrix_duplicate(C, C_copy);

  a = A_copy->matrix; c = C_copy->matrix;
  task = cj_Object_new(CJ_TASK);
  cj_Task_set(task->task, &cj_Syrk_ln_task_function);

  /* Pushing arguments. */
  cj_Object *arg_A = cj_Object_append(CJ_MATRIX, a);
  cj_Object *arg_C = cj_Object_append(CJ_MATRIX, c);
  arg_A->rwtype = CJ_R;
  arg_C->rwtype = CJ_RW;
  cj_Dqueue_push_tail(task->task->arg, arg_A);
  cj_Dqueue_push_tail(task->task->arg, arg_C);

  /* Setup task name. */
  snprintf(task->task->name, 64, "Syrk_ln%d_A_%d_%d_C_%d_%d", 
      task->task->id,
      a->offm/BLOCK_SIZE, a->offn/BLOCK_SIZE,
      c->offm/BLOCK_SIZE, c->offn/BLOCK_SIZE );

  cj_Task_dependency_analysis(task);
}

void cj_Syrk_ln_blk_var1 (cj_Object *A, cj_Object *C) {
	cj_Object *AT,    *A0,
	*AB,    *A1,
	*A2;
	cj_Object *CTL, *CTR,    *C00, *C01, *C02,
	*CBL, *CBR,    *C10, *C11, *C12,
	*C20, *C21, *C22;
	
	fprintf(stderr, "Syrk_nn_blk_var1 (A(%d, %d), C(%d, %d)): \n", 
			A->matrix->m, A->matrix->n,
			C->matrix->m, C->matrix->n);
	fprintf(stderr, "{\n");
	
	AT = cj_Object_new(CJ_MATRIX); A0 = cj_Object_new(CJ_MATRIX);
	AB = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX);
	A2 = cj_Object_new(CJ_MATRIX);
	
	CTL = cj_Object_new(CJ_MATRIX); CTR = cj_Object_new(CJ_MATRIX); C00 = cj_Object_new(CJ_MATRIX); C01 = cj_Object_new(CJ_MATRIX); C02 = cj_Object_new(CJ_MATRIX);
	CBL = cj_Object_new(CJ_MATRIX); CBR = cj_Object_new(CJ_MATRIX); C10 = cj_Object_new(CJ_MATRIX); C11 = cj_Object_new(CJ_MATRIX); C12 = cj_Object_new(CJ_MATRIX);
	C20 = cj_Object_new(CJ_MATRIX); C21 = cj_Object_new(CJ_MATRIX); C22 = cj_Object_new(CJ_MATRIX);
	
	int b;
	
	cj_Matrix_part_2x1(A, AT,
					   AB,     0, CJ_TOP);
	cj_Matrix_part_2x2(C, CTL, CTR,
					   CBL, CBR,     0, 0,  CJ_TL);
/*	
	fprintf(stderr, "flag1: preheader\n");
	fprintf(stderr, "flag1: AT->Matrix->n:%d\n", AT->matrix->n);    //-----------> n=16 is an error here...
	fprintf(stderr, "flag1: A->Matrix->n:%d\n", A->matrix->n);
	*/
	while (AT->matrix->m < A->matrix->m) {
		
		fprintf(stderr, "flag2: header\n");
		b = min(AB->matrix->n, BLOCK_SIZE);
		
		cj_Matrix_repart_2x1_to_3x1(AT,                A0, 
									/* ** */            /* ** */
									A1, 
									AB,                A2,        b, CJ_BOTTOM );
		
		cj_Matrix_repart_2x2_to_3x3( CTL, /**/ CTR,       C00, /**/ C01, C02,
									/* ************* */   /* ******************** */
									C10, /**/ C11, C12,
									CBL, /**/ CBR,       C20, /**/ C21, C22,
									b, b, CJ_BR );
		
		/*------------------------------------------------------------*/
		
		/* C10 = C10 + A1 * A0' */
		//FLA_Gemm_internal( FLA_NO_TRANSPOSE, FLA_TRANSPOSE, 
		//                   alpha, A1, A0, beta, C10,
		//                   FLA_Cntl_sub_gemm( cntl ) );
		
		cj_Gemm_nt_blk_var1(A1, A0, C10);
		//output result....offset....base......m.....n
		//fprintf(stderr, "Enter inside the while loop!\n");
		
		/* C11 = C11 + A1 * A1' */
		//FLA_Syrk_internal( FLA_LOWER_TRIANGULAR, FLA_NO_TRANSPOSE, 
		//                   alpha, A1, beta, C11,
		//                   FLA_Cntl_sub_syrk( cntl ) );
		
		cj_Syrk_ln_blk_var5(A1, C11);
		
		/*------------------------------------------------------------*/
		
		cj_Matrix_cont_with_3x1_to_2x1( AT,                A0, 
									   A1, 
									   /* ** */           /* ** */
									   AB,                A2,     CJ_TOP );
		
		cj_Matrix_cont_with_3x3_to_2x2( CTL, /**/ CTR,       C00, C01, /**/ C02,
									   C10, C11, /**/ C12,
									   /* ************** */  /* ****************** */
									   CBL, /**/ CBR,       C20, C21, /**/ C22,
									   CJ_TL );
		
		
	}
	
}

void cj_Syrk_ln_blk_var2 (cj_Object *A, cj_Object *C) {
  cj_Object *AT,              *A0,
            *AB,              *A1,
                              *A2;

  cj_Object *CTL,   *CTR,      *C00, *C01, *C02, 
            *CBL,   *CBR,      *C10, *C11, *C12,
                               *C20, *C21, *C22;

  fprintf(stderr, "Syrk_nn_blk_var2 (A(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "{\n");

  AT = cj_Object_new(CJ_MATRIX); A0 = cj_Object_new(CJ_MATRIX);
  AB = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX);
                                 A2 = cj_Object_new(CJ_MATRIX);

  CTL = cj_Object_new(CJ_MATRIX); CTR = cj_Object_new(CJ_MATRIX); C00 = cj_Object_new(CJ_MATRIX); C01 = cj_Object_new(CJ_MATRIX); C02 = cj_Object_new(CJ_MATRIX);
  CBL = cj_Object_new(CJ_MATRIX); CBR = cj_Object_new(CJ_MATRIX); C10 = cj_Object_new(CJ_MATRIX); C11 = cj_Object_new(CJ_MATRIX); C12 = cj_Object_new(CJ_MATRIX);
                                                                  C20 = cj_Object_new(CJ_MATRIX); C21 = cj_Object_new(CJ_MATRIX); C22 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_2x1( A,    AT, 
                            AB,            0, CJ_TOP );

  cj_Matrix_part_2x2( C,    CTL, CTR,
                                 CBL, CBR,     0, 0, CJ_TL );

  while ( AT->matrix->m < A->matrix->m ){

	b = min(AB->matrix->n, BLOCK_SIZE);

    cj_Matrix_repart_2x1_to_3x1( AT,                A0, 
                              /* ** */           /* ** */
                                                    A1, 
                                 AB,                A2,        b, CJ_BOTTOM );

    cj_Matrix_repart_2x2_to_3x3( CTL, /**/ CTR,       C00, /**/ C01, C02,
                        /* ************* */   /* ******************** */
                                                      C10, /**/ C11, C12,
                                 CBL, /**/ CBR,       C20, /**/ C21, C22,
                                  b, b, CJ_BR );

    /*------------------------------------------------------------*/

    /* C21 = C21 + A2 * A1' */
    //cj_Gemm_internal( FLA_NO_TRANSPOSE, FLA_TRANSPOSE, 
    //                   alpha, A2, A1, beta, C21,
    //                   FLA_Cntl_sub_gemm( cntl ) );
    cj_Gemm_nt_blk_var1(A2, A1, C21);

    /* C11 = C11 + A1 * A1' */
    //cj_Syrk_internal( FLA_LOWER_TRIANGULAR, FLA_NO_TRANSPOSE, 
    //                   alpha, A1, beta, C11,
    //                   FLA_Cntl_sub_syrk( cntl ) );
    cj_Syrk_ln_blk_var5(A1, C11);

    /*------------------------------------------------------------*/

    cj_Matrix_cont_with_3x1_to_2x1( AT,                A0, 
                                                       A1, 
                                 /* ** */           /* ** */
                                    AB,                A2,     CJ_TOP );

    cj_Matrix_cont_with_3x3_to_2x2( CTL, /**/ CTR,       C00, C01, /**/ C02,
                                                         C10, C11, /**/ C12,
                                 /* ************** */  /* ****************** */
                                    CBL, /**/ CBR,       C20, C21, /**/ C22,
                                    CJ_TL );

  }
  fprintf(stderr, "}\n");
}
	
void cj_Syrk_ln_blk_var3 (cj_Object *A, cj_Object *C) {
  cj_Object *AT,  *A0,  *AB,  *A1,  *A2;
  cj_Object *CTL, *CTR, *C00, *C01, *C02,
            *CBL, *CBR, *C10, *C11, *C12,
                        *C20, *C21, *C22;

  fprintf(stderr, "Syrk_ln_blk_var3 (A(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "{\n");

  AT = cj_Object_new(CJ_MATRIX);  A0 = cj_Object_new(CJ_MATRIX);
  AB = cj_Object_new(CJ_MATRIX);  A1 = cj_Object_new(CJ_MATRIX);
                                  A2 = cj_Object_new(CJ_MATRIX);

  CTL = cj_Object_new(CJ_MATRIX); CTR = cj_Object_new(CJ_MATRIX); 
  C00 = cj_Object_new(CJ_MATRIX); C01 = cj_Object_new(CJ_MATRIX); C02 = cj_Object_new(CJ_MATRIX);

  CBL = cj_Object_new(CJ_MATRIX); CBR = cj_Object_new(CJ_MATRIX);
  C10 = cj_Object_new(CJ_MATRIX); C11 = cj_Object_new(CJ_MATRIX); C12 = cj_Object_new(CJ_MATRIX);
  C20 = cj_Object_new(CJ_MATRIX); C21 = cj_Object_new(CJ_MATRIX); C22 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_2x1(A,     AT,
                            AB,          0,    CJ_BOTTOM);
  cj_Matrix_part_2x2(C,     CTL, CTR,
                            CBL, CBR,    0, 0, CJ_BR);

  while (AB->matrix->m < A->matrix->m) {
    b = min(AT->matrix->m, BLOCK_SIZE);

    cj_Matrix_repart_2x1_to_3x1(AT,           A0,
                             /* ** */      /* ** */
                                              A1,
                                AB,           A2,      b,  CJ_BOTTOM);

    cj_Matrix_repart_2x2_to_3x3(CTL, /**/ CTR,       C00, C01, /**/ C02,
                                                     C10, C11, /**/ C12,
                            /* ************** */  /* ****************** */
                                CBL, /**/ CBR,       C20, C21, /**/ C22,
                                b, b, CJ_TL);

    /* ------------------------------------------------------------------ */
    //cj_Gemm_nt(A2, A1, C21);
    /* ------------------------------------------------------------------ */

  }

  fprintf(stderr, "}\n");
}

void cj_Syrk_ln_blk_var5 (cj_Object *A, cj_Object *C) {
  cj_Object *AL, *AR, *A0, *A1, *A2;

  fprintf(stderr, "  Syrk_ln_blk_var5 (A(%d, %d), C(%d, %d)): \n", 
      A->matrix->m, A->matrix->n,
      C->matrix->m, C->matrix->n);
  fprintf(stderr, "  {\n");

  AL = cj_Object_new(CJ_MATRIX); AR = cj_Object_new(CJ_MATRIX);
  A0 = cj_Object_new(CJ_MATRIX); A1 = cj_Object_new(CJ_MATRIX); A2 = cj_Object_new(CJ_MATRIX);

  int b;

  cj_Matrix_part_1x2(A, AL, AR, 0, CJ_LEFT);

  while (AL->matrix->n < A->matrix->n) {
    b = min(AR->matrix->n, BLOCK_SIZE);

    cj_Matrix_repart_1x2_to_1x3(AL, /**/ AR,       A0, /**/ A1, A2,
        b, CJ_RIGHT);

    /* ------------------------------------------------------------------ */
    cj_Object *alpha, *beta;
    alpha = cj_Object_new(CJ_CONSTANT);
    beta  = cj_Object_new(CJ_CONSTANT);
    cj_Syrk_ln_task(alpha, A1, beta, C);
    /* ------------------------------------------------------------------ */

    cj_Matrix_cont_with_1x3_to_1x2(AL, /**/ AR,       A0, A1, /**/ A2,
        CJ_LEFT);
  }
  fprintf(stderr, "  }\n");
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

  cj_Queue_end();
  cj_Syrk_ln_blk_var2(A, C);
  cj_Queue_begin();
}
