#include <stdio.h>
#include <stdlib.h>
#include "cj_Macro.h"
#include "cj_Object.h"


void cj_Blas_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_BLAS_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}


void cj_Gemm_blk_var1 (cj_Object *A, cj_Object *B, cj_Object *C) {

}

void cj_Gemm_blk_var2 (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *BL, *BR, *B0, *B1, *B2, *CL, *CR, *C0, *C1, *C2;

  BL = cj_Object_new(CJ_MATRIX);
  BR = cj_Object_new(CJ_MATRIX);
  B0 = cj_Object_new(CJ_MATRIX);
  B1 = cj_Object_new(CJ_MATRIX);
  B2 = cj_Object_new(CJ_MATRIX);
  CL = cj_Object_new(CJ_MATRIX);
  CR = cj_Object_new(CJ_MATRIX);
  C0 = cj_Object_new(CJ_MATRIX);
  C1 = cj_Object_new(CJ_MATRIX);
  C2 = cj_Object_new(CJ_MATRIX);

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
    cj_Gemm_blk_var1 (A, B1, C1);
	/* ------------------------------------------------------------------ */

	cj_Matrix_cont_with_1x3_to_1x2(BL, /**/ BR,     B0, /**/ B1, B2, 
			                       CJ_LEFT);
	cj_Matrix_cont_with_1x3_to_1x2(CL, /**/ CR,     C0, /**/ C1, C2, 
			                       CJ_LEFT);

  }
}

void cj_Gemm_blk_var3 (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Object *AT, *A0, *AB, *A1, *A2, *CT, *C0, *CB, *C1, *C2;

  AT = cj_Object_new(CJ_MATRIX);
  A0 = cj_Object_new(CJ_MATRIX);
  AB = cj_Object_new(CJ_MATRIX);
  A1 = cj_Object_new(CJ_MATRIX);
  A2 = cj_Object_new(CJ_MATRIX);
  CT = cj_Object_new(CJ_MATRIX);
  C0 = cj_Object_new(CJ_MATRIX);
  CB = cj_Object_new(CJ_MATRIX);
  C1 = cj_Object_new(CJ_MATRIX);
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
    cj_Gemm_blk_var2 (A1, B, C1);
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
}

void cj_Gemm (cj_Object *A, cj_Object *B, cj_Object *C) {
  cj_Matrix *a, *b, *c;
  if (!A || !B || !C) 
	  cj_Blas_error("gemm", "matrices haven't been initialized yet.");
  if (!A->matrix || !B->matrix || !C->matrix)
	  cj_Blas_error("gemm", "Object types are not matrix type.");
  a = A->matrix;
  b = B->matrix;
  c = C->matrix;
  if ((c->m != a->m) || (c->n != b->n) || (a->n != b->m)) 
    cj_Blas_error("gemm", "matrices dimension aren't matched.");

  cj_Gemm_blk_var3(A, B, C);
}
