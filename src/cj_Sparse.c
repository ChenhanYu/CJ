/*
 * cj_Sparse.c
 * The sparse linear algebra routine implementation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cj.h>

void cj_Sparse_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_SPARSE_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
};

void cj_Sparse_mv (cj_Object *A, cj_Object *B, cj_Object *C) {
  if (A->objtype == CJ_CSC) {
    //cj_Csc_mv(A, B, C);
    return;
  }

  if (A->objtype == CJ_MATRIX) {
    cj_Gemm_nn(A, B, C);
    return;
  }

  if (A->objtype != CJ_SPARSE || B->objtype != CJ_MATRIX || C->objtype != CJ_MATRIX) {
    cj_Sparse_error("mv", "Object types don't match.");
  }

  int m0, m1, m2;

  cj_Sparse *a = A->sparse;
  cj_Matrix *b = B->matrix;
  cj_Matrix *c = C->matrix;

  if (c->m != a->m || a->n != b->m || c->n != b->n) {
    cj_Sparse_error("mv", "Dimensions don't match.");
  }
  
  cj_Object *BT = cj_Object_new(CJ_MATRIX), *B0 = cj_Object_new(CJ_MATRIX), 
            *BB = cj_Object_new(CJ_MATRIX), *B1 = cj_Object_new(CJ_MATRIX), 
                                            *B2 = cj_Object_new(CJ_MATRIX);
  cj_Object *CT = cj_Object_new(CJ_MATRIX), *C0 = cj_Object_new(CJ_MATRIX), 
            *CB = cj_Object_new(CJ_MATRIX), *C1 = cj_Object_new(CJ_MATRIX), 
                                            *C2 = cj_Object_new(CJ_MATRIX);
  if (a->s00->objtype == CJ_SPARSE) m0 = a->s00->sparse->m;
  else m0 = a->s00->matrix->m;

  if (a->s11->objtype == CJ_SPARSE) m1 = a->s11->sparse->m;
  else m1 = a->s11->matrix->m;
  
  m2 = a->s22->matrix->m;

  cj_Matrix_part_2x1(B, BT,
                        BB, m2, CJ_BOTTOM);
  cj_Matrix_part_2x1(C, CT,
                        CB, m2, CJ_BOTTOM);

  {
    cj_Matrix_repart_2x1_to_3x1(BT,       B0,
                                          B1,
                             /* ** */  /* ** */
                                BB,       B2,     m0, CJ_TOP);
    cj_Matrix_repart_2x1_to_3x1(CT,       C0,
                                          C1,
                             /* ** */  /* ** */
                                CB,       C2,     m0, CJ_TOP);
    cj_Sparse_mv(a->s00, B0, C0);
    cj_Sparse_mv(a->s00, B0, C0);

  }
}

cj_Csc *cj_Csc_new () {
  cj_Csc *csc = (cj_Csc*) malloc(sizeof(cj_Csc));
  if (!csc) cj_Sparse_error("Csc_new", "memory allocation failed.");
  csc->eletype = CJ_DOUBLE;
  csc->elelen  = sizeof(double); 
  csc->m       = 0;
  csc->n       = 0;
  csc->nnz     = 0;
  csc->colptr  = NULL;
  csc->rowind  = NULL;
  csc->values  = NULL;
  return csc;
};

void cj_Csc_set (cj_Object *object, int m, int n, int nnz) {
  if (object->objtype != CJ_CSC) {
    cj_Sparse_error("Csc_set", "This is not a csc matrix.");
  }
  cj_Csc *csc = object->csc;
  csc->m      = m;
  csc->n      = n;
  csc->nnz    = nnz;
#ifdef CJ_HAVE_CUDA
  cudaMallocHost((void**)&csc->colptr, sizeof(int)*(n + 1));
  cudaMallocHost((void**)&csc->rowind, sizeof(int)*nnz);
  cudaMallocHost((void**)&csc->values, csc->elelen*nnz);
#else
  csc->colptr = (int*) malloc(sizeof(int)*(n + 1));
  csc->rowind = (int*) malloc(sizeof(int)*nnz);
  csc->values = malloc(csc->elelen*nnz);
#endif
}

cj_Sparse *cj_Sparse_new () {
  cj_Sparse *sparse = (cj_Sparse*) malloc(sizeof(cj_Sparse));
  if (!sparse) cj_Sparse_error("Sparse_new", "memory allocation failed.");
  sparse->m    = 0;
  sparse->n    = 0;
  sparse->nnz  = 0;
  sparse->offm = 0;
  sparse->offn = 0;
  sparse->s00  = NULL;
  sparse->s11  = NULL;
  sparse->s20  = NULL;
  sparse->s21  = NULL;
  sparse->s22  = NULL;
  return sparse;
}
