/* 
 * test_chol.c
 * Test file for the Cholesky Decomposition routine in the BLAS
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cj.h>

int main () {
  cj_Object *A, *B, *C, *D;
  //int ma = 4, na = 4, mb = na, nb = 4, mc = ma, nc = nb;
  //int ma = 8, na = 8, mb = na, nb = 8, mc = ma, nc = nb;
  int ma = 16, na = 16, mb = na, nb = 16, mc = ma, nc = nb;
  //int ma = 256, na = 256, mb = na, nb = 256, mc = ma, nc = nb;
  //int ma = 128, na = 128, mb = na, nb = 128, mc = ma, nc = nb;
  //int ma = 4096, na = 4096, mb = na, nb = 4096, mc = ma, nc = nb;
  //int ma = 2048, na = 2048, mb = na, nb = 2048, mc = ma, nc = nb;
  //int ma = 64, na = 64, mb = na, nb = 64, mc = ma, nc = nb;
  int md = ma, nd = nc;
  int nworker = 4;

  cj_Init(nworker);

  A = cj_Object_new(CJ_MATRIX);
  B = cj_Object_new(CJ_MATRIX);
  C = cj_Object_new(CJ_MATRIX);
  D = cj_Object_new(CJ_MATRIX);
 
  cj_Matrix_set(A, ma, na);
  cj_Matrix_set(B, mb, nb);
  cj_Matrix_set(C, mc, nc);
  cj_Matrix_set(D, md, nd);

  //cj_Matrix_set_lowertril_one(A);
  //cj_Matrix_set_2identity(A);
  cj_Matrix_set_special_chol (A);
  cj_Matrix_set_identity(B);
  cj_Matrix_set_identity(C);


  cj_Matrix_print(A);
  /* A -> LL^T */
  cj_Chol_l(A);

  sleep(2);

  cj_Matrix_distribution_print(A);
  //cj_Object_acquire(C);

  sleep(2);

  cj_Matrix_distribution_print(A);
  cj_Matrix_print(A);

  cj_Term();

  return 0;
}
