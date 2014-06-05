/* 
 * test_cache.c
 * Test file for the software cache 
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <cj.h>

int main () {
  cj_Object *A, *B, *C, *D;
  int ma = 8, na = 8, mb = na, nb = 8, mc = ma, nc = nb;
  //int ma = 8192, na = 8192, mb = na, nb = 8192, mc = ma, nc = nb;
  int md = ma, nd = nc;
  int nworker = 4;
  int iter;
  cj_Init(nworker);

  A = cj_Object_new(CJ_MATRIX);
  B = cj_Object_new(CJ_MATRIX);
  C = cj_Object_new(CJ_MATRIX);
  D = cj_Object_new(CJ_MATRIX);

  cj_Matrix_set(A, ma, na);
  cj_Matrix_set(B, mb, nb);
  cj_Matrix_set(C, mc, nc);
  cj_Matrix_set(D, md, nd);

  cj_Matrix_set_identity(A);
  cj_Matrix_set_identity(B);
  cj_Matrix_set_identity(C);
  cj_Matrix_set_identity(D);


  for (iter = 0; iter < 2; iter++) {
    cj_Gemm_nn(A, B, C);
    cj_Gemm_nn(A, C, D);
    cj_Gemm_nn(A, C, C);
    cj_Gemm_nn(A, A, B);
    cj_Gemm_nn(A, B, B);
  }

  //sleep(3);

  cj_Profile_output_timeline();

  //exit(0);
  cj_Term();

  return 0;
}
