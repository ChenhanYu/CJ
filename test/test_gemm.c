#include <stdio.h>
#include <stdlib.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"
#include "../src/cj.h"

int main () {
  cj_Object *A, *B, *C, *D;
  int ma = 128, na = 128, mb = na, nb = 128, mc = ma, nc = nb;
  int md = ma, nd = nc;

  cj_Init();

  A = cj_Object_new(CJ_MATRIX);
  B = cj_Object_new(CJ_MATRIX);
  C = cj_Object_new(CJ_MATRIX);
  D = cj_Object_new(CJ_MATRIX);
 
  cj_Matrix_set(A, ma, na);
  cj_Matrix_set(B, mb, nb);
  cj_Matrix_set(C, mc, nc);
  cj_Matrix_set(D, md, nd);

  /* C = A*B */
  cj_Gemm(A, B, C);
  cj_Gemm(A, C, D);

  cj_Term();

  return 0;
}
