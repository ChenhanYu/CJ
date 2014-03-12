#include <stdio.h>
#include <stdlib.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"

int main () {
  cj_Object *A, *B, *C;
  int ma = 257, na = 128, mb = na, nb = 256, mc = ma, nc = nb;

  A = cj_Object_new(CJ_MATRIX);
  B = cj_Object_new(CJ_MATRIX);
  C = cj_Object_new(CJ_MATRIX);
 
  cj_Matrix_set(A, ma, na);
  cj_Matrix_set(B, mb, nb);
  cj_Matrix_set(C, mc, nc);

  /* C = A*B */
  cj_Gemm(A, B, C);

  return 0;
}
