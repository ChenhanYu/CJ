#include <stdio.h>
#include <stdlib.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"

int main () {
  cj_Object *A, *B, *C;
  int ma = 256, na = 128, mb = na, nb = 256, mc = ma, nc = nb;

  A = cj_Object_new(CJ_MATRIX);
  B = cj_Object_new(CJ_MATRIX);
  C = cj_Object_new(CJ_MATRIX);
  
  /* C = A*B */
  cj_Gemm(A, B, C);

  return 0;
}
