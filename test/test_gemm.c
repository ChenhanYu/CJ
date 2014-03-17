#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"
#include "../src/cj_Device.h"
#include "../src/cj.h"

int main () {
  cj_Object *A, *B, *C, *D;
  //int ma = 256, na = 256, mb = na, nb = 256, mc = ma, nc = nb;
  int ma = 128, na = 128, mb = na, nb = 128, mc = ma, nc = nb;
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

  /* C = A*B */
  cj_Gemm_nn(A, B, C);
  cj_Gemm_nn(C, B, D);

  cj_Term();

  return 0;
}
