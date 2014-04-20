#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <CJ.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"
#include "../src/cj.h"
#include "../src/cj_Profile.h"

int main () {
  cj_Object *A, *B, *C, *D;
  //int ma = 4, na = 4, mb = na, nb = 4, mc = ma, nc = nb;
  //int ma = 8, na = 8, mb = na, nb = 8, mc = ma, nc = nb;
  int ma = 16, na = 16, mb = na, nb = 16, mc = ma, nc = nb;
  //int ma = 64, na = 64, mb = na, nb = 64, mc = ma, nc = nb;
  //int ma = 256, na = 256, mb = na, nb = 256, mc = ma, nc = nb;
  //int ma = 128, na = 128, mb = na, nb = 128, mc = ma, nc = nb;
  //int ma = 4096, na = 4096, mb = na, nb = 4096, mc = ma, nc = nb;
  //int ma = 8192, na = 8192, mb = na, nb = 8192, mc = ma, nc = nb;
  //int ma = 2048, na = 2048, mb = na, nb = 2048, mc = ma, nc = nb;
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
  //cj_Matrix_print(A);
  //cj_Matrix_distribution_print(A);
  for (iter = 0; iter < 2; iter++) {
    cj_Gemm_nn(A, B, C);
    cj_Gemm_nn(A, C, D);
    cj_Gemm_nn(A, C, C);
    cj_Gemm_nn(A, A, B);
    cj_Gemm_nn(A, B, B);
  }

  sleep(3);

  cj_Profile_output_timeline();

  exit(0);
  //cj_Matrix_distribution_print(A);
  //cj_Matrix_distribution_print(B);
  //cj_Matrix_distribution_print(C);
  //cj_Matrix_distribution_print(D);
  
  //cj_Object_acquire(B);
  //cj_Object_acquire(C);
  //cj_Object_acquire(D);
  //cj_Matrix_print(B);
  //cj_Matrix_print(C);
  //cj_Matrix_print(D);
  
  cj_Term();

  return 0;
}
