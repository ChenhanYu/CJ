#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <CJ.h>
#include "../src/cj_Object.h"
#include "../src/cj_Blas.h"
#include "../src/cj_Lapack.h"
#include "../src/cj.h"

int main (int argc, char* argv[]) {
  cj_Object *A11, *B11, *B21, *B22, *C11, *C21, *C22;
  int i, m = 6144;
	for (i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-M")) {
			sscanf(argv[i+1], "%d", &m);
    }
  }

  int ma = m, na = m;
  int mb = m, nb = m;
  int mc = m, nc = m;
  int nworker = 4;
  float time_ms;

  cj_Init(nworker);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);
  cudaEventRecord(beg, 0);

  A11 = cj_Object_new(CJ_MATRIX);
  B11 = cj_Object_new(CJ_MATRIX); B21 = cj_Object_new(CJ_MATRIX);
  C11 = cj_Object_new(CJ_MATRIX); C21 = cj_Object_new(CJ_MATRIX);

  cj_Matrix_set(A11, ma, na);
  cj_Matrix_set(B11, mb, nb); cj_Matrix_set(B21, mb, nb);
  cj_Matrix_set(C11, mb, nb); cj_Matrix_set(C21, mb, nb);

  cj_Matrix_set_special_chol (A11);
  cj_Matrix_set_special_chol (B11);
  cj_Matrix_set_special_chol (C11);

  cj_Chol_l(B11); cj_Trsm_rlt(B11, B21); cj_Syrk_ln(B21, A11);
  cj_Chol_l(C11); cj_Trsm_rlt(C11, C21); cj_Syrk_ln(C21, A11);
  cj_Chol_l(A11);

  cj_Term();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  fprintf(stderr, "  glace_nested(%d) : %f s\n", m, time_ms/1000);	
  

  cj_Profile_output_timeline();

  return 0;
}
