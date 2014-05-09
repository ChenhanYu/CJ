/*
 * cj_Blas.h
 * The header file for cj_Blas.c
 * Implement a subset of BLAS (Basic Linear Algebra Subroutine) based on the FLAME implementation.
 */
void cj_Gemm_nn_task_function (void*);
void cj_Gemm_nn (cj_Object*, cj_Object*, cj_Object*);
extern void sgemm_( char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int * );
extern void dgemm_( char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int * );

void cj_Syrk_ln_task_function(void*);
void cj_Syrk_ln_task(cj_Object*, cj_Object*, cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var1 (cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var2 (cj_Object*, cj_Object*);
void cj_Syrk_ln_blk_var5 (cj_Object*, cj_Object*);
void cj_Syrk_ln (cj_Object*, cj_Object*);

void cj_Trsm_rlt_task_function(void*);
