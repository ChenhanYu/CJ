/*
 * cj_Lapack.h
 * The header file for cj_Lapack.c
 * Implement a subset of LAPACK (Linear Algebra Package) based on the FLAME implementation.
 */
void cj_Chol_l_task_function (void*);
void cj_Chol_l (cj_Object *);
#ifdef CJ_HAVE_CUDA
void hybrid_dpotrf (cublasHandle_t*, int, double*, int, int*);
#endif
extern void spotrf_(char *, int *, float *, int *, int *);
extern void dpotrf_(char *, int *, double *, int *, int *);
