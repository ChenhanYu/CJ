void cj_Chol_l_task_function (void*);
void cj_Chol_l (cj_Object *);
void hybrid_dpotrf (cublasHandle_t*, int, double*, int, int*);
extern void spotrf_(char *, int *, float *, int *, int *);
extern void dpotrf_(char *, int *, double *, int *, int *);
