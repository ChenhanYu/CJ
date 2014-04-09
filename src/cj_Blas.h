void cj_Gemm_nn_task_function (void*);
void cj_Gemm_nn (cj_Object*, cj_Object*, cj_Object*);
void cj_Syrk_ln (cj_Object*, cj_Object*);


void cj_Syrk_ln_blk_var1 (cj_Object *A, cj_Object *C);
void cj_Syrk_ln_blk_var3 (cj_Object *A, cj_Object *C);

extern void sgemm_( char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int * );
extern void dgemm_( char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int * );
