void cj_Init(int);
void cj_Term();
void cj_Queue_begin();
void cj_Queue_end();

/* cj_Distribution function prototypes */
cj_Distribution *cj_Distribution_new();

/* cj_Task function prototypes */
cj_Task *cj_Task_new ();
void cj_Task_set (cj_Task*, void (*function)(void*));
void cj_Task_dependency_analysis (cj_Object*);
void cj_Task_dependency_add(cj_Object*, cj_Object*);
void cj_Task_dependencies_update (cj_Object*);

/* cj_Worker function prototypes */
float cj_Worker_estimate_cost (cj_Task*, cj_Worker*);
void cj_Worker_wait_prefetch (cj_Worker*, int, int);

