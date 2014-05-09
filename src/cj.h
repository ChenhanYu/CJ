/*
 * cj.h
 * The header file for cj.h
 * cj_error: Error Information Output.
 * cj_Lock: Multithread lock for race.
 * cj_Distribution: Data distribution in which device.
 * cj_Task: Fine grained task, each task is corresponding to a linear algebra operation on a block.
 * cj_Worker: Each worker is corresponding to a thread.
 * cj_Queue: Ready Queue for the dynamic scheduling.
 * cj_Schedule: ready_queue, remaining time(Priority Scheduling), lock management for all the workers.
 */
void cj_Init(int);
void cj_Term();
void cj_Queue_begin();
void cj_Queue_end();

/* cj_Distribution function prototypes */
cj_Distribution *cj_Distribution_new();

/* cj_Task function prototypes */
cj_Task *cj_Task_new ();
void cj_Task_set (cj_Task*, cj_taskType, void (*function)(void*));
void cj_Task_dependency_analysis (cj_Object*);
void cj_Task_dependency_add(cj_Object*, cj_Object*);
void cj_Task_dependencies_update (cj_Object*);

/* cj_Worker function prototypes */
float cj_Worker_estimate_cost (cj_Task*, cj_Worker*);
void cj_Worker_wait_prefetch (cj_Worker*, int, int);

