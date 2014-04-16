/* cj_Object function prototypes */
cj_Object *cj_Object_new (cj_objType);
cj_Object *cj_Object_append (cj_objType, void*);
void cj_Object_acquire (cj_Object*);

/* cj_Matrix function prototypes */
void cj_Matrix_duplicate (cj_Object*, cj_Object*);
void cj_Matrix_set (cj_Object*, int, int);
void cj_Matrix_set_identity (cj_Object*);
void cj_Matrix_print (cj_Object*);
void cj_Matrix_distribution_print (cj_Object*);
void cj_Matrix_part_2x1 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_1x2 (cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_part_2x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, int, cj_Quadrant);
void cj_Matrix_repart_2x1_to_3x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_repart_1x2_to_1x3 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, int, cj_Side);
void cj_Matrix_cont_with_3x1_to_2x1 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);
void cj_Matrix_cont_with_1x3_to_1x2 (cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Object*, cj_Side);

/* cj_Dqueue function prototypes */
int cj_Dqueue_get_size (cj_Object*);
void cj_Dqueue_push_head (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_head (cj_Object*);
void cj_Dqueue_push_tail (cj_Object*, cj_Object*);
cj_Object *cj_Dqueue_pop_tail (cj_Object*);
void cj_Dqueue_clear(cj_Object*);

