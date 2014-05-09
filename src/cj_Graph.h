/*
 * cj_Graph.h
 * The header file for cj_Graph.c
 * Build the DAG and generate the dependency graph(.dot file) as a intermediate output
 */

void cj_Graph_init ();
void cj_Graph_vertex_add (cj_Object*);
cj_Object *cj_Graph_vertex_get ();
void cj_Graph_edge_add (cj_Object*);
cj_Object *cj_Graph_edge_get ();
void cj_Graph_output_dot ();

/* cj_Vertex function prototypes */
void cj_Vertex_set (cj_Object*, cj_Object*);
cj_Vertex *cj_Vertex_new ();

/* cj_Edge function prototypes */
void cj_Edge_set (cj_Object*, cj_Object*, cj_Object*, cj_Bool);
cj_Edge *cj_Edge_new ();

/*
struct graph_s {
  cj_Object *vertex;
  cj_Object *edge;
};

typedef struct graph_s cj_Graph;
*/


