struct graph_s {
  cj_Object *vertex;
  cj_Object *edge;
};

typedef struct graph_s cj_Graph;

void cj_Graph_init ();
void cj_Graph_vertex_add (cj_Object*);
void cj_Graph_edge_add (cj_Object*);
void cj_Graph_output_dot ();
