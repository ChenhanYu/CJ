#include <stdio.h>
#include <stdlib.h>
#include "cj_Object.h"
#include "cj_Graph.h"

static cj_Graph *graph;

void cj_Graph_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_GRAPH_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Graph_init () {
  graph = (cj_Graph*) malloc(sizeof(cj_Graph));
  if (!graph) cj_Graph_error("Graph_new", "memory allocation failed.");
  graph->vertex = cj_Object_new(CJ_DQUEUE);
  if (!graph->vertex) cj_Graph_error("Graph_new", "memory allocation failed.");
  graph->edge = cj_Object_new(CJ_DQUEUE);
  if (!graph->edge) cj_Graph_error("Graph_new", "memory allocation failed.");
}

void cj_Graph_vertex_add (cj_Object *vertex) {
  if (!graph) cj_Graph_error("Graph_vertex_add", "Need initialization!");
  if (vertex->objtype != CJ_VERTEX) cj_Graph_error("Graph_vertex_add", "This is not a vertex.");
  cj_Dqueue_push_tail(graph->vertex, vertex);
}

cj_Object *cj_Graph_vertex_get () {return graph->vertex;};

void cj_Graph_edge_add (cj_Object *edge) {
  if (!graph) cj_Graph_error("Graph_edge_add", "Need initialization!");
  if (edge->objtype != CJ_EDGE) cj_Graph_error("Graph_edge_add", "This is not an edge.");
  cj_Dqueue_push_tail(graph->edge, edge);
}

void cj_Graph_output_dot () {
  if (!graph) cj_Graph_error("Graph_output_dot", "Need initialization!");
  if (cj_Dqueue_get_size(graph->edge) > 0) {
    cj_Object *now = graph->edge->dqueue->head;
    FILE * pFile;
	pFile = fopen("output.dot","w");
	fprintf(pFile, "digraph CJ_GRAPH { \n");
    while (now) {
      cj_Task *in = now->edge->in;
      cj_Task *out = now->edge->out;
      fprintf(pFile, "  %s -> %s;\n", in->name, out->name);
      now = now->next;
	}
	fprintf(pFile, "}\n");
    fclose(pFile);
  }
}
