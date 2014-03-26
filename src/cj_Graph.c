#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
/*
#include "cj_Device.h"
#include "cj_Object.h"
#include "cj_Graph.h"
*/
#include <CJ.h>
#include "cj_Object.h"

static cj_Graph *graph;

void cj_Graph_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_GRAPH_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Vertex_set (cj_Object *object, cj_Object *target) {
  cj_Vertex *vertex;
  if (object->objtype != CJ_VERTEX) cj_Object_error("Vertex_set", "The object is not a vertex.");
  if (target->objtype != CJ_TASK) cj_Object_error("Vertex_set", "The target is not a task.");
  object->vertex->task = target->task;
}

cj_Vertex *cj_Vertex_new () {
  cj_Vertex *vertex;
  vertex = (cj_Vertex*) malloc(sizeof(cj_Vertex));
  return vertex;
}

void cj_Edge_set (cj_Object *object, cj_Object *in, cj_Object *out) {
  cj_Edge *edge;
  if (object->objtype != CJ_EDGE) cj_Object_error("Edge_set", "The object is not a vertex.");
  if (in->objtype != CJ_TASK) cj_Object_error("Edge_set", "The 1.st target is not a task.");
  if (out->objtype != CJ_TASK) cj_Object_error("Edge_set", "The 2.nd target is not a task.");
  object->edge->in = in->task;
  object->edge->out = out->task;
}

cj_Edge *cj_Edge_new () {
  cj_Edge *edge;
  edge = (cj_Edge*) malloc(sizeof(cj_Edge));
  return edge;
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
