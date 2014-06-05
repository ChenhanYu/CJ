/*
 * cj_Graph.c
 * Build the DAG and generate the dependency graph(.dot file) as a intermediate output
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cj.h>

static cj_Graph *graph;

void cj_Graph_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_GRAPH_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

void cj_Vertex_set (cj_Object *object, cj_Object *target) {
  if (object->objtype != CJ_VERTEX) cj_Graph_error("Vertex_set", "The object is not a vertex.");
  if (target->objtype != CJ_TASK) cj_Graph_error("Vertex_set", "The target is not a task.");
  object->vertex->task  = target->task;
  cj_Color color;
  if (target->task->tasktype == CJ_TASK_GEMM) color = CJ_ORANGE;
  else if (target->task->tasktype == CJ_TASK_TRSM) color = CJ_RED;
  else if (target->task->tasktype == CJ_TASK_SYRK) color = CJ_BLUE;
  else if (target->task->tasktype == CJ_TASK_POTRF) color = CJ_GREEN;
  else color = CJ_BLACK;
  object->vertex->color = color;
}

cj_Vertex *cj_Vertex_new () {
  cj_Vertex *vertex = (cj_Vertex*) malloc(sizeof(cj_Vertex));
  if (!vertex) cj_Graph_error("Vertex_new", "memory allocation failed.");
  return vertex;
}

void cj_Edge_set (cj_Object *object, cj_Object *in, cj_Object *out , cj_Bool war) {
  if (object->objtype != CJ_EDGE) cj_Graph_error("Edge_set", "The object is not a vertex.");
  if (in->objtype != CJ_TASK) cj_Graph_error("Edge_set", "The 1.st target is not a task.");
  if (out->objtype != CJ_TASK) cj_Graph_error("Edge_set", "The 2.nd target is not a task.");
  object->edge->in = in->task;
  object->edge->out = out->task;
  object->edge->war = war;
}

cj_Edge *cj_Edge_new () {
  cj_Edge *edge = (cj_Edge*) malloc(sizeof(cj_Edge));
  if (!edge) cj_Graph_error("Vertex_new", "memory allocation failed.");
  return edge;
}

void cj_Graph_init () {
  graph = (cj_Graph*) malloc(sizeof(cj_Graph));
  graph->nvisit = 0;
  graph->vertex = cj_Object_new(CJ_DQUEUE);
  graph->edge   = cj_Object_new(CJ_DQUEUE);
  if (!graph || ! graph->vertex || !graph->edge) {
    cj_Graph_error("Graph_new", "memory allocation failed.");
  }
}

void cj_Graph_vertex_add (cj_Object *vertex) {
  if (!graph) cj_Graph_error("Graph_vertex_add", "Need initialization!");
  if (vertex->objtype != CJ_VERTEX) cj_Graph_error("Graph_vertex_add", "This is not a vertex.");
  cj_Dqueue_push_tail(graph->vertex, vertex);
}

cj_Object *cj_Graph_vertex_get () {
  if (!graph) cj_Graph_error("Graph_vertex_get", "Need initialization!");
  return graph->vertex;
};

void cj_Graph_edge_add (cj_Object *edge) {
  if (!graph) cj_Graph_error("Graph_edge_add", "Need initialization!");
  if (edge->objtype != CJ_EDGE) cj_Graph_error("Graph_edge_add", "This is not an edge.");
  cj_Dqueue_push_tail(graph->edge, edge);
}

//Output the dot file
void cj_Graph_output_dot () {
  if (!graph) cj_Graph_error("Graph_output_dot", "Need initialization!");
  if (cj_Dqueue_get_size(graph->edge) > 0 && cj_Dqueue_get_size(graph->vertex)) {
    cj_Object *vert_I = graph->vertex->dqueue->head;
    cj_Object *edge_I = graph->edge->dqueue->head;
    FILE * pFile = fopen("output.dot","w");
    fprintf(pFile, "digraph CJ_GRAPH { \n");
    fprintf(pFile, "  node [shape=none,color=black]; \n");
    while (vert_I) {
      cj_Task *task  = vert_I->vertex->task;
      cj_Color color = vert_I->vertex->color;

      fprintf(pFile, "  %s[label=<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n", task->name);
      fprintf(pFile, "  <TR><TD>%s</TD><TD>%d</TD></TR>\n", task->name, task->id);
      fprintf(pFile, "  <TR><TD COLSPAN=\"2\">%s</TD></TR></TABLE>>\n", task->label);
      if (color == CJ_RED)         fprintf(pFile, "  color=red];\n");
      else if (color == CJ_GREEN)  fprintf(pFile, "  color=green];\n");
      else if (color == CJ_BLUE)   fprintf(pFile, "  color=blue];\n");
      else if (color == CJ_ORANGE) fprintf(pFile, "  color=orange];\n");
      else if (color == CJ_PURPLE) fprintf(pFile, "  color=purple];\n");
      else if (color == CJ_YELLOW) fprintf(pFile, "  color=yellow];\n");
      else                         fprintf(pFile, "  color=black];\n");
      vert_I = vert_I->next;
    }
    while (edge_I) {
      cj_Task *in  = edge_I->edge->in;
      cj_Task *out = edge_I->edge->out;
      if (edge_I->edge->war == TRUE) {
        fprintf(pFile, "  %s -> %s[style=dotted];\n", in->name, out->name);
      }
      else {
        fprintf(pFile, "  %s -> %s;\n", in->name, out->name);
      }
      edge_I = edge_I->next;
    }
    fprintf(pFile, "}\n");
    fclose(pFile);
  }
}
