#include "cj_Object.h"
#include "cj_Graph.h"

/* This is a dqueue. */

static cj_Object *taskQueue;

void cj_Init() {
  cj_Graph_init();
}

void cj_Term() {
  cj_Graph_output_dot();
}
