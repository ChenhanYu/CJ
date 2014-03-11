#include <stdio.h>
#include <stdlib.h>
#include "cj_Object.h"

void cj_Object_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_OBJECT_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}

cj_Matrix *cj_Matrix_new (int m, int n) {
  cj_Matrix *matrix;

  return matrix;
}

void cj_Matrix_delete (cj_Matrix *matrix) {

}

cj_Object *cj_Object_new (cj_Type type) {
  cj_Object *object;
  object = (cj_Object*) malloc(sizeof(cj_Object));
  if (!object) cj_Object_error("new", "memory allocation failed");

  if (type == CJ_MATRIX) {
    object->eletype = CJ_MATRIX;
  }
  else if (type == CJ_DQUEUE) {

  }

  object->prev = NULL;
  object->next = NULL;

  return object;
}
