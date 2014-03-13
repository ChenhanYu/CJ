#include <stdio.h>
#include <stdlib.h>
#include "cj_Object.h"

void cj_Lapack_error (const char *func_name, char* msg_text) {
  fprintf(stderr, "CJ_LAPACK_ERROR: %s(): %s\n", func_name, msg_text);
  abort();
  exit(0);
}


