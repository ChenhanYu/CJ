/*
 * cj_Profile.c
 * Profiling information, which is generated in the run-time.
 * cj_Event: Record the event and related time in the run-time.
 * cj_Profile: Record the profiling information.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <cj.h>

static cj_Profile profile;


/* ---------------------------------------------------------------------
 * cj_Event
 * ---------------------------------------------------------------------
 *  */

cj_Event *cj_Event_new () {
  cj_Event *event = (cj_Event*) malloc(sizeof(cj_Event));
  event->beg     = ((float) clock())/1000;
  event->end     = 0.0;
  event->task_id = -1;
  return event;
};

void cj_Event_set (cj_Object *object, int task_id, cj_eveType evetype) {
  if (object->objtype != CJ_EVENT) {

  }
  cj_Event *event = object->event;
  event->task_id = task_id;
  event->evetype = evetype;
}

void cj_Event_delete (cj_Event *event) {
  free(event);
};

/* ---------------------------------------------------------------------
 * cj_Profile
 * ---------------------------------------------------------------------
 *  */

void cj_Profile_worker_record (cj_Worker *worker, cj_eveType evetype) {
  if (evetype == CJ_EVENT_TASK_RUN_BEG) {
    cj_Object *event = cj_Object_new(CJ_EVENT);
    cj_Event_set(event, worker->current_task->id, evetype);
    cj_Dqueue_push_tail(profile.worker_timeline[worker->id], event);
  }
  else if (evetype == CJ_EVENT_TASK_RUN_END || evetype == CJ_EVENT_FETCH_END) {
    cj_Object *event = profile.worker_timeline[worker->id]->dqueue->tail;
    event->event->end = ((float) clock())/1000;
  }
  else {
    cj_Object *event = cj_Object_new(CJ_EVENT);
    cj_Event_set(event, -1, evetype);
    cj_Dqueue_push_tail(profile.worker_timeline[worker->id], event);
  }
}

void cj_Profile_init () {
  int i;
  for (i = 0; i < MAX_WORKER; i++) {
    profile.worker_timeline[i] = cj_Object_new(CJ_DQUEUE);
  }
};

void cj_Profile_output_timeline () {
  //if (!graph) cj_Profile_error("Graph_output_timeline", "Need initialization!");
  FILE * pFile = fopen("timeline.m","w");
  int i;
  fprintf(pFile, "figure;\n");
  for (i = 0; i < MAX_WORKER; i++) {
    fprintf(pFile, "worker%d = { \n", i);
    cj_Object *event = profile.worker_timeline[i]->dqueue->head;
    while (event) {
      if (event->event->evetype == CJ_EVENT_TASK_RUN_BEG) {
        if (event->event->beg != event->event->end)
          fprintf(pFile, "'TASK_RUN' [1 0 0] [%f %f]\n", event->event->beg/1000, event->event->end/1000);
      }
      else if (event->event->evetype == CJ_EVENT_FETCH_BEG) {
        if (event->event->beg != event->event->end)
          fprintf(pFile, "'FETCH' [0 1 0] [%f %f]\n", event->event->beg/1000, event->event->end/1000);
      }
      event = event->next;
    }
    fprintf(pFile, "};\n");
    fprintf(pFile, "for i = 1:length(worker%d)\n", i);
    fprintf(pFile, "  tmp = worker%d{i, 3};\n", i);
    fprintf(pFile, "  x = [tmp(1, 1) tmp(1, 2) tmp(1, 2) tmp(1, 1)];\n");
    fprintf(pFile, "  if (strcmp(worker%d{i, 1}, 'FETCH'))\n", i);
    fprintf(pFile, "    y = [%d %d %d+0.4 %d+0.4];\n", i, i, i, i);
    fprintf(pFile, "  else\n");
    fprintf(pFile, "    y = [%d-0.4 %d-0.4 %d %d];\n", i, i, i, i);
    fprintf(pFile, "  end\n");
    fprintf(pFile, "  c = worker%d{i, 2};\n", i);
    fprintf(pFile, "  patch(x, y, c);\n");
    fprintf(pFile, "end\n");
    //fprintf(pFile, "text( + 1, %d + 0.2, 'Fetch', 'fontsize', 10);\n", i);
    //fprintf(pFile, "text( + 1, %d - 0.2, 'Run', 'fontsize', 10);\n", i);
  }
  fprintf(pFile, "grid;");

  fclose(pFile);
}
