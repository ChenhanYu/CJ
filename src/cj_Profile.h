/*
 * cj_Profile.h
 * The header file for cj_Profile.c
 * Profiling information, which is generated in the run-time.
 * cj_Event: Record the event and related time in the run-time.
 * cj_Profile: Record the profiling information.
 */

cj_Event *cj_Event_new ();
void cj_Profile_worker_record (cj_Worker*, cj_eveType);
void cj_Profile_init ();
void cj_Profile_output_timeline ();
