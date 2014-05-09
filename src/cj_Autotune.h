/*
 * cj_Autotune.h
 * The header file for cj_Autotune.c
 * Create the auto-tune information (the GEMM calculation time for a block) for the CPU and GPU
 * If auto-tuner is run first time, the CPU and GPU information will be stored as cj_autotune.bin under the ../test folder;
 * Otherwise, the CPU and GPU information will be directly read from the "../test/cj_autotune.bin", which will save some time.
 */
void cj_Autotune_init ();
cj_Autotune *cj_Autotune_get_ptr ();
