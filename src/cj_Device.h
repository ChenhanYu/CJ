/*
 * cj_Device.h
 * The header file for cj_Device.c
 * Implement the software cache for the GPU device
 */
void cj_Cache_read_in (cj_Device*, int, cj_Object*);
void cj_Cache_write_back (cj_Device*, int, cj_Object*);
void cj_Cache_async_write_back (cj_Device*, int, cj_Object*);
int cj_Cache_fetch (cj_Device*, cj_Object*);
void cj_Cache_sync (cj_Device*);
void cj_Device_sync (cj_Device*);

cj_Device *cj_Device_new(cj_devType, int);
void cj_Device_bind(cj_Worker*, cj_Device*);

/* memcpy from device to host */
void cj_Device_memcpy_d2h (char*, uintptr_t, size_t, cj_Device*);
/* memcpy from host to devide */
void cj_Device_memcpy_h2d (uintptr_t, char*, size_t, cj_Device*);
void cj_Device_memcpy2d_d2h (char*, size_t, uintptr_t, size_t, size_t, size_t, cj_Device*);
void cj_Device_async_memcpy2d_d2h (char*, size_t, uintptr_t, size_t, size_t, size_t, cj_Device*);
void cj_Device_memcpy2d_h2d (uintptr_t, size_t, char*, size_t, size_t, size_t, cj_Device*);
