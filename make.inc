CC             = gcc
NVCC           = nvcc

ARCH           = ar
ARCHFLAGS      = cr
RANLIB         = ranlib

CFLAGS         = -openmp -O2
NVCCFLAGS      = -O2 -arch sm_35

LIBCJ          = ../lib/libcj.a

INC            = -I../include
LIB            = $(LIBCJ)
LDFLAGS        = $(INC) $(LIB)
