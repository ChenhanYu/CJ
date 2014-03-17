CC             = gcc
NVCC           = nvcc

ARCH           = ar
ARCHFLAGS      = cr
RANLIB         = ranlib

CFLAGS         = -openmp -O2
NVCCFLAGS      = -O2 -arch sm_35

CUDA_DIR       = /usr/local/cuda

LIBCJ          = ../lib/libcj.a 

INC            = -I../include -I$(CUDA_DIR)/include

LIB            = $(LIBCJ) -L$(CUDA_DIR)/lib -lcudart -lcublas
LDFLAGS        = $(INC) $(LIB)
