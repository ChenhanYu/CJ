CC             = icc
#CC             = gcc
NVCC           = nvcc

ARCH           = ar
ARCHFLAGS      = cr
RANLIB         = ranlib

CFLAGS         = -openmp -O2 -DCJ_HAVE_CUDA
#CFLAGS         = -openmp -O2
NVCCFLAGS      = -O2 -arch sm_35

CUDA_DIR       = /opt/NVIDIA/cuda
#CUDA_DIR       = /Developer/NVIDIA/CUDA-5.5

LIBCJ          = ../lib/libcj.a 

INC            = -I../include -I$(CUDA_DIR)/include
#INC            = -I../include

LIB            = $(LIBCJ) -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lpthread -lm -mkl=sequential 
#LIB            = $(LIBCJ) -L$(CUDA_DIR)/lib -lcudart -lcublas -lblas
#LIB            = $(LIBCJ) -lpthread -lm -mkl=sequential
#LIB            = $(LIBCJ) -lpthread -lblas

LDFLAGS        = $(INC) $(LIB)
