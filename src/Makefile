CJ_DIR = ..
include ../make.inc

D_CC_SRC = cj.c \
		   cj_Object.c \
		   cj_Graph.c \
		   cj_Blas.c \
		   cj_Sparse.c \
		   cj_Device.c \
		   cj_Autotune.c \
			 cj_Profile.c

D_CC_OBJ = $(D_CC_SRC:.c=.o)

all: $(LIBCJ) clean

$(LIBCJ): $(D_CC_OBJ) 
	$(ARCH) $(ARCHFLAGS) $@ $(D_CC_OBJ) $(D_CU_OBJ)
	$(RANLIB) $@

%.o : %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean:
		rm -f *.cu_o *.o *~ *.linkinfo
