CJ_DIR = ..
include ../make.inc

D_CC_SRC = test_gemm.c test_syrk.c test_cache.c test_trsm.c test_chol.c

D_CC_EXE = $(D_CC_SRC:.c=.x)

all: $(D_CC_EXE)

%.x : %.c $(LIBCJ)
	$(CC) $(CFLAGS) $< -o $@ $(INC) $(LIB)

clean:
	rm -f *.x *~
