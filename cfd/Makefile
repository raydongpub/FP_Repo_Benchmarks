# NOTE: we set the OMP_NUM_THREADS as an environment variable before calling "make"
# OMP_NUM_THREADS=8

TUN=euler3d_cpu_tuned
# TUN=euler3d_cpu_mixed_tuned

GMP=euler3d_cpu_gmp
DBL=euler3d_cpu_double
# MIX=euler3d_cpu_mixed

LIBS=-lm -lmpfr -lgmp
FLAG=-g -Wall -O3 -fopenmp -Dblock_length=$(OMP_NUM_THREADS)
CAMP=2
ITER=-DTEMP_PLEASE_OVERRIDE

all: $(TUN).ex $(GMP).ex $(DBL).ex #$(MIX).ex 

$(TUN).ex: $(TUN).cpp
	g++ $(FLAG) $(TUN).cpp -o $(TUN).ex $(LIBS) -DCPR=$(CAMP) $(ITER)
#-DTUNE$(ITER)

$(GMP).ex: $(GMP).o read_file.o
	g++ $(FLAG) $(GMP).o read_file.o -o $(GMP).ex $(LIBS)

$(GMP).o: $(GMP).c
	gcc $(FLAG) -c $(GMP).c -o $(GMP).o

read_file.o: read_file.cpp
	g++ -c read_file.cpp -o read_file.o

$(DBL).ex: $(DBL).cpp
	g++ $(FLAG) $(DBL).cpp -o $(DBL).ex

# $(MIX).ex: $(MIX).cpp
# 	g++ $(FLAG) $(MIX).cpp -o $(MIX).ex $(LIBS) -DCPR=$(CAMP)

clean:
	rm -f *.ex *.o
