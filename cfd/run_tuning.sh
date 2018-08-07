#!/bin/bash

# I. outside loops: "O" + {op}
#define OSQR
#define ODIV
#define OADD
#define OMUL  # AD = 2
outside='OSQR ODIV OADD OMUL'

# II. inside loops: "L" + {op} + {loop_id}
# Loop 1
#define LDIV1
#define LMUL1
#define LADD1
#define LSQR1
loop_1='LDIV1 LMUL1 LADD1 LSQR1'
# Loop 3
#define LSQR3
loop_3='LSQR3'
# Loop 5
#define LDIV5
#define LMUL5
#define LADD5
loop_5='LDIV5 LMUL5 LADD5'
# Loop 4
#define LSQR4
#define LMUL4
#define LADD4 # AD = 1
loop_4='LSQR4 LMUL4 LADD4'

# III. shared functions: "F" + {op} + {func_id}
# Func 5
#define FADD5
#define FMUL5
func_5='FADD5 FMUL5'
# Func 1
#define FDIV1
func_1='FDIV1'
# Func 4
#define FSQR4
#define FDIV4
#define FMUL4
func_4='FSQR4 FDIV4 FMUL4'
# Func 2
#define FMUL2
#define FADD2 # AD = 1
func_2='FMUL2 FADD2'
# Func 3 
#define FADD3
#define FMUL3 # AD = 2
func_3='FADD3 FMUL3'


# Example: 
# make euler3d_cpu_tuned.ex ITER='-DOSQR -DODIV'
# echo -e '\n'

exe="euler3d_cpu_tuned.ex"
data_file="./data/fvcorr.domn.097K"
num_steps="2048"

dens_out="density_cmp.txt"
mome_out="momentum_cmp.txt"
ener_out="density_energy_cmp.txt"

base_name='_tuned_'
result_dir='./results_07-23-18/'

all_iters=$outside
all_iters+=' '$loop_1' '$loop_3' '$loop_5' '$loop_4
all_iters+=' '$func_5' '$func_1' '$func_4' '$func_2' '$func_3

touch cfd_timing.txt
counter=1
tuning_flags=''
for iter in $all_iters
do
  echo ' '
  tuning_flags+='-D'$iter' '
  echo $tuning_flags
  
  make clean
  make $exe ITER="$tuning_flags"

  # new_dir_name=$counter$base_name$iter
  # new_loc=$result_dir$new_dir_name
  # echo $new_loc
  # mkdir -p $new_loc

  ./$exe $data_file $num_steps >> cfd_timing.txt
  
  # mv $dens_out $new_loc
  # mv $mome_out $new_loc
  # mv $ener_out $new_loc

  ((counter++))
done
