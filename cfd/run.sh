#!/bin/bash

# Define the exe names:
exe1="./euler3d_cpu_double.ex"
# exe2="./euler3d_cpu_mixed.ex"
# exe3="./euler3d_cpu_mixed_tuned.ex"
# exe4="./euler3d_cpu_gmp.ex"
exe5="./euler3d_cpu_tuned.ex"

# Specify a data file:
data_file="./data/fvcorr.domn.097K"
# data_file="./data/fvcorr.domn.193K"
# data_file="./data/missile.domn.0.2M"

# Run the tests:
$exe1 $data_file $1
# $exe2 $data_file $1
# $exe3 $data_file $1
# $exe4 $data_file $1
$exe5 $data_file $1
