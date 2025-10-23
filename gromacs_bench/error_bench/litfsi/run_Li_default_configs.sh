#!/bin/bash


export OMP_NUM_THREADS=1
rm -rf run_Li_config_default_*

for i in {1..100}
do
    echo "Running experiment $i"
    rm -rf run_Li_config_default_$i
    mkdir run_Li_config_default_$i
    cd run_Li_config_default_$i
    mpiexec -np 1 gmx_mpi_d grompp -f ../nvt-pme-default.mdp -c ../configs/Li_frame_test_$i.gro -p ../pswf64.top -o nvt-pme-default.tpr #> /dev/null 2>&1
    mpiexec -np 1 gmx_mpi_d mdrun -deffnm nvt-pme-default -npme 0 #> /dev/null 2>&1
    cd ..
done
