#!/bin/bash

export OMP_NUM_THREADS=1
rm -rf run_8000_pme_default_config_*
for i in {1..100}
do
    echo "Running experiment $i"
    mkdir run_8000_pme_default_config_$i
    cd run_8000_pme_default_config_$i
    mpiexec -np 1 gmx_mpi_d grompp -f ../pme-default.mdp -c ../configs/zero_vel_frame_test_$i.gro -p ../topol.top -o nvt-pme-default.tpr > /dev/null 2>&1
    mpiexec -np 1 gmx_mpi_d mdrun -deffnm nvt-pme-default -npme 0 > /dev/null 2>&1
    cd ..
done
