#!/bin/bash


export OMP_NUM_THREADS=1
#rm -rf run_Membrane_config_optimal_*

for i in {1..100}
do
    echo "Running experiment $i"
    rm -rf run_Membrane_config_optimal_$i
    mkdir run_Membrane_config_optimal_$i
    cd run_Membrane_config_optimal_$i
    mpiexec -np 1 gmx_mpi_d grompp -f ../pme-optimal.mdp -c ../configs/Membrane_frame_test_$i.gro -p ../topol.top -n ../atomistic-system.ndx -o nvt-pme-optimal.tpr -maxwarn 3 #> /dev/null 2>&1
    mpiexec -np 1 gmx_mpi_d mdrun -deffnm nvt-pme-optimal -npme 0 #> /dev/null 2>&1
    cd ..
done
