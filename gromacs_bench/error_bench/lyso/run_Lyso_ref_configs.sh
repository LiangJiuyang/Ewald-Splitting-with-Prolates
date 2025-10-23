#!/bin/bash


export OMP_NUM_THREADS=1

for i in {1..100}
do
    echo "Running experiment $i"
    rm -rf run_Lyso_config_ref_$i
    mkdir run_Lyso_config_ref_$i
    cd run_Lyso_config_ref_$i
    mpiexec -np 1 gmx_mpi_d grompp -f ../pme-ref.mdp -c ../configs/Lyso_frame_test_$i.gro -p ../topol27.top -o nvt-pme-ref.tpr #> /dev/null 2>&1
    mpiexec -np 1 gmx_mpi_d mdrun -deffnm nvt-pme-ref -npme 0 #> /dev/null 2>&1
    cd ..
done
