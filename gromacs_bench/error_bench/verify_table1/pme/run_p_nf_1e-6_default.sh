#!/bin/bash


export OMP_NUM_THREADS=1

for p in {5..5}
do
    #for i in {0..25}
    #for i in {26..50}
    for i in {51..75}
    do
      nf=$((540+$i*4))
      echo "p = $p, nf = $nf"
      rm -rf run_Li_1e-6_p_${p}_nf_${nf}_default
      mkdir run_Li_1e-6_p_${p}_nf_${nf}_default
      cd run_Li_1e-6_p_${p}_nf_${nf}_default
      cp ../nvt-pme-p-default-1e-6.mdp .
      sed -i -e "s/188/${nf}/g" nvt-pme-p-default-1e-6.mdp
      sed -i -e "s/pme_order = 11/pme_order = ${p}/g" nvt-pme-p-default-1e-6.mdp
      mpiexec -np 1 ../../../gmx_mpi_d grompp -f ./nvt-pme-p-default-1e-6.mdp -c /Users/libin/git/tmp/gromacs/build/bin/data/Li_frame_test_50.gro -p ../../pswf64.top -o nvt-pme-default.tpr #> /dev/null 2>&1
      mpiexec -np 1 ../../../gmx_mpi_d mdrun -deffnm nvt-pme-default -npme 0 #> /dev/null 2>&1
      cd ..
    done
done
