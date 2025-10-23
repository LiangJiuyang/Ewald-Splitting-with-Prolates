#!/bin/bash


export OMP_NUM_THREADS=1

for p in {6..10}
do
    for i in {0..15}
    do
      nf=$((124+$i*2))
      echo "p = $p, nf = $nf"
      rm -rf run_Li_5e-4_p_${p}_nf_${nf}
      mkdir run_Li_5e-4_p_${p}_nf_${nf}
      cd run_Li_5e-4_p_${p}_nf_${nf}
      cp ../nvt-pme-p-optimal-5e-4.mdp .
      sed -i -e "s/188/${nf}/g" nvt-pme-p-optimal-5e-4.mdp
      sed -i -e "s/pme_order = 11/pme_order = ${p}/g" nvt-pme-p-optimal-5e-4.mdp
      mpiexec -np 1 ../../../gmx_mpi_d grompp -f ./nvt-pme-p-optimal-5e-4.mdp -c /Users/libin/git/tmp/gromacs/build/bin/data/Li_frame_test_50.gro -p ../../pswf64.top -o nvt-pme-optimal.tpr #> /dev/null 2>&1
      mpiexec -np 1 ../../../gmx_mpi_d mdrun -deffnm nvt-pme-optimal -npme 0 #> /dev/null 2>&1
      cd ..
    done
done
