#!/bin/bash

#SBATCH --job-name=ccm_liang
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH -C ib-genoa
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=96    
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=
#SBATCH -o PSWF2.out
#SBATCH -e %j.err
#SBATCH --time=40:00:00 
#SBATCH --exclusive

module purge
module load gcc openmpi fftw

ulimit -s unlimited
ulimit -l unlimited

mpiexec --map-by socket:pe=1 -np 1 /mnt/home/jliang/local/gromacs_new/build-genoa/bin/gmx_mpi grompp -f nvt-pme.mdp -c New1000Box.gro -p topol.top -o nvt-pme-v.tpr # .mdp MD setup
mpiexec --map-by socket:pe=1 -np 1 /mnt/home/jliang/local/gromacs_new/build-genoa/bin/gmx_mpi mdrun -npme 0 -dlb yes -maxh 0.15 -resethway -noconfout -cpt 4.5 -s nvt-pme-v.tpr 

mpiexec --map-by socket:pe=1 -np 1 /mnt/home/jliang/ceph/gromacs/build-no-genoa/bin/gmx_mpi_d grompp -f nvt-pme.mdp -c New27000Box.gro -p topol.top -o nvt-pme-v.tpr # .mdp MD setup
mpiexec --map-by socket:pe=1 -np 1 /mnt/home/jliang/local/gromacs_new/build_force_test/bin/gmx_mpi_d mdrun -nsteps 0 -npme 0 -dlb yes -maxh 0.15 -resethway -noconfout -cpt 4.5 -s nvt-pme-v.tpr 

/mnt/home/jliang/local/gromacs_new/build_force_test/bin/gmx_mpi_d   /mnt/home/jliang/ceph/gromacs/build-no-genoa/bin/gmx_mpi_d
