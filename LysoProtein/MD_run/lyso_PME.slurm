#!/bin/bash
#SBATCH --job-name=PME_lyso
#SBATCH --partition=ccm
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=end
#SBATCH --mail-user=tpan1039@sjtu.edu.cn

module load gcc
module load openmpi
module load fftw

ulimit -s unlimited
ulimit -l unlimited


mpiexec --map-by socket:pe=1 -np 1 /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/gmx_mpi grompp -f ../pme.mdp -c /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/plumed-spike-omicron/mw-wtmeta-omi/mini.gro -p /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/plumed-spike-omicron/mw-wtmeta-omi/topol.top  -o pme.tpr

mpiexec --map-by socket:pe=1 -np 96 /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/gmx_mpi mdrun -deffnm /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/plumed-spike-omicron/mw-wtmeta-omi/dyn -npme 0 -dlb no

#mpiexec --map-by socket:pe=16 -np 1 /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/gmx_mpi genconf -f pme.gro -o pme50.gro -nbox 3 3 3
mpiexec --map-by socket:pe=1 -np 96 /mnt/home/jliang/local/gromacs_new/MD_Example/Pilar/gmx_mpi genconf -f input.gro -o output.gro -nbox 2 2 2

