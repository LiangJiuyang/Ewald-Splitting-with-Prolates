#!/bin/bash -l
#SBATCH --partition=ccm
#SBATCH --constraint=genoa #ib-rome

# Request 1 node(s)
#SBATCH --nodes=1
# Set the number of tasks per node (=MPI ranks)
#SBATCH --ntasks-per-node=96
# Set the number of threads per rank (=OpenMP threads)
#SBATCH --cpus-per-task=1
#SBATCH -e %j.err

# Wall clock limit:
#SBATCH --time=96:40:00

#### OpenMPI Compiler
module load gcc
module load openmpi
module load fftw

export OMP_NUM_THREADS=1

# Note: please change "/mnt/home/jliang/local/gromacs/build-no-genoa/bin/gmx_mpi" as your own path of GROMACS executable file. 
# Reference Solution
#mpiexec -np 1 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi adjust-estat -s ./benchMEM.tpr -o ./benchMEM-REF.tpr -order 12 -fs 0.03 -rtol 1e-8 -rcoul 1
#mpirun --map-by socket:pe=1 -np 96 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi mdrun -ntomp 1 -rcon 0.9 -maxh 0.5 -resethway -noconfout -cpt 15 -s ./benchMEM-REF.tpr -nsteps 20000 -dlb yes

# PME
#mpiexec -np 1 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi adjust-estat -s ./benchMEM.tpr -o ./benchMEM-PME.tpr -order 4 -fs 0.12 -rtol 1e-5 -rcoul 1 
#mpirun --map-by socket:pe=1 -np 96 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi mdrun -ntomp 1 -rcon 0.9 -maxh 0.5 -resethway -noconfout -cpt 15 -s ./benchMEM-PME.tpr -nsteps 20000 -dlb yes

# ESP
mpiexec -np 1 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi adjust-estat -s ./benchMEM.tpr -o ./benchMEM-ESP.tpr -order 5 -fs 0.2 -rtol 4e-4 -rcoul 7e-1
mpirun --map-by socket:pe=1 -np 96 /mnt/home/jliang/ceph/gromacs/build-genoa/bin/gmx_mpi mdrun -ntomp 1 -rcon 0.9 -maxh 0.5 -resethway -noconfout -cpt 15 -s ./benchMEM-ESP.tpr -nsteps 20000 -dlb yes
