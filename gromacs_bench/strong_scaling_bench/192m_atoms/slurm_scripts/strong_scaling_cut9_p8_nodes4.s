#!/bin/bash -l
#SBATCH --job-name test_small
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=96
#SBATCH --time=48:00:00
#SBATCH --mail-user=llu@flatironinstitute.org
#SBATCH --output=strong_pswfcut9_p8_%A_%a.out
#SBATCH --partition=ccm
#SBATCH --constraint=ib-genoa

SRCDIR=/mnt/ceph/users/$USER/dev/gromacs_pswf/build/bin

cd $SRCDIR
module load gcc openmpi fftw

RUNDIR=$SLURM_SUBMIT_DIR/strong-pswfcut9-p8-$SLURM_NNODES-$SLURM_NTASKS-$SLURM_ARRAY_TASK_ID
mkdir -p $RUNDIR

#cp $SRCDIR/nvt-pme-9cut-p8-1.5e-5-64m-single.tpr $RUNDIR
cp $SRCDIR/nvt-pme-9cut-p8-1.5e-5-64m.mdp $RUNDIR
cd $RUNDIR
export OMP_NUM_THREADS=1

echo 
echo "Job starts: $(date)"
echo "Hostname: $(hostname)"
echo

#mpiexec --map-by socket:pe=1 --use-hwthread-cpus $SRCDIR/gmx_mpi_d mdrun -dlb no -deffnm nvt-pme-v -npme 0 -notunepme
##-dd 72 40 1
#mpiexec --map-by socket:pe=1 --use-hwthread-cpus $SRCDIR/gmx_mpi mdrun -dlb no -deffnm nvt-pme-v -npme 0 -notunepme
mpiexec --map-by socket:pe=1 $SRCDIR/gmx_mpi mdrun -s $SRCDIR/nvt-pme-9cut-p8-1.5e-5-64m-single.tpr -dlb no -deffnm nvt-pme-9cut-p8-1.5e-5-64m-single -npme 0 -notunepme -cpt -1

exe_status=$?;

echo "Job ends: $(date) exe_status: $(exe_status)"

exit $exe_status
