#!/bin/bash -l
#SBATCH --job-name test_small
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=96
#SBATCH --time=01:00:00
#SBATCH --mail-user=llu@flatironinstitute.org
#SBATCH --output=strong_pme_p12_%A_%a.out
#SBATCH --partition=preempt
#SBATCH --qos=preempt
######SBATCH --partition=ccm
#SBATCH --constraint=ib-genoa

SRCDIR=/mnt/ceph/users/$USER/dev/gromacs/build/bin

cd $SRCDIR
module load gcc openmpi fftw

RUNDIR=$SLURM_SUBMIT_DIR/strong-pme-p12-$SLURM_NNODES-$SLURM_NTASKS-$SLURM_ARRAY_TASK_ID
mkdir -p $RUNDIR

cp $SRCDIR/nvt-pme-9cut-p12-1.5e-5-single.tpr $RUNDIR
cp $SRCDIR/nvt-pme-9cut-p12-1.5e-5.mdp $RUNDIR
cd $RUNDIR
export OMP_NUM_THREADS=1

echo 
echo "Job starts: $(date)"
echo "Hostname: $(hostname)"
echo

#mpiexec --map-by socket:pe=1 --use-hwthread-cpus $SRCDIR/gmx_mpi_d mdrun -dlb no -deffnm nvt-pme-v -npme 0 -notunepme
##-dd 72 40 1
#mpiexec --map-by socket:pe=1 --use-hwthread-cpus $SRCDIR/gmx_mpi mdrun -dlb no -deffnm nvt-pme-v -npme 0 -notunepme
mpiexec --map-by socket:pe=1 $SRCDIR/gmx_mpi mdrun -dlb no -deffnm nvt-pme-9cut-p12-1.5e-5-single -npme 0 -notunepme -cpt -1

exe_status=$?;

echo "Job ends: $(date) exe_status: $(exe_status)"

exit $exe_status
