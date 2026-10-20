#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ivt_pctl
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o logs/ivtpctl-%j.txt
#SBATCH -e logs/ivtpctl-err-%j.txt
#SBATCH --export=NONE
#
# Per-grid-cell, per-calendar-month IVT percentiles (85, 87.5, 90, 92.5, 95) for ERA5
# 1980-2024: the locally relative threshold of the literature-based AR label. One node on
# purpose (small jobs start quickly): 24 ranks = 12 calendar months x 2 year chunks, 22 or 23
# monthly files each, about 5 GB per rank. Finished pieces are skipped, so the job can be
# resubmitted if it runs out of time. Histograms (about 30 GB) stay in
# ivt_only/ivt_pctl_monthly/ until the result has been checked; delete them afterwards.
#
#   sbatch Run_ivt_percentiles.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2

PY="$(which python3) -W ignore -u ./phase4_ivt_percentiles.py"
srun -n $SLURM_NTASKS -c 2 --export=ALL $PY --stage hist      || exit 1
srun -n 13 -c 4 --export=ALL            $PY --stage merge     || exit 1
srun -n 1 -c 4 --export=ALL             $PY --stage assemble
echo "exit $?"
