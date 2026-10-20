#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J etc_miss
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o logs/etcmiss-%j.txt
#SBATCH -e logs/etcmiss-err-%j.txt
#SBATCH --export=NONE
#
# Why does WNet miss NAECC cyclones? One rank per year (2000-2010, the years of the tracker
# sweep); every NAECC cyclone is assigned to the pipeline stage that lost it (mask, centre,
# track, fragment) and the shipped tracker rules are applied to the NAECC tracks themselves.
# One node on purpose. Finished years are skipped, so the job can be resubmitted.
#
#   sbatch Run_etc_miss_attribution.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0       # as in Run_etc_tracks_production.sh

PY="$(which python3) -W ignore -u ./phase4_etc_miss_attribution.py"
srun -n $SLURM_NTASKS -c 4 --export=ALL $PY --years 2000 2010 || exit 1
srun -n 1 -c 4 --export=ALL $PY --summarize
echo "exit $?"
