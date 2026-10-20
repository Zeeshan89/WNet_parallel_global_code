#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J mcs_ids
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH -o logs/mcsids-%j.txt
#SBATCH -e logs/mcsids-err-%j.txt
#SBATCH --export=NONE
#
# Add the MCS_id storm-track variable to the exported MCS files (phase4_add_mcs_ids.py), so each
# file carries the mask AND the identity of each system, with the same tracking parameters as the
# published MCS evaluation (overlap 0.5, no area or lifetime pruning). MCS_detection is not
# touched. Files that already have MCS_id are skipped, so the job is resumable.
#
#   sbatch --dependency=afterok:<v3 export> Run_add_mcs_ids.sh
#   sbatch --export=ALL,DIR=<other MCS dir> Run_add_mcs_ids.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
DIR="${DIR:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v3/MCS_WNet_final}"
n=$(ls $DIR/*_MCS_detection.nc 2>/dev/null | wc -l)
echo "MCS files in $DIR: $n; started $(date)"
[ "$n" -ge 1 ] || { echo "ERROR: no MCS files"; exit 1; }

srun -n $SLURM_NTASKS -c 1 --export=ALL $(which python3) -W ignore -u ./phase4_add_mcs_ids.py --dir "$DIR"
RC=$?
echo "exit $RC  $(date)"
srun -n 1 -c 1 --export=ALL $(which python3) -W ignore -u ./phase4_add_mcs_ids.py --dir "$DIR" --check
exit $RC
