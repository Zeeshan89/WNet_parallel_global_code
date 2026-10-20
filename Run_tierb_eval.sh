#!/bin/bash -l
# Run_tierb_eval.sh — did the Tier B label change actually improve TC against
# IBTrACS? Scores blob centroids directly from cache + checkpoint, so no export or
# tracking step is needed for the verdict.
#
# Two runs, each pairing a checkpoint with the cache it was TRAINED on:
#   C_full200  + tensor_cache_v3_split   old label (wind OR vorticity)
#   D_tierb200 + tensor_cache_v4_split   new label (wind AND vorticity)
#
# Pairing matters: phase4_evaluate_tc.py also reports the heuristic mask from the
# cache as a reference row, so each run shows its own teacher alongside its model.
# Crossing them would compare a model against a label it never saw.
#
# --te-domain restricts every detector to the TempestExtremes box, the only fair
# ground for the competitor row (its TC trajectory file spans lat 10..46 only).
#
#SBATCH --account=CLI138
#SBATCH -J tierb_eval
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/tierbeval-%j.txt
#SBATCH -e logs/tierbeval-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
YEARS="${YEARS:-1983 1994 2003 2012 2021}"

echo "############ OLD LABEL — C_full200 on v3 ############"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_evaluate_tc.py \
    --checkpoint ARs_Hurricanes_ETCs_MCS/Phase2/C_full200/best.pth \
    --cache-dir "$NB/tensor_cache_v3_split" \
    --years $YEARS --te-domain \
    --outdir phase4_results/tierb_old

echo
echo "############ NEW LABEL — D_tierb200 on v4 ############"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_evaluate_tc.py \
    --checkpoint ARs_Hurricanes_ETCs_MCS/Phase2/D_tierb200/best.pth \
    --cache-dir "$NB/tensor_cache_v4_split" \
    --years $YEARS --te-domain \
    --outdir phase4_results/tierb_new

echo "exit $?"
