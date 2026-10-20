#!/bin/bash -l
# Run_ibtracs_eval.sh — evaluate the IBTrACS-target model against the old one.
#
# Chained with afterany, NOT afterok: the 100-epoch run is expected to hit its
# 45 min wall around epoch ~78 (30 s/epoch after a ~7 min cold Lustre read), so it
# will end TIMEOUT rather than COMPLETED. best.pth is written on every improvement,
# so a timed-out run still leaves a usable checkpoint — the script checks for it
# and exits cleanly if training genuinely failed.
#
# Two runs, each pairing a checkpoint with the cache it was TRAINED on, because
# phase4_evaluate_tc.py reports the cache's own label as a reference row:
#   C_full200    + v3   CyTRACK heuristic label
#   E_ibtracs100 + v5   IBTrACS observational label
# Crossing them would score a model against a label it never saw.
#
# IBTRACS_WIND_SOURCE=any on BOTH sides. The new target was built from all-agency
# intensities; scoring it against a usa_wind-only reference would count genuine
# W Pacific / N Indian detections as false alarms and penalise the model for the
# basin-coverage bug we just fixed. Applying it to the old model too keeps the
# comparison like-for-like.
#
# Held-out years only. IBTrACS is now a TRAINING signal, so the test years are the
# only honest measure — this is the circularity that made the previous system's
# POD of 0.99 meaningless.
#
#SBATCH --account=CLI138
#SBATCH -J ibt_eval
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:35:00
#SBATCH -o logs/ibteval-%j.txt
#SBATCH -e logs/ibteval-err-%j.txt
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
export IBTRACS_WIND_SOURCE=any

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CKPT_NEW=ARs_Hurricanes_ETCs_MCS/Phase2/E_ibtracs100/best.pth
CKPT_OLD=ARs_Hurricanes_ETCs_MCS/Phase2/C_full200/best.pth
YEARS="${YEARS:-1983 1994 2003 2012 2021}"

if [ ! -s "$CKPT_NEW" ]; then
  echo "ERROR: $CKPT_NEW missing — training produced no checkpoint. Stopping."
  exit 1
fi
echo "checkpoint: $CKPT_NEW ($(stat -c%y "$CKPT_NEW"))"
echo "held-out years: $YEARS   IBTRACS_WIND_SOURCE=$IBTRACS_WIND_SOURCE"

echo "########## NEW — IBTrACS target (E_ibtracs100 + v5) ##########"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_evaluate_tc.py \
    --checkpoint "$CKPT_NEW" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --te-domain --outdir phase4_results/ibt_new

echo
echo "########## OLD — CyTRACK heuristic (C_full200 + v3) ##########"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_evaluate_tc.py \
    --checkpoint "$CKPT_OLD" --cache-dir "$NB/tensor_cache_v3_split" \
    --years $YEARS --te-domain --outdir phase4_results/ibt_old

echo
echo "########## export TC detections from the new model ##########"
srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT_NEW" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane \
    --outdir ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/nc

echo "exit $?"
