#!/bin/bash -l
# Run_export_hourly_mcs.sh — HOURLY MCS detections, 1980-2024.
#
# Same checkpoint, same script, same weights as the 6-hourly export. The model is
# a stateless per-frame 2D CNN (WNet_model.py:178-190), so cadence is set purely by
# the cache handed to --cache-dir; nothing here is retrained or re-tuned.
# phase4_export_detections.py needs no modification: it reads split_{YYYYMM}.npy,
# takes nT from arr.shape[0] and the time axis from the .meta.npy, so a 744-step
# month exports as 744 steps.
#
# Requires the hourly cache from Run_hourly_split_cache.sh.
#
# Resumable: existing months are skipped unless --overwrite, so resubmit on a wall
# hit. The 6-hourly 4-class export of the whole record took 8m43s on these 8 nodes;
# this is 6x the steps but one class instead of four, so 45 min is generous.
#
# Submit:
#   sbatch Run_export_hourly_mcs.sh
#   sbatch --export=YEARS="2003 2012" Run_export_hourly_mcs.sh    # subset

#SBATCH --account=CLI138
#SBATCH -J wnet_exp_hrly
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/exphourly-%j.txt
#SBATCH -e logs/exphourly-err-%j.txt
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
CACHE="${CACHE:-$NB/tensor_cache_hourly}"
OUT="${OUT:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2_hourly}"
YEARS="${YEARS:-$(seq 1980 2024 | tr '\n' ' ')}"
# Same checkpoint as the 6-hourly export, so the only difference between the two
# products is cadence. C_full200 (best.pth = epoch 157) supersedes C_full80.
CKPT="${CKPT:-ARs_Hurricanes_ETCs_MCS/Phase2/C_full200/best.pth}"

echo "Hourly MCS export"
echo "  checkpoint : $CKPT"
echo "  cache      : $CACHE"
echo "  out        : $OUT"
echo "  years      : $YEARS"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" \
    --cache-dir "$CACHE" \
    --years $YEARS --classes MCS \
    --subdirs --outdir "$OUT"

RC=$?; echo "exit $RC"; du -sh "$OUT" 2>/dev/null
echo "files: $(ls $OUT/MCS_WNet_final 2>/dev/null | wc -l)/540"
exit $RC
