#!/bin/bash -l
# Run_export_etc_c200.sh — re-export the SHIPPED ETC masks (C_full200/best.pth, the model behind
# Global_ARs_Hurr_ETCs_MCS_finalized_v2), whose .nc files were deleted after tracking.
#
# The tracker sweeps ran on the Global_pw50_e200 masks because those were the only ETC masks
# left on disk. Their shipped-rules baseline (S0: NAEC POD 0.560, FAR 0.281) reproduces the
# C_full200 result (variant D: 0.565, 0.279), so the masks are nearly interchangeable for
# tracking; but the paper product is the C_full200 model, so the winning tracker configuration
# should be run on ITS masks, and only the tracker changes in the paper's story.
# 16 nodes / 128 GPUs, about 15 min; resumable.
#
#   sbatch Run_export_etc_c200.sh

#SBATCH --account=CLI138
#SBATCH -J exp_etc_c200
#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/exp-etc-c200-%j.txt
#SBATCH -e logs/exp-etc-c200-err-%j.txt
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
CK=${CK:-ARs_Hurricanes_ETCs_MCS/Phase2/C_full200/best.pth}
OUT=${OUT:-$NB/Global_C200_etc}
YEARS=$(seq 1980 2024 | tr '\n' ' ')
[ -s "$CK" ] || { echo "ERROR: checkpoint $CK missing"; exit 1; }
echo "ETC checkpoint : $CK"; echo "outdir         : $OUT/ETCs_WNet_final"; echo "started        : $(date)"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes ETC --subdirs --outdir "$OUT"
RC=$?
echo "ETC files: $(ls $OUT/ETCs_WNet_final 2>/dev/null | wc -l)/540   exit $RC   $(date)"
exit $RC
