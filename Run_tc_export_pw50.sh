#!/bin/bash -l
# Run_tc_export_final.sh — full-record TC detections from the IBTrACS-target model.
# Writes a NEW directory rather than overwriting Hurricanes_WNet_final, so the
# C_full200 product stays intact until the swap is a deliberate choice.
#SBATCH --account=CLI138
#SBATCH -J tc_exp_pw50
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/tcexport-pw50-%j.txt
#SBATCH -e logs/tcexport-pw50-err-%j.txt
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
CKPT=${CKPT:-ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50/best_hurricane.pth}
OUT=${OUT:-$NB/Global_TC_pw50}
YEARS=$(seq 1980 2024 | tr '\n' ' ')
if [ ! -s "$CKPT" ]; then echo "ERROR: $CKPT missing"; exit 1; fi
echo "checkpoint : $CKPT"; echo "cache      : $NB/tensor_cache_v5_ibtracs"; echo "outdir     : $OUT"
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane --subdirs --outdir "$OUT"
RC=$?; echo "exit $RC"; du -sh "$OUT" 2>/dev/null
echo "files: $(ls $OUT/Hurricanes_WNet_final 2>/dev/null | wc -l)/540"; exit $RC
