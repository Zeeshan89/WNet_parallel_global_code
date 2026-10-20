#!/bin/bash -l
# Run_export_smooth.sh — AR and MCS re-export with smooth upsampling (--upsample bilinear):
# same checkpoints and caches as the shipped v2 files, only the 0.94 deg -> 0.25 deg step differs.
#   KIND=6h      AR (H_final_e300/best_ar.pth) + MCS (best.pth), tensor_cache_v5_ibtracs
#   KIND=hourly  MCS (best.pth), tensor_cache_hourly
# The write (zlib) is CPU-bound, so 3 ranks share each GPU (24 per node). Resumable: months that
# exist are skipped, so a job that hits the wall is simply resubmitted.
#   sbatch -N 2 --export=ALL,KIND=hourly,YEARS="1980 ... 1994" Run_export_smooth.sh
#SBATCH --account=CLI138
#SBATCH -J exp_smooth
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH -o logs/expsmooth-%j.txt
#SBATCH -e logs/expsmooth-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CK=ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300
: "${KIND:?}"
YEARS="${YEARS:-$(seq 1980 2024 | tr '\n' ' ')}"
N=$((SLURM_JOB_NUM_NODES * 24))
run() {   # $1 checkpoint  $2 class  $3 cache  $4 outdir
  srun -N $SLURM_JOB_NUM_NODES -n $N -c 2 --gpus-per-node=8 --export=ALL \
    $(which python3) -W ignore -u ./phase4_export_detections.py --checkpoint "$1" --classes $2 \
      --cache-dir "$3" --years $YEARS --subdirs --outdir "$4" --upsample bilinear | grep -v "steps ->"
}
echo "KIND=$KIND years: $YEARS  ranks=$N  $(date)"
if [ "$KIND" = 6h ]; then
  OUT=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2_smooth
  run $CK/best.pth MCS $NB/tensor_cache_v5_ibtracs $OUT
  run $CK/best_ar.pth AR $NB/tensor_cache_v5_ibtracs $OUT
  echo "files: AR $(ls $OUT/ARs_WNet_final | wc -l)/540  MCS $(ls $OUT/MCS_WNet_final | wc -l)/540"
else
  OUT=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2_hourly_smooth
  run $CK/best.pth MCS $NB/tensor_cache_hourly $OUT
  echo "files: MCS $(ls $OUT/MCS_WNet_final | wc -l)/540"
fi
echo "done $(date)"
