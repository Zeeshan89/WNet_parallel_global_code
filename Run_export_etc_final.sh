#!/bin/bash -l
# Run_export_etc_final.sh — full-record ETC detections from one checkpoint (default: the final
# retrain's best.pth) into <OUT>/ETCs_WNet_final. 16 nodes / 128 GPUs, about 15 min; resumable.
#
#   sbatch --export=ALL,CK_ETC=<ckpt>,OUT=<dir> Run_export_etc_final.sh

#SBATCH --account=CLI138
#SBATCH -J exp_etc_final
#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/exp-etc-final-%j.txt
#SBATCH -e logs/exp-etc-final-err-%j.txt
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
CK_ETC=${CK_ETC:-ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300/best.pth}
OUT=${OUT:-$NB/Global_final}
YEARS=$(seq 1980 2024 | tr '\n' ' ')
[ -s "$CK_ETC" ] || { echo "ERROR: checkpoint $CK_ETC missing"; exit 1; }
echo "ETC checkpoint : $CK_ETC"; echo "outdir         : $OUT/ETCs_WNet_final"; echo "started        : $(date)"
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK_ETC" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes ETC --subdirs --outdir "$OUT"
RC=$?
echo "ETC files: $(ls $OUT/ETCs_WNet_final 2>/dev/null | wc -l)/540   exit $RC   $(date)"
exit $RC
