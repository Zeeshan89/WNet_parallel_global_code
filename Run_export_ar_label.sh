#!/bin/bash -l
# Run_export_ar_label.sh — full-record AR detections from the model retrained on the
# literature-based AR label (best_ar.pth of Run_phase2_ar_label.sh).
#
# Only the AR class is exported: TC / ETC / MCS keep their shipped products. The export
# reads predictors only (channels 0-5), so any cache works; v5 is used because it is the
# one every other export read. 16 nodes / 128 GPUs so the 540 months take about 15 min.
# Resumable: months whose .nc exists are skipped.
#
#   sbatch --dependency=afterok:<train job> Run_export_ar_label.sh
#   sbatch --export=CK=...,OUT=... Run_export_ar_label.sh

#SBATCH --account=CLI138
#SBATCH -J exp_ar_lit
#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/exp-arlit-%j.txt
#SBATCH -e logs/exp-arlit-err-%j.txt
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
P2=ARs_Hurricanes_ETCs_MCS/Phase2
CK=${CK:-$P2/G_arlit_e200/best_ar.pth}
OUT=${OUT:-$NB/Global_arlit}
YEARS=$(seq 1980 2024 | tr '\n' ' ')

[ -s "$CK" ] || { echo "ERROR: checkpoint $CK missing"; exit 1; }
echo "AR checkpoint : $CK"; echo "outdir        : $OUT/ARs_WNet_final"; echo "started       : $(date)"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes AR --subdirs --outdir "$OUT"
RC=$?
echo "AR files: $(ls $OUT/ARs_WNet_final 2>/dev/null | wc -l)/540   exit $RC   $(date)"
exit $RC
