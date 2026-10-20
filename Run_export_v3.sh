#!/bin/bash -l
# Run_export_v3.sh — the v3 product: all four storm classes, 1980-2024, from the final retrain
# on the new AR and ETC labels (H_final_e300), into
#   Global_ARs_Hurr_ETCs_MCS_finalized_v3/{ARs,Hurricanes,ETCs,MCS}_WNet_final
#
# Two passes on purpose, into the same output directory:
#   1. Hurricane, ETC and MCS from best.pth      (epoch 219, chosen on 4-class skill vs the floors)
#   2. AR from best_ar.pth                       (epoch 195, chosen on AR IoU alone: 0.7525 vs
#                                                 0.747 at epoch 219 — the AR head is the point
#                                                 of this retrain, so it gets its own checkpoint)
# The passes do not overlap, so the resumable "skip months that already exist" logic is safe.
#
# 16 nodes / 128 GPUs; a single class took 2-10 min, so four classes fit the 25-minute limit.
# Resumable: rerun to fill in anything a walltime kill left behind.
#
#   sbatch Run_export_v3.sh
#   sbatch --export=ALL,TAG=H_final_e300_single,OUT=<dir> Run_export_v3.sh   # the single-job control

#SBATCH --account=CLI138
#SBATCH -J exp_v3
#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/exp-v3-%j.txt
#SBATCH -e logs/exp-v3-err-%j.txt
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
TAG="${TAG:-H_final_e300}"
CK="ARs_Hurricanes_ETCs_MCS/Phase2/$TAG"
OUT="${OUT:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v3}"
YEARS=$(seq 1980 2024 | tr '\n' ' ')
for f in "$CK/best.pth" "$CK/best_ar.pth"; do [ -s "$f" ] || { echo "ERROR: $f missing"; exit 1; }; done
echo "run            : $TAG"
echo "outdir         : $OUT"
echo "Hurricane/ETC/MCS <- $CK/best.pth ; AR <- $CK/best_ar.pth"
echo "started        : $(date)"

echo "=== pass 1: Hurricane, ETC, MCS from best.pth ==="
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK/best.pth" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane ETC MCS --subdirs --outdir "$OUT"
RC1=$?; echo "pass 1 exit $RC1"

echo "=== pass 2: AR from best_ar.pth ==="
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK/best_ar.pth" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes AR --subdirs --outdir "$OUT"
RC2=$?; echo "pass 2 exit $RC2"

echo "=== files written (540 expected per class) ==="
for d in ARs Hurricanes ETCs MCS; do echo "  ${d}_WNet_final: $(ls $OUT/${d}_WNet_final 2>/dev/null | wc -l)/540"; done
du -sh "$OUT" 2>/dev/null
echo "Finished $(date)"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; exit $?
