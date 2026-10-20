#!/bin/bash -l
# Run_export_pw50_e200.sh — full-record TC *and* ETC detections from the corrected
# 200-epoch run (F_ibtracs_pwH50_e200).
#
# TWO CHECKPOINTS FROM TWO DIFFERENT RUNS, ON PURPOSE
# ---------------------------------------------------
#   Hurricane  <- F_ibtracs_pwH50/best_hurricane.pth       (30-epoch run, epoch 13)
#   ETC        <- F_ibtracs_pwH50_e200/best.pth            (200-epoch run, epoch 163)
#
# Both runs have the corrected per-class pos_weight [8,50,25,15]; they differ only
# in length, and therefore in LR schedule.
#
# Why TC comes from the SHORT run. `--select-class hurricane` maximises TC *IoU*,
# but IoU is not what predicts track POD: the 30-epoch run moved TC IoU only
# 0.5084 -> 0.5161 while track POD moved 0.780 -> 0.850, because RECALL is what
# buys the temporal continuity a track needs. At essentially equal IoU the two
# candidates are:
#     30-epoch  epoch  13 : recall 0.754, precision 0.620, IoU 0.5161  <- chosen
#     200-epoch epoch  36 : recall 0.670, precision 0.690, IoU 0.5150
# and the 30-epoch checkpoint is already measured at POD 0.850 / CSI 0.742.
# Selecting the 200-epoch run on recall instead is not an option: its recall peak
# is epoch 17 at 0.876 but precision 0.385, which would flood the field.
# The useful TC state passes early, while a 200-epoch cosine still has a high LR.
#
# Why ETC comes from the LONG run. The commoner heads need the epochs: the
# 30-epoch run left ETC IoU at 0.599 against 0.760 at epoch 163. Note the long run
# is still slightly below the buggy-era E_ibtracs200 on ETC (0.760 vs 0.792) --
# raising ETC's pos_weight cap 8 -> 25 cost a little pixel IoU. Whether that
# matters for ETC TRACKS is measured by the tracking job, and
# E_ibtracs200/best.pth is the fallback if it does.
#
# Both go into ONE output directory via --subdirs, which writes
# Hurricanes_WNet_final/ and ETCs_WNet_final/ side by side -- the layout
# Hurricane_track.py and updated_monthly_ETC_Tracker.py already glob.
#
# Resumable: phase4_export_detections.py skips months whose .nc already exists,
# so a walltime kill just needs a resubmit.
#
# Submit:  sbatch --dependency=afterok:<train job> Run_export_pw50_e200.sh

#SBATCH --account=CLI138
#SBATCH -J exp_pw50_e200
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:35:00
#SBATCH -o logs/exp-pw50-e200-%j.txt
#SBATCH -e logs/exp-pw50-e200-err-%j.txt
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
CK_TC=${CK_TC:-$P2/F_ibtracs_pwH50/best_hurricane.pth}
CK_ETC=${CK_ETC:-$P2/F_ibtracs_pwH50_e200/best.pth}
OUT=${OUT:-$NB/Global_pw50_e200}
YEARS=$(seq 1980 2024 | tr '\n' ' ')

for f in "$CK_TC" "$CK_ETC"; do
  if [ ! -s "$f" ]; then echo "ERROR: checkpoint $f missing"; exit 1; fi
done
echo "TC  checkpoint : $CK_TC"
echo "ETC checkpoint : $CK_ETC"
echo "outdir         : $OUT"
echo "started        : $(date)"

echo "=== Hurricane (from best_hurricane.pth) ==="
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK_TC" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane --subdirs --outdir "$OUT"
RC1=$?; echo "hurricane export exit $RC1"

echo "=== ETC (from best.pth) ==="
srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CK_ETC" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes ETC --subdirs --outdir "$OUT"
RC2=$?; echo "etc export exit $RC2"

echo "hurricane files: $(ls $OUT/Hurricanes_WNet_final 2>/dev/null | wc -l)/540"
echo "etc files      : $(ls $OUT/ETCs_WNet_final 2>/dev/null | wc -l)/540"
du -sh "$OUT" 2>/dev/null
echo "Finished $(date)"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; exit $?
