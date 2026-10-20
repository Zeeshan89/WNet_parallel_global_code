#!/bin/bash -l
# Run_export_tc_e13_overwrite.sh — re-export ONLY the Hurricane class into the
# unified product directory, from the epoch-13 checkpoint, OVERWRITING what is
# there.
#
# Why this exists
# ---------------
# Run_export_pw50_e200.sh was submitted with CK_TC pointing at the 200-epoch run's
# best_hurricane.pth (epoch 36). CK_TC was later edited to the 30-epoch run's
# epoch-13 checkpoint, but sbatch CAPTURES THE SCRIPT AT SUBMIT TIME -- editing the
# file afterwards and releasing the hold re-runs the STORED copy, so the edit never
# took effect and epoch 36 was exported. Cancel-and-resubmit is the only way to
# change a queued job's script.
#
# The accident measured the difference cleanly, held-out years, global:
#     epoch 13 (pixel recall 0.754): 469 tracks, 386 hits, 68 miss, POD 0.850,
#                                    FAR 0.141, CSI 0.742
#     epoch 36 (pixel recall 0.670): 424 tracks, 367 hits, 87 miss, POD 0.808,
#                                    FAR 0.080, CSI 0.752
# CSI is flat, so the recall knob rides an iso-CSI curve: 19 extra storms found for
# 32 extra false alarms. The objective here is fewest MISSED storms, and FAR 0.141
# is still 3.6x better than TempestExtremes' 0.514, so epoch 13 is the choice.
#
# --overwrite is REQUIRED: phase4_export_detections.py skips months whose .nc
# already exists, so without it this job would be a no-op over the epoch-36 files.
# ETC is deliberately NOT re-exported -- its export used the intended checkpoint
# and re-running it would only burn time.
#
# Submit:  sbatch Run_export_tc_e13_overwrite.sh

#SBATCH --account=CLI138
#SBATCH -J exp_tc_e13
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH -o logs/exp-tc-e13-%j.txt
#SBATCH -e logs/exp-tc-e13-err-%j.txt
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
CKPT=ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50/best_hurricane.pth
OUT=$NB/Global_pw50_e200
YEARS=$(seq 1980 2024 | tr '\n' ' ')

if [ ! -s "$CKPT" ]; then echo "ERROR: $CKPT missing"; exit 1; fi
echo "checkpoint : $CKPT   (expect epoch 13)"
echo "outdir     : $OUT/Hurricanes_WNet_final   (OVERWRITING epoch-36 files)"
echo "started    : $(date)"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
  $(which python3) -W ignore -u ./phase4_export_detections.py \
    --checkpoint "$CKPT" --cache-dir "$NB/tensor_cache_v5_ibtracs" \
    --years $YEARS --classes Hurricane --subdirs --outdir "$OUT" --overwrite
RC=$?
echo "exit $RC"
echo "hurricane files: $(ls $OUT/Hurricanes_WNet_final 2>/dev/null | wc -l)/540"
echo "etc files      : $(ls $OUT/ETCs_WNet_final 2>/dev/null | wc -l)/540  (untouched)"
echo "Finished $(date)"
exit $RC
