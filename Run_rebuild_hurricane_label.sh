#!/bin/bash -l
# Run_rebuild_hurricane_label.sh — rebuild the Hurricane label channel of the split
# cache under the Tier B decision (wind AND vorticity), 1980-2024.
#
# Writes a NEW cache (tensor_cache_v4_split) rather than mutating v3, so the
# C_full80/C_full200 results stay reproducible and the change can be reverted by
# pointing training back at v3.
#
# Only channel 7 changes. The 6 raw predictors and the AR/ETC/MCS labels are copied
# through byte-for-byte, so this is a hurricane-only recompute rather than the
# ~23 min/month full scipy mask rebuild, and the untouched channels cannot drift.
#
# The rebuild path was verified against production on 2003-09: rebuilding with the
# OLD require_both=False reproduced the existing cache channel at binary agreement
# 1.00000, MAE 3e-6, positive rate 0.00446 vs 0.00446. Only then was it trusted for
# the full record.
#
# Tier B measurement that motivated the change (2003+2012, 2924 steps, physics only):
#   OR  -> POD 0.888, FAR 0.986, CSI 0.014, 72.2 detections/step
#   AND -> POD 0.887, FAR 0.973, CSI 0.027, 36.1 detections/step
#   i.e. 105,604 fewer false objects for 2 fewer storms out of 3,271.
#
# 96 ranks, ~6 months each at ~4 min. Resumable: a month already present is skipped.
#
# Submit:
#   sbatch Run_rebuild_hurricane_label.sh
#   sbatch --export=Y0=1980,Y1=2024 Run_rebuild_hurricane_label.sh

#SBATCH --account=CLI138
#SBATCH -J hurr_label
#SBATCH -p batch
#SBATCH --nodes=12
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:35:00
#SBATCH -o logs/hurrlabel-%j.txt
#SBATCH -e logs/hurrlabel-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export OUTDIR="${OUTDIR:-$NB/tensor_cache_v4_split}"
export Y0="${Y0:-1980}"; export Y1="${Y1:-2024}"
mkdir -p "$OUTDIR"

NM=$(( (Y1 - Y0 + 1) * 12 ))
export NM
echo "=============================================================="
echo " Hurricane label rebuild — require_both=True (wind AND vorticity)"
echo "   out    : $OUTDIR"
echo "   years  : $Y0-$Y1  ($NM months over $SLURM_NTASKS ranks)"
echo "   started: $(date)"
echo "=============================================================="

srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  r=$SLURM_PROCID; n=$SLURM_NTASKS
  i=$r
  while [ "$i" -lt "$NM" ]; do
    y=$((Y0 + i / 12)); m=$((i % 12 + 1))
    if [ ! -s "$OUTDIR/split_$(printf %04d%02d $y $m).npy" ]; then
      python3 -W ignore -u ./phase4_rebuild_hurricane_label.py \
          --year $y --month $m --outdir "$OUTDIR" \
          >> "logs/hurrlabel_rank${r}.log" 2>&1 \
        || echo "[rank $r] FAILED $y-$m"
    fi
    i=$((i + n))
  done
  echo "[rank $r] done"
'
RC=$?
echo "Finished $(date), exit $RC"
echo "months: $(ls $OUTDIR/split_*.npy 2>/dev/null | grep -vc meta)/$NM"
du -sh "$OUTDIR" 2>/dev/null
exit $RC
