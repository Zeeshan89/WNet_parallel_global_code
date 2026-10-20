#!/bin/bash -l
# Run_tc_label_screen.sh — Tier B: screen tightened CyTRACK criteria at the LABEL
# level against IBTrACS, one month per rank.
#
# No retraining is involved. The probability sweep showed the model converges onto
# its teacher (model CSI 0.1102 vs mask 0.1103), so the mask's own score against
# IBTrACS predicts what a retrained model would reach. Ranking candidates costs a
# mask recomputation instead of a cache rebuild plus a training run; only the
# winner is worth paying for.
#
# Scores the PHYSICS ONLY — the IBTrACS injection is not applied, because it adds
# storms the physics missed without touching false alarms and would hide exactly
# the effect being measured.
#
# 2003 + 2012 (both held-out test years), all 12 months = 24 ranks. Two full years
# rather than peak season only, so a candidate cannot win by being tuned to the
# Atlantic: the criteria are global and the S-hemisphere season falls in DJF.
#
# ~15-25 min per month at native 721x1440 with 12 candidates, hence one month per
# rank and a 40 min wall. Resumable: a month whose npz exists is skipped.
#
# Submit:
#   sbatch Run_tc_label_screen.sh
#   sbatch --export=YEARS="2003 2012" Run_tc_label_screen.sh

#SBATCH --account=CLI138
#SBATCH -J tc_screen
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH -o logs/tcscreen-%j.txt
#SBATCH -e logs/tcscreen-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=9

export OUTDIR="${OUTDIR:-tc_label_screen}"
export YEARS="${YEARS:-2003 2012}"
mkdir -p "$OUTDIR"

echo "=============================================================="
echo " TC label screening (Tier B) — physics only, no IBTrACS injection"
echo "   years   : $YEARS  (all months)"
echo "   outdir  : $OUTDIR"
echo "   started : $(date)"
echo "=============================================================="

srun -n $SLURM_NTASKS -c 9 --export=ALL bash -c '
  set -- $YEARS
  ny=$#; r=$SLURM_PROCID
  yi=$((r / 12)); m=$(((r % 12) + 1))
  [ "$yi" -ge "$ny" ] && exit 0
  eval y=\${$((yi+1))}
  python3 -W ignore -u ./phase4_tc_label_screen.py \
      --year $y --month $m --outdir "$OUTDIR" \
      > "logs/tcscreen_${y}${m}.log" 2>&1
  echo "[rank $r] $y-$m rc=$?"
'
RC=$?
echo "Finished $(date), exit $RC"
echo "partials: $(ls $OUTDIR/screen_*.npz 2>/dev/null | wc -l)"
exit $RC
