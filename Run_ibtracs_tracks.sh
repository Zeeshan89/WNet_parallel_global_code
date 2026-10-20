#!/bin/bash -l
# Run_ibtracs_tracks.sh — TC tracking + track-level scoring for the IBTrACS-target
# model. Chained after Run_ibtracs_eval.sh, which exports the detections.
#
# Tracking uses phase4_run_trackers.py unmodified, so any change in the resulting
# tracks comes from the detections rather than a changed tracking rule. One year
# per rank: run_tracking() loops years inside one process, so 5 years serially is
# ~20 min while sharded it is one year's cost.
#
# Scoring uses IBTRACS_WIND_SOURCE=any to match the target the model was trained on.
#SBATCH --account=CLI138
#SBATCH -J ibt_trk
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=11
#SBATCH --time=00:40:00
#SBATCH -o logs/ibttrk-%j.txt
#SBATCH -e logs/ibttrk-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=11
export IBTRACS_WIND_SOURCE=any

BASE=ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs
export DET="$BASE/nc"; export OUTD="$BASE/tracks"
export YEARS="${YEARS:-1983 1994 2003 2012 2021}"
if [ ! -d "$DET" ] || [ -z "$(ls -A $DET 2>/dev/null)" ]; then
  echo "ERROR: no detections in $DET — the export step did not produce output."; exit 1
fi
echo "detections: $(ls $DET/*.nc 2>/dev/null | wc -l) files"

srun -n $SLURM_NTASKS -c 11 --export=ALL bash -c '
  set -- $YEARS; i=$((SLURM_PROCID+1))
  [ "$i" -gt "$#" ] && exit 0
  eval y=\${$i}
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$OUTD" \
      --years $y --which hurricane > "logs/ibttrk_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] $y rc=$?"
'
echo "=== track-level scoring vs IBTrACS (any-agency wind) ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind tc \
    --track-dir "$OUTD" --years $YEARS --te-domain \
    --json phase4_results/tracks_tc_ibtracs.json || true
echo "=== same, global ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind tc \
    --track-dir "$OUTD" --years $YEARS --box none \
    --json phase4_results/tracks_tc_ibtracs_global.json || true
echo "done $(date)"
