#!/bin/bash -l
# Run_etc_tracks_C200.sh — ETC tracking on the C_full200 detections, 2000-2010.
#
# Reads the full-record export directly
# (Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final), so no separate
# Phase2_detections export is needed. phase4_run_trackers.py globs
# era5dara_*_ETCs_detection.nc inside --det-dir, which that directory satisfies.
#
# MINIMUM DURATION 48 h is the tracker's own built-in value, not something imposed
# here: updated_monthly_ETC_Tracker.py:96 `self.min_hours = 48.0`, enforced at :351
# in validate_track_meteorology. min_track_length stays at 8, the value
# phase4_run_trackers.py:57 already passes — the 8 -> 12 tightening measured earlier
# is NOT applied here, so this run is a clean baseline for the new detections.
#
# One YEAR PER RANK. ETC tracking measured ~10.6 min/year (3 years took 31:52), so
# 11 years serially would be ~2 h; sharded it is one year's cost and backfills fast.
# Each year writes ETC_tracks_<y>_meteorological_v2.txt, so ranks cannot collide.
#
# Submit:
#   sbatch Run_etc_tracks_C200.sh
#   sbatch --export=Y0=2000,Y1=2010 Run_etc_tracks_C200.sh

#SBATCH --account=CLI138
#SBATCH -J etc_trk_c200
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH -o logs/etctrk200-%j.txt
#SBATCH -e logs/etctrk200-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=9

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export DET="${DET:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final}"
export OUTD="${OUTD:-$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_C200/tracks}"
export Y0="${Y0:-2000}"; export Y1="${Y1:-2010}"

echo "=============================================================="
echo " ETC tracking on C_full200 detections"
echo "   detections : $DET"
echo "   out        : $OUTD/etc"
echo "   years      : $Y0-$Y1  (one per rank)"
echo "   min duration: 48 h (ETCTracker built-in, unmodified)"
echo "   min_track_length: 8 (unchanged baseline)"
echo "   started    : $(date)"
echo "=============================================================="
grep -nE "self.min_hours = " updated_monthly_ETC_Tracker.py | head -1

srun -n $SLURM_NTASKS -c 9 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID))
  [ "$y" -gt "$Y1" ] && exit 0
  if [ -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then
    echo "[rank $SLURM_PROCID] $y already done"; exit 0
  fi
  python3 -W ignore -u ./phase4_run_trackers.py \
      --det-dir "$DET" --out-dir "$OUTD" \
      --years $y --which etc \
      > "logs/etctrk200_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] year $y rc=$?"
'
RC=$?
echo "Finished $(date), exit $RC"
echo "years with tracks: $(ls $OUTD/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)"
exit $RC
