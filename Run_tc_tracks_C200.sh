#!/bin/bash -l
# Run_tc_tracks_C200.sh — TC tracking on the C_full200 detections, 2000-2010.
#
# Reads the full-record export directly
# (Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final), so no separate
# Phase2_detections export is needed — that directory already holds every year.
#
# MINIMUM DURATION 48 h is the tracker's own built-in value, not something imposed
# here: Hurricane_track.py:62 `min_duration_hours = 48` and :75 `min_hours = 48.0`
# ("IBTrACS standard"), enforced in validate_hurricane_track. phase4_run_trackers.py
# is imported unmodified so the tracking rules stay identical to the C_full80 runs
# and any difference in the tracks comes from the detections alone.
#
# One YEAR PER RANK. run_tracking() loops years inside one process, so 11 years
# serially would be ~2 h; sharded it is the cost of a single year (~10-15 min) and
# backfills quickly. Each year writes its own CONUS_year_<y>/ subdirectory, so the
# ranks cannot collide.
#
# 4 ranks per node rather than 8: Hurricane_track.py sorts large wind/vorticity
# fields per year, so give each rank room.
#
# Submit:
#   sbatch Run_tc_tracks_C200.sh
#   sbatch --export=Y0=2000,Y1=2010 Run_tc_tracks_C200.sh

#SBATCH --account=CLI138
#SBATCH -J tc_trk_c200
#SBATCH -p batch
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH -o logs/tctrk200-%j.txt
#SBATCH -e logs/tctrk200-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=14

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export DET="${DET:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final}"
export OUTD="${OUTD:-$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_C200/tracks}"
export Y0="${Y0:-2000}"; export Y1="${Y1:-2010}"

echo "=============================================================="
echo " TC tracking on C_full200 detections"
echo "   detections : $DET"
echo "   out        : $OUTD/hurricane"
echo "   years      : $Y0-$Y1  (one per rank)"
echo "   min duration: 48 h (Hurricane_track.py built-in, unmodified)"
echo "   started    : $(date)"
echo "=============================================================="
grep -nE "min_duration_hours = |min_hours = " Hurricane_track.py | head -2

srun -n $SLURM_NTASKS -c 14 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID))
  [ "$y" -gt "$Y1" ] && exit 0
  if [ -s "$OUTD/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt" ]; then
    echo "[rank $SLURM_PROCID] $y already done"; exit 0
  fi
  python3 -W ignore -u ./phase4_run_trackers.py \
      --det-dir "$DET" --out-dir "$OUTD" \
      --years $y --which hurricane \
      > "logs/tctrk200_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] year $y rc=$?"
'
RC=$?
echo "Finished $(date), exit $RC"
echo "years with tracks: $(ls -d $OUTD/hurricane/CONUS_year_* 2>/dev/null | wc -l)"
exit $RC
