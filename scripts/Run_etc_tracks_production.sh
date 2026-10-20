#!/bin/bash -l
# Run_etc_tracks_production.sh — ETC tracks for the FULL RECORD, 1980-2024,
# using the configuration chosen by phase4_etc_tracker_design.py.
#
# CONFIGURATION AND WHY
# ---------------------
#   ETC_MOUNTAIN_FRAME      = 0     mountain mask OFF (both stages)
#   ETC_MOUNTAIN_TRACK_FRAC = 1.0
#   min_track_length        = 8     tracker default, unchanged
#   min duration            = 48 h  tracker built-in, unchanged
#
# Chosen against TWO independent references on their common ground
# (lon -135..-55, lat 20..55) with symmetric QC, over 2000-2010:
#
#   variant (minlen 8)              TE POD  TE CSI   NAEC POD  NAEC CSI
#   A frame ON,  frac 0.30 (old)     0.524   0.263      0.559     0.456
#   B frame OFF, frac 0.30           0.465   0.250      0.493     0.408
#   C frame OFF, frac 0.70           0.556   0.273      0.563     0.443
#   D mask OFF                       0.562   0.273      0.565     0.440
#
# Removing the mask raises detection on BOTH references. The mask was discarding
# ~19% of real TE ETCs and 18.6% of real ETC genesis points, because its Rockies
# box covers lee cyclogenesis (Colorado lows, Alberta clippers) rather than an
# artefact factory. The artefact it targeted is real, but STATIONARY, and the
# tracker's own propagation filters (min_avg_propagation_ms 3.5, min_net_east_km
# 500, retrograde limits) already remove stationary lows without the geographic
# collateral damage.
#
# B is the worst variant, which is the tell: dropping only the per-frame mask lets
# mountain centres into tracks and then the 30% track rule kills the very lee
# cyclones just admitted. C ~= D, so the 0.70 rule is nearly inert; D is simpler.
#
# min_track_length stays 8 because the objective is fewest missed storms. Raising
# it to 12 gives a better balanced CSI (0.291/0.403) but lowers POD on both
# references (0.487/0.469) — more missing, not less.
#
# Tracking is GLOBAL (lon -180..180, lat -70..70). The design window is CONUS-only
# because that is where both references overlap; no ETC reference is global, so the
# out-of-box tracks are produced but unvalidated.
#
# One year per rank: 45 years, ~10 min each, so ~15 min wall instead of ~7 h.
# Resumable — a year whose file exists is skipped.
#
# Submit:
#   sbatch Run_etc_tracks_production.sh
#   sbatch --export=Y0=1980,Y1=2024 Run_etc_tracks_production.sh

#SBATCH --account=CLI138
#SBATCH -J etc_prod
#SBATCH -p batch
#SBATCH --nodes=6
#SBATCH --ntasks=45
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/etcprod-%j.txt
#SBATCH -e logs/etcprod-err-%j.txt
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
export DET="${DET:-$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final}"
export OUTD="${OUTD:-$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_C200/tracks_final}"
export Y0="${Y0:-1980}"; export Y1="${Y1:-2024}"

# The chosen configuration, set here so the job log records it explicitly rather
# than inheriting whatever the module defaults happen to be.
export ETC_MOUNTAIN_FRAME=0
export ETC_MOUNTAIN_TRACK_FRAC=1.0

echo "=============================================================="
echo " ETC production tracking — C_full200 detections, FULL RECORD"
echo "   detections    : $DET"
echo "   out           : $OUTD/etc"
echo "   years         : $Y0-$Y1  (one per rank)"
echo "   mountain mask : OFF  (FRAME=$ETC_MOUNTAIN_FRAME FRAC=$ETC_MOUNTAIN_TRACK_FRAC)"
echo "   min_track_len : 8 (tracker default)"
echo "   min duration  : 48 h (tracker built-in)"
echo "   started       : $(date)"
echo "=============================================================="

srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID))
  [ "$y" -gt "$Y1" ] && exit 0
  if [ -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then
    echo "[rank $SLURM_PROCID] $y already done"; exit 0
  fi
  python3 -W ignore -u ./phase4_run_trackers.py \
      --det-dir "$DET" --out-dir "$OUTD" --years $y --which etc \
      > "logs/etcprod_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] year $y rc=$?"
'
RC=$?
echo "Finished $(date), exit $RC"
echo "years with tracks: $(ls $OUTD/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)/$((Y1-Y0+1))"
exit $RC
