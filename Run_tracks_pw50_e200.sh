#!/bin/bash -l
# Run_tracks_pw50_e200.sh — full-record TC *and* ETC tracking on the corrected
# 200-epoch detections (Global_pw50_e200), then score both.
#
# SHIPPED TRACKER CONFIGS ARE NOT THE FILE DEFAULTS. Re-running without these
# env vars silently gives the old, worse behaviour, so they are set here:
#   TC_MAX_AREA_PX=2000       stock 500 discards 34% of blobs, the largest first
#   TC_MIN_DURATION_H=24      stock 48 is unsupported by the detection field:
#                             only 1 missed storm has a >=5pt/48h run of
#                             detections near it, but 36 have a >=5pt/24h one.
#                             Measured worth: +36 storms, POD 0.700 -> 0.780.
#   ETC_MOUNTAIN_FRAME=0      mountain mask off, both stages
#   ETC_MOUNTAIN_TRACK_FRAC=1.0
#   min_track_length=8        ETC tracker default, unchanged
#
# One YEAR PER RANK for both trackers, with a skip-if-done guard, so the job is
# resumable: a walltime kill loses only the years in flight and resubmitting the
# identical script finishes the rest. 45 ranks, 45 years.
# Costs measured previously: TC ~15 min for the whole record sharded; ETC
# ~10.6 min per year, hence one year per rank rather than strided.
#
# $1 = which tracker(s) to run: tc | etc | both (default both). TC and ETC are
# fully independent -- different detection dirs, different output subdirectories,
# different trackers -- so splitting them into two CONCURRENT 20 min jobs both
# schedules faster than one 40 min job and finishes sooner in wall clock:
#     sbatch --time=00:20:00 Run_tracks_pw50_e200.sh tc
#     sbatch --time=00:20:00 Run_tracks_pw50_e200.sh etc
# Each arm scores only its own class, so neither waits on the other.
#
# Submit:  sbatch --dependency=afterok:<export job> Run_tracks_pw50_e200.sh

#SBATCH --account=CLI138
#SBATCH -J trk_pw50_e200
#SBATCH -p batch
#SBATCH --nodes=6
#SBATCH --ntasks=45
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH -o logs/trk-pw50-e200-%j.txt
#SBATCH -e logs/trk-pw50-e200-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
WHICH="${1:-both}"
case "$WHICH" in tc|etc|both) ;; *) echo "ERROR: \$1 must be tc|etc|both"; exit 1;; esac
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

export TC_MAX_AREA_PX=2000
export TC_MIN_DURATION_H=24
export ETC_MOUNTAIN_FRAME=0
export ETC_MOUNTAIN_TRACK_FRAC=1.0
export IBTRACS_WIND_SOURCE=any

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export DET_TC=${DET_TC:-$NB/Global_pw50_e200/Hurricanes_WNet_final}
export DET_ETC=${DET_ETC:-$NB/Global_pw50_e200/ETCs_WNet_final}
export OUTD=${OUTD:-ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/tracks_pw50_e200}
export Y0=${Y0:-1980}; export Y1=${Y1:-2024}

NEED=""
[ "$WHICH" = tc ]  || [ "$WHICH" = both ] && NEED="$NEED $DET_TC"
[ "$WHICH" = etc ] || [ "$WHICH" = both ] && NEED="$NEED $DET_ETC"
for d in $NEED; do
  if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "ERROR: no detections in $d"; exit 1
  fi
done
echo "TC  detections : $(ls $DET_TC/*.nc  | wc -l) files"
echo "ETC detections : $(ls $DET_ETC/*.nc | wc -l) files"
echo "out            : $OUTD"
echo "TC gates       : TC_MAX_AREA_PX=$TC_MAX_AREA_PX TC_MIN_DURATION_H=$TC_MIN_DURATION_H"
echo "ETC gates      : MOUNTAIN_FRAME=$ETC_MOUNTAIN_FRAME FRAC=$ETC_MOUNTAIN_TRACK_FRAC"
echo "arm            : $WHICH"
echo "started        : $(date)"

if [ "$WHICH" = tc ] || [ "$WHICH" = both ]; then
echo "=== TC tracking, one year per rank ==="
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID)); [ "$y" -gt "$Y1" ] && exit 0
  [ -s "$OUTD/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt" ] \
      && { echo "[$y] TC already done"; exit 0; }
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET_TC" --out-dir "$OUTD" \
      --years $y --which hurricane > "logs/trkpw50e200_tc_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] TC $y rc=$?"
'
fi

if [ "$WHICH" = etc ] || [ "$WHICH" = both ]; then
echo "=== ETC tracking, one year per rank ==="
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID)); [ "$y" -gt "$Y1" ] && exit 0
  [ -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ] \
      && { echo "[$y] ETC already done"; exit 0; }
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET_ETC" --out-dir "$OUTD" \
      --years $y --which etc > "logs/trkpw50e200_etc_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] ETC $y rc=$?"
'
fi

NTC=$(ls -d $OUTD/hurricane/CONUS_year_* 2>/dev/null | wc -l)
NETC=$(ls $OUTD/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)
NWANT=$((Y1-Y0+1))
echo "TC  years: $NTC/$NWANT     ETC years: $NETC/$NWANT"
INCOMPLETE=0
{ [ "$WHICH" = tc ]  || [ "$WHICH" = both ]; } && [ "$NTC"  -lt "$NWANT" ] && INCOMPLETE=1
{ [ "$WHICH" = etc ] || [ "$WHICH" = both ]; } && [ "$NETC" -lt "$NWANT" ] && INCOMPLETE=1
if [ "$INCOMPLETE" = 1 ]; then
  echo "INCOMPLETE — resubmit this same script to continue: sbatch $0"
  MISS=""
  for y in $(seq $Y0 $Y1); do
    ok=1
    { [ "$WHICH" = tc ] || [ "$WHICH" = both ]; } && \
      [ ! -s "$OUTD/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt" ] && ok=0
    { [ "$WHICH" = etc ] || [ "$WHICH" = both ]; } && \
      [ ! -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ] && ok=0
    [ "$ok" = 1 ] || MISS="$MISS $y"
  done
  echo "  (arm $WHICH)"
  echo "  missing:$MISS"
  exit 0
fi

if [ "$WHICH" = tc ] || [ "$WHICH" = both ]; then
echo "=== score TC on the held-out years vs IBTrACS (TE box) ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind tc --track-dir "$OUTD" \
    --years 1983 1994 2003 2012 2021 --te-domain \
    --json phase4_results/tracks_tc_pw50_e200_tebox.json || true
echo "=== same, global domain ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind tc --track-dir "$OUTD" \
    --years 1983 1994 2003 2012 2021 \
    --json phase4_results/tracks_tc_pw50_e200.json || true
fi

if [ "$WHICH" = etc ] || [ "$WHICH" = both ]; then
echo "=== score ETC vs TempestExtremes (CONUS box) ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind etc --track-dir "$OUTD" \
    --years 1983 1994 2003 2012 2021 \
    --json phase4_results/tracks_etc_pw50_e200_te.json || true
echo "=== score ETC vs NAEC (common box, symmetric QC) ==="
$(which python3) -W ignore -u ./phase4_evaluate_tracks.py --kind etc --track-dir "$OUTD" \
    --years 1983 1994 2003 2012 2021 --ref naec --common-box --qc \
    --json phase4_results/tracks_etc_pw50_e200_naec.json || true
fi
echo "done $(date)"
