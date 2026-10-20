#!/bin/bash -l
# Run_etc_tracks.sh — ETC tracking only, for the test years still missing.
#
# Run_p4tracks.sh does export -> hurricane -> ETC in one 1 h job and ran out of
# wall after hurricane 2021, so ETC tracks exist only for 1983 and 1994. The
# detections themselves are complete (12 ETC files for each of the five test
# years in Phase2_detections/nc), so this only needs the tracking step.
#
# No GPU: phase4_run_trackers.py --which etc reads the exported NetCDFs and runs
# ETCTracker on the CPU. The tracker itself is imported unmodified, so the tracks
# stay comparable with 1983/1994.
#
# 1 h wall: a 2 h request was scheduled ~1.5 h out, while an hour backfills far
# sooner. The year-skip below makes a timeout cheap to resume.
#
# Submit:
#   sbatch Run_etc_tracks.sh
#   sbatch --export=YEARS="2003 2012 2021" Run_etc_tracks.sh

#SBATCH --account=CLI138
#SBATCH -J wnet_etctrk
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH -o logs/etctrk-%j.txt
#SBATCH -e logs/etctrk-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

BASE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_detections
YEARS="${YEARS:-2003 2012 2021}"

# Resume: the tracker writes one ETC_tracks_<year>_meteorological_v2.txt per year and
# has no skip logic of its own, so drop years already on disk. With a 1 h wall a
# timeout then costs only the unfinished years on resubmit, not the whole set.
TODO=""
for y in $YEARS; do
  f="$BASE/tracks/etc/ETC_tracks_${y}_meteorological_v2.txt"
  if [ -s "$f" ]; then echo "skip $y (exists: $f)"; else TODO="$TODO $y"; fi
done
if [ -z "$TODO" ]; then echo "nothing to do — all years present"; exit 0; fi

echo "ETC tracking for years:$TODO"
srun -n1 -c7 --export=ALL \
  $(which python3) -W ignore -u ./phase4_run_trackers.py \
    --det-dir $BASE/nc --out-dir $BASE/tracks \
    --years $TODO --which etc

RC=$?
echo "exit $RC"
ls -la $BASE/tracks/etc/
exit $RC
