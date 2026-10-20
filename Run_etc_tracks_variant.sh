#!/bin/bash -l
# Run_etc_tracks_variant.sh — full-record (1980-2024) ETC tracks with one tracker configuration
# given as ETC_* environment variables, one year per rank on two nodes (about 15-20 min).
#
#   sbatch --export=ALL,TAG=S4_common,ETC_FILTER_PROFILE=common Run_etc_tracks_variant.sh
#   sbatch --export=ALL,TAG=S8_common_mslp,ETC_FILTER_PROFILE=common,ETC_CENTER_MODE=mslp \
#          Run_etc_tracks_variant.sh
#
# TAG names the output directory ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200/tracks_<TAG>/etc.
# The switches are documented in updated_monthly_ETC_Tracker.py; with none set, the output is
# the shipped tracker's. The mountain mask is off, as in production. Finished years are skipped.
# Afterwards, the paper analysis on these tracks:
#   cd ../etc_naec_common && ETC_VERSION=v2 ETC_FIG_SUFFIX=_<TAG> \
#      ETC_DIR=.../tracks_<TAG>/etc sbatch Run_etc_paper.sh   (and Run_naec_kde4.sh)

#SBATCH --account=CLI138
#SBATCH -J etc_tracks
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=45
#SBATCH --ntasks-per-node=23
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH -o logs/etctracks-%j.txt
#SBATCH -e logs/etctracks-err-%j.txt

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
TAG="${TAG:?set TAG, e.g. --export=ALL,TAG=S4_common,ETC_FILTER_PROFILE=common}"
export DET="${DET:-$NB/Global_pw50_e200/ETCs_WNet_final}"
export OUTD="${OUTD:-$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200/tracks_$TAG}"
export Y0="${Y0:-1980}"; export Y1="${Y1:-2024}"
export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0
echo "ETC tracks $TAG: $Y0-$Y1; detections $DET; out $OUTD/etc"
env | grep "^ETC_" | sort
echo "started $(date)"

srun -n $SLURM_NTASKS -c 1 --export=ALL bash -c '
  y=$((Y0 + SLURM_PROCID)); [ "$y" -gt "$Y1" ] && exit 0
  if [ -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then echo "[$y] done"; exit 0; fi
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$OUTD" --years $y --which etc \
      > "logs/etctracks_${TAG}_y${y}.log" 2>&1
  echo "[$y] rc=$?"'
echo "finished $(date): $(ls $OUTD/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)/$((Y1 - Y0 + 1)) years"
