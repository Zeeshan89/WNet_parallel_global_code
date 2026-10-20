#!/bin/bash -l
# Run_etc_tracker_sweep3.sh — third ETC tracker sweep, 2000-2010, two nodes.
#
# Sweep 2 (phase4_results/etc_tracker_sweep.json, NAEC common box): closed-contour MSLP centres
# are the decisive change. S11 (common filters, mslp centres 200 Pa / 6 deg, gap 2 held radius):
# POD 0.612, FAR 0.116, CSI 0.566, 14 duplicates, against the shipped S0 0.560 / 0.281 / 0.434 /
# 383. S13 (100 Pa): POD 0.665, FAR 0.210, CSI 0.561. What the mslp mode gives up against the
# filter-only S4 (POD 0.770, FAR 0.344) are blobs that hold no closed low. This sweep brackets
# the closed-contour depth and radius and adds a centroid fallback for cyclone-scale blobs
# without a closed low (ETC_CENTER_FALLBACK_MIN, cells of 0.25 deg; a 600 km disc is ~1460):
#
#   S14_common_mslp_d100_fb700   100 Pa + centroid fallback for blobs >= 700 cells
#   S15_common_mslp_d150         150 Pa
#   S16_common_mslp_d50          50 Pa
#   S17_common_mslp_d100_r8      100 Pa within 8 degrees
#   S18_common_mslp_d200_fb700   200 Pa + fallback 700
#
# All with ETC_FILTER_PROFILE=common, gap 2 with the link radius held (as S11). Same detections
# and output base as sweeps 1-2, so the final scoring covers S0-S18. 55 runs, one per rank.
#
#   sbatch Run_etc_tracker_sweep3.sh

#SBATCH --account=CLI138
#SBATCH -J etc_trk_sweep3
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=55
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=00:35:00
#SBATCH -o logs/etctrksweep3-%j.txt
#SBATCH -e logs/etctrksweep3-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export DET="${DET:-$NB/Global_pw50_e200/ETCs_WNet_final}"
export BASE="${BASE:-$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200/tracker_sweep}"
export Y0="${Y0:-2000}"; export Y1="${Y1:-2010}"
export NY=$((Y1 - Y0 + 1)) NV=5
echo "ETC tracker sweep 3: $NV variants x $NY years; detections $DET; out $BASE/<variant>/etc; $(date)"

srun -n $SLURM_NTASKS -c 1 --export=ALL bash -c '
  export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0
  export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0
  for ((task = SLURM_PROCID; task < NV * NY; task += SLURM_NTASKS)); do
    v=$((task / NY)); y=$((Y0 + task % NY))
    unset ETC_CC_DEPTH_PA ETC_CC_RADIUS_DEG ETC_CENTER_FALLBACK_MIN
    case $v in
      0) TAG=S14_common_mslp_d100_fb700; export ETC_CC_DEPTH_PA=100 ETC_CENTER_FALLBACK_MIN=700 ;;
      1) TAG=S15_common_mslp_d150; export ETC_CC_DEPTH_PA=150 ;;
      2) TAG=S16_common_mslp_d50; export ETC_CC_DEPTH_PA=50 ;;
      3) TAG=S17_common_mslp_d100_r8; export ETC_CC_DEPTH_PA=100 ETC_CC_RADIUS_DEG=8 ;;
      4) TAG=S18_common_mslp_d200_fb700; export ETC_CC_DEPTH_PA=200 ETC_CENTER_FALLBACK_MIN=700 ;;
    esac
    if [ -s "$BASE/$TAG/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then
      echo "[rank $SLURM_PROCID] $TAG $y done"; continue
    fi
    python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$BASE/$TAG" \
        --years $y --which etc > "logs/etctrk_${TAG}_y${y}.log" 2>&1
    echo "[rank $SLURM_PROCID] $TAG $y rc=$?"
  done'
echo "tracking finished $(date), exit $?"
for d in "$BASE"/S*; do echo "  $(basename $d): $(ls $d/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)/$NY"; done

srun -n 1 -c 1 --export=ALL python3 -W ignore -u ./phase4_etc_tracker_design.py \
    --sweep-dir "$BASE" --min-lens 8 --json phase4_results/etc_tracker_sweep.json
echo "exit $?"
