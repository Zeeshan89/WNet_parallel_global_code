#!/bin/bash -l
# Run_etc_mask_sweep.sh — ETC tracker mountain-mask variants, 2000-2010.
#
# The mask acts DURING tracking, so unlike min_track_length it cannot be explored
# post-hoc; each variant needs its own tracker run. Four variants:
#
#   A  maskA_frame1_frac030   frame ON,  frac 0.30   (current default = baseline)
#   B  maskB_frame0_frac030   frame OFF, frac 0.30   (keep track rule, allow genesis)
#   C  maskC_frame0_frac070   frame OFF, frac 0.70   (only reject lows that live there)
#   D  maskD_off              frame OFF, frac 1.00   (mask fully disabled)
#
# Variant A is included even though Phase2_C200 already has it, so all four come
# from an identical code path and no difference can be blamed on the harness.
#
# Rationale for testing at all: against TempestExtremes ETCs 2000-2010, the mask
# discards ~19% of REAL storms and 18.6% of real ETCs have their genesis inside a
# box — the Rockies box is lee cyclogenesis, not an artefact factory. The tracker
# already carries propagation-based filters (min_avg_propagation_ms 3.5,
# min_net_east_km 500, retrograde limits) that target stationary orographic lows
# directly, so the geographic mask may be redundant as well as costly.
#
# 4 variants x 11 years = 44 ranks, one year per rank. Resumable per variant-year.
#
# Submit:
#   sbatch Run_etc_mask_sweep.sh
#   sbatch --export=Y0=2000,Y1=2010 Run_etc_mask_sweep.sh

#SBATCH --account=CLI138
#SBATCH -J etc_mask_sweep
#SBATCH -p batch
#SBATCH --nodes=6
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/etcmask-%j.txt
#SBATCH -e logs/etcmask-err-%j.txt
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
export DET="$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final"
export BASE="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_C200/mask_sweep"
export Y0="${Y0:-2000}"; export Y1="${Y1:-2010}"
NY=$((Y1 - Y0 + 1))
export NY

echo "=============================================================="
echo " ETC mountain-mask sweep — 4 variants x $NY years"
echo "   detections : $DET"
echo "   out        : $BASE/<variant>/etc"
echo "   years      : $Y0-$Y1"
echo "   min duration 48 h and min_track_length 8 unchanged in every variant"
echo "   started    : $(date)"
echo "=============================================================="

srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  v=$((SLURM_PROCID / NY)); k=$((SLURM_PROCID % NY)); y=$((Y0 + k))
  [ "$y" -gt "$Y1" ] && exit 0
  case $v in
    0) TAG=maskA_frame1_frac030; export ETC_MOUNTAIN_FRAME=1 ETC_MOUNTAIN_TRACK_FRAC=0.30 ;;
    1) TAG=maskB_frame0_frac030; export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=0.30 ;;
    2) TAG=maskC_frame0_frac070; export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=0.70 ;;
    3) TAG=maskD_off;            export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.00 ;;
    *) exit 0 ;;
  esac
  OUT="$BASE/$TAG"
  if [ -s "$OUT/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then
    echo "[rank $SLURM_PROCID] $TAG $y done"; exit 0
  fi
  python3 -W ignore -u ./phase4_run_trackers.py \
      --det-dir "$DET" --out-dir "$OUT" --years $y --which etc \
      > "logs/etcmask_${TAG}_y${y}.log" 2>&1
  echo "[rank $SLURM_PROCID] $TAG $y rc=$?"
'
RC=$?
echo "Finished $(date), exit $RC"
for t in maskA_frame1_frac030 maskB_frame0_frac030 maskC_frame0_frac070 maskD_off; do
  echo "  $t: $(ls $BASE/$t/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)/$NY"
done
exit $RC
