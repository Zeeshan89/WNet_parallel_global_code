#!/bin/bash -l
# Run_etc_tracker_sweep.sh — ETC tracker design switches, 2000-2010, one node.
#
# phase4_etc_miss_attribution.py: WNet matches 0.49 of NAECC cyclones; 0.35 are lost at the
# TRACK stage (mask covers them, a centre is nearby, no valid track) against 0.11 in the mask,
# and the shipped tracker rules applied to the NAECC tracks themselves pass only 0.53 of them
# (link floor of 72 km per 6 h, path >= 1800 km, retrograde limits, net-eastward >= 500 km).
# The switches act DURING tracking, so each variant needs its own tracker run:
#
#   S0_base          shipped rules (control; mountain mask off as in production)
#   S1_gap2 S2_gap3 S3_gap4                     ETC_MAX_GAP_STEPS
#   S4_common        ETC_FILTER_PROFILE=common  (shared QC only, no lower link speed)
#   S5_common_gap2
#   S6_common_gap2_wrap_global                  + ETC_LONWRAP=1 ETC_LINK_MODE=global
#   S7_common_gap3_wrap_global
#
# Detections: Global_pw50_e200/ETCs_WNet_final (the C_full200 ETC files no longer exist).
# 8 variants x 11 years = 88 tracker runs shared by 44 ranks; finished runs are skipped, so
# the job can be resubmitted. Scoring against NAEC and TempestExtremes runs at the end.
#
#   sbatch Run_etc_tracker_sweep.sh

#SBATCH --account=CLI138
#SBATCH -J etc_trk_sweep
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=44
#SBATCH --cpus-per-task=1
#SBATCH --time=00:50:00
#SBATCH -o logs/etctrksweep-%j.txt
#SBATCH -e logs/etctrksweep-err-%j.txt
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
export NY=$((Y1 - Y0 + 1)) NV=8
echo "ETC tracker sweep: $NV variants x $NY years; detections $DET; out $BASE/<variant>/etc; $(date)"

srun -n $SLURM_NTASKS -c 1 --export=ALL bash -c '
  export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0          # production setting
  for ((task = SLURM_PROCID; task < NV * NY; task += SLURM_NTASKS)); do
    v=$((task / NY)); y=$((Y0 + task % NY))
    unset ETC_MAX_GAP_STEPS ETC_FILTER_PROFILE ETC_LONWRAP ETC_LINK_MODE
    case $v in
      0) TAG=S0_base ;;
      1) TAG=S1_gap2; export ETC_MAX_GAP_STEPS=2 ;;
      2) TAG=S2_gap3; export ETC_MAX_GAP_STEPS=3 ;;
      3) TAG=S3_gap4; export ETC_MAX_GAP_STEPS=4 ;;
      4) TAG=S4_common; export ETC_FILTER_PROFILE=common ;;
      5) TAG=S5_common_gap2; export ETC_FILTER_PROFILE=common ETC_MAX_GAP_STEPS=2 ;;
      6) TAG=S6_common_gap2_wrap_global
         export ETC_FILTER_PROFILE=common ETC_MAX_GAP_STEPS=2 ETC_LONWRAP=1 ETC_LINK_MODE=global ;;
      7) TAG=S7_common_gap3_wrap_global
         export ETC_FILTER_PROFILE=common ETC_MAX_GAP_STEPS=3 ETC_LONWRAP=1 ETC_LINK_MODE=global ;;
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

srun -n 1 -c 8 --export=ALL python3 -W ignore -u ./phase4_etc_tracker_design.py \
    --sweep-dir "$BASE" --min-lens 8 --json phase4_results/etc_tracker_sweep.json
echo "exit $?"
