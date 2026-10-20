#!/bin/bash -l
# Run_etc_tracker_sweep2.sh — second ETC tracker sweep, 2000-2010, one node.
#
# Sweep 1 (Run_etc_tracker_sweep.sh, phase4_results/etc_tracker_sweep.json): the "common"
# filter profile lifted NAEC POD 0.56 -> 0.77 (CSI 0.434 -> 0.465); gap bridging on its own
# LOST hits (POD 0.22 at gap 4) because the link radius grows with the gap; global linking was
# worse than the legacy order; duplicates tripled. This sweep adds
#
#   S8_common_mslp              closed-contour MSLP centres (ETC_CENTER_MODE=mslp)
#   S9_common_gap2_nogrow       gap 2 with the link radius held at 800 km
#   S10_common_gap3_nogrow      gap 3, radius held
#   S11_common_mslp_gap2_nogrow mslp centres + gap 2, radius held
#   S12_base_mslp               shipped filters + mslp centres (the centre effect alone)
#   S13_common_mslp_d100        mslp centres with a 100 Pa closed contour (TE uses 200)
#
# On one real time step 58 of 88 mask blobs were tiny fragments (< 100 cells of 0.25 deg,
# about one model pixel) and 55 of them held no closed low; 12 of 14 disc-sized blobs did.
# Same detections and output base as sweep 1, so the final scoring covers S0-S13 together.
# 6 variants x 11 years = 66 runs, one per rank on two nodes (about 25 min); finished runs are skipped.
#
#   sbatch Run_etc_tracker_sweep2.sh

#SBATCH --account=CLI138
#SBATCH -J etc_trk_sweep2
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=66
#SBATCH --ntasks-per-node=33
#SBATCH --cpus-per-task=1
#SBATCH --time=00:35:00
#SBATCH -o logs/etctrksweep2-%j.txt
#SBATCH -e logs/etctrksweep2-err-%j.txt
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
export NY=$((Y1 - Y0 + 1)) NV=6
echo "ETC tracker sweep 2: $NV variants x $NY years; detections $DET; out $BASE/<variant>/etc; $(date)"

srun -n $SLURM_NTASKS -c 1 --export=ALL bash -c '
  export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0          # production setting
  for ((task = SLURM_PROCID; task < NV * NY; task += SLURM_NTASKS)); do
    v=$((task / NY)); y=$((Y0 + task % NY))
    unset ETC_MAX_GAP_STEPS ETC_FILTER_PROFILE ETC_LONWRAP ETC_LINK_MODE ETC_CENTER_MODE ETC_GAP_RADIUS_GROWTH
    case $v in
      0) TAG=S8_common_mslp; export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ;;
      1) TAG=S9_common_gap2_nogrow; export ETC_FILTER_PROFILE=common ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0 ;;
      2) TAG=S10_common_gap3_nogrow; export ETC_FILTER_PROFILE=common ETC_MAX_GAP_STEPS=3 ETC_GAP_RADIUS_GROWTH=0 ;;
      3) TAG=S11_common_mslp_gap2_nogrow
         export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0 ;;
      4) TAG=S12_base_mslp; export ETC_CENTER_MODE=mslp ;;
      5) TAG=S13_common_mslp_d100; export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ETC_CC_DEPTH_PA=100 ;;
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
