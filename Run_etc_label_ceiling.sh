#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J etc_label_ceil
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=22
#SBATCH --ntasks-per-node=22
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o logs/etclabelceil-%j.txt
#SBATCH -e logs/etclabelceil-err-%j.txt
#SBATCH --export=NONE
#
# What the retrained model can at best deliver with each ETC label: the LABEL masks themselves
# (ETC_label_candidates/{legacy,cc}, 2000-2010, from Run_build_etc_label.sh STAGE=native) are
# (1) run through the S15 tracker and scored against NAEC / TempestExtremes exactly like the
#     tracker sweep (phase4_etc_tracker_design.py), and
# (2) tested for NAECC coverage by depth quintile with phase4_etc_miss_attribution.py --det-dir
#     (the "mask" stage: the shallowest quintile is where the shipped label fails).
# The model reproduces its label (ETC IoU ~0.76), so the label ceiling predicts the retrain.
#
#   sbatch --dependency=afterok:<native build> Run_etc_label_ceiling.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export LAB="$NB/ETC_label_candidates"
export BASE="$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2/label_ceiling"
export Y0=2000 Y1=2010 NY=11
# the S15 tracker, mountain mask off
export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0
export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ETC_CC_DEPTH_PA=150 ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0
for s in legacy cc; do n=$(ls $LAB/$s/era5dara_*_ETCs_detection.nc 2>/dev/null | wc -l); echo "$s: $n/132 native label months"; [ "$n" -ge 132 ] || exit 1; done

echo "=== 1. S15 tracker on the label masks (22 ranks = 2 labels x 11 years) ==="
srun -n 22 -c 2 --export=ALL bash -c '
  s=$([ $((SLURM_PROCID / NY)) -eq 0 ] && echo legacy || echo cc); y=$((Y0 + SLURM_PROCID % NY))
  TAG="L_${s}_S15"
  if [ -s "$BASE/$TAG/etc/ETC_tracks_${y}_meteorological_v2.txt" ]; then echo "[$TAG $y] done"; exit 0; fi
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$LAB/$s" --out-dir "$BASE/$TAG" --years $y --which etc \
      > "logs/etctrk_${TAG}_y${y}.log" 2>&1; echo "[$TAG $y] rc=$?"'
srun -n 1 -c 2 --export=ALL python3 -W ignore -u ./phase4_etc_tracker_design.py \
    --sweep-dir "$BASE" --min-lens 8 --json phase4_results/etc_label_ceiling.json

echo "=== 2. NAECC coverage of the label masks by depth quintile ==="
# the diagnostic's centre stage uses the shipped centroid tracker; the S15 variables above
# would make find_etc_centers demand an MSLP field, so they are cleared for this step
unset ETC_FILTER_PROFILE ETC_CENTER_MODE ETC_CC_DEPTH_PA ETC_MAX_GAP_STEPS ETC_GAP_RADIUS_GROWTH
for s in legacy cc; do
  srun -n 11 -c 2 --export=ALL python3 -W ignore -u ./phase4_etc_miss_attribution.py --years $Y0 $Y1 \
      --det-dir "$LAB/$s" --trk-dir "$BASE/L_${s}_S15/etc" --outdir "evaluations/etc_label_coverage_$s"
  srun -n 1 -c 2 --export=ALL python3 -W ignore -u ./phase4_etc_miss_attribution.py --summarize \
      --outdir "evaluations/etc_label_coverage_$s" | sed -n '/by depth/,/by season/p'
done
echo "exit $?  $(date)"
