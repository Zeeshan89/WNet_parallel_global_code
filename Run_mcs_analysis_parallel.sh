#!/bin/bash -l
# Run_mcs_analysis_parallel.sh — parallel CONUS-masked MCS analysis.
# mcs-wnet-newfinalcode.py itself is a serial year-loop, so we fan out ONE process per
# (dataset, year) across nodes via srun. 30 work items = 2 datasets x 15 years (2004-2018):
#   ranks  0..14 -> NEW 6-hourly (Outputs_6hourly_mcsonly) -> outputs_wnet_new6h_IWV
#   ranks 15..29 -> HOURLY       (Outputs_hourly)          -> outputs_wnet_hourly_IWV
# Each rank runs the analysis for a SINGLE year (MCS_START_YEAR=MCS_END_YEAR), so all 30
# years compute concurrently (~one year's runtime instead of 30x serial).
#
# Submit: sbatch -M andes -D <codedir> Run_mcs_analysis_parallel.sh

#SBATCH -A cli138
#SBATCH -J mcs_analysis_par
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 2
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o logs/mcs_par-%j.txt
#SBATCH -e logs/mcs_par-err-%j.txt

set -u
export CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
export BASE=$CODE/ARs_Hurricanes_ETCs_MCS
export PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
export MCS_SKIP_SUMMARY=1   # skip per-ID summary.csv (unused by plots; the hourly bottleneck)
cd "$CODE"; mkdir -p logs

echo "=== parallel CONUS-masked MCS analysis (HOURLY only, 15 years)  start=$(date) ==="

srun -N 2 -n 15 --ntasks-per-node=8 -c 4 --export=ALL bash -c '
  r=$SLURM_PROCID
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
  # 6-hourly (new6h) is already complete (15/15 CONUS-masked) — HOURLY ONLY here.
  # if [ "$r" -lt 15 ]; then
  #   y=$((2004 + r));      DIR="$BASE/Outputs_6hourly_mcsonly/"; OUT="$CODE/outputs_wnet_new6h_IWV"; PAT="era5dara_{year}{month:02d}_MCS_detection*.nc"
  # else
  y=$((2004 + r)); DIR="$BASE/Outputs_hourly/"; OUT="$CODE/outputs_wnet_hourly_IWV"; PAT="era5dara_hourly_{year}{month:02d}_MCS_detection*.nc"
  # fi
  echo "[rank $r] year=$y -> $(basename $OUT)"
  MCS_START_YEAR=$y MCS_END_YEAR=$y MCS_DIR="$DIR" MCS_OUTPUT_DIR="$OUT" MCS_PATTERN="$PAT" \
      "$PY" -u mcs-wnet-newfinalcode.py > "logs/mcs_par_r${r}_y${y}.log" 2>&1
  echo "[rank $r] year=$y done rc=$? $(date)"
'
echo "=== parallel analysis finished $(date) ==="
echo "  new6h years:  $(ls -d $CODE/outputs_wnet_new6h_IWV/20?? 2>/dev/null | wc -l)/15"
echo "  hourly years: $(ls -d $CODE/outputs_wnet_hourly_IWV/20?? 2>/dev/null | wc -l)/15"
