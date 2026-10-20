#!/bin/bash -l
#SBATCH -A cli138
#SBATCH -J wnet_phase3
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o logs/phase3-%j.txt
#SBATCH -e logs/phase3-err-%j.txt

set -u
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
BASE=$CODE/ARs_Hurricanes_ETCs_MCS
export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
cd "$CODE"; mkdir -p logs "$BASE/Figures_hourly_vs_6hourly"

echo "=== Phase 3 analysis  $(date) ==="
$PY phase3_fullperiod.py \
   --hourly_dir "$BASE/Outputs_hourly" \
   --sixhr_dir  "$BASE/Outputs_6hourly_mcsonly" \
   --outdir     "$BASE/Figures_hourly_vs_6hourly"
echo "=== done rc=$?  $(date) ==="
