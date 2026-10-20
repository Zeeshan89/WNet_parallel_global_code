#!/bin/bash
#SBATCH --job-name=precompute_tensors
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --partition=batch
#SBATCH --account=cli138
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zaslam@vols.utk.edu


set -u
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
BASE=$CODE/ARs_Hurricanes_ETCs_MCS
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16
cd "$CODE"; mkdir -p logs

echo "=== NEW 6-hourly analysis (2014-2018) -> outputs_wnet_new6h_IWV  $(date) ==="
MCS_START_YEAR=2004 MCS_END_YEAR=2018 \
  MCS_DIR="$BASE/Outputs_6hourly_mcsonly/" \
  MCS_OUTPUT_DIR="$CODE/outputs_wnet_new6h_IWV" \
  $PY -u mcs-wnet-newfinalcode.py

echo "=== NEW hourly analysis (2014-2018) -> outputs_wnet_hourly_IWV  $(date) ==="
MCS_START_YEAR=2004 MCS_END_YEAR=2018 \
  MCS_DIR="$BASE/Outputs_hourly/" \
  MCS_OUTPUT_DIR="$CODE/outputs_wnet_hourly_IWV" \
  MCS_PATTERN='era5dara_hourly_{year}{month:02d}_MCS_detection*.nc' \
  $PY -u mcs-wnet-newfinalcode.py

echo "=== DONE $(date) ==="
echo "  new6h years:   $(ls -d $CODE/outputs_wnet_new6h_IWV/20?? 2>/dev/null | wc -l)"
echo "  hourly years:  $(ls -d $CODE/outputs_wnet_hourly_IWV/20?? 2>/dev/null | wc -l)"
