#!/bin/bash -l
# Run_hourly_test_andes.sh — HOURLY MCS inference on ANDES CPU (Phase 2).
# Runs the trained WNet with MCS_ONLY=1 (ch0-10 zeroed) on hourly inputs:
#   BT      = OLR/brightness_temp_hourly/era5_bt_hourly_{year}.nc   (BT_DIR + BT_PREFIX)
#   moisture= data_hourly[...]/era5dara_hourly_{YYYYMM}.nc          (DATA_DIR)
#   climatol= existing iwv_monthly_climatology.nc                   (reused)
#   lifetime= 6 hourly steps = 6 h                                  (MCS_MIN_LIFETIME_STEPS)
# Override via --export=DATA_DIR=..,OUTPUT_DIR=..,START_MONTH=..,END_MONTH=..,NTASKS=..
#
# Submit: sbatch -M andes -D <codedir> Run_hourly_test_andes.sh

#SBATCH -A cli138
#SBATCH -J wnet_hourly
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:30:00
#SBATCH -o logs/hourly-%j.txt
#SBATCH -e logs/hourly-err-%j.txt

set -u
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
BASE=$CODE/ARs_Hurricanes_ETCs_MCS
CKPT=$BASE/Trail9/final_checkpoint_2006_2024.pth
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

cd "$CODE"; mkdir -p logs

# ── hourly wiring (env-gated overrides read by test.py) ──
export MCS_ONLY=1
export BT_DIR="/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly/"
export BT_PREFIX="era5_bt_hourly_"
export MCS_MIN_LIFETIME_STEPS="${MCS_MIN_LIFETIME_STEPS:-6}"

DATA_DIR="${DATA_DIR:-/lustre/orion/cli138/proj-shared/z7a/Storms_data/data_hourly_smoke/}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/Outputs_hourly_smoke}"
START_YEAR="${START_YEAR:-2004}"; END_YEAR="${END_YEAR:-2004}"
START_MONTH="${START_MONTH:-7}"; END_MONTH="${END_MONTH:-7}"
NTASKS="${NTASKS:-1}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
mkdir -p "$OUTPUT_DIR"

echo "=== HOURLY MCS (Andes CPU) ==="
echo "  job=${SLURM_JOB_ID:-local} node=$SLURM_NODELIST tasks=$NTASKS x $OMP_NUM_THREADS thr"
echo "  MCS_ONLY=$MCS_ONLY  min_lifetime_steps=$MCS_MIN_LIFETIME_STEPS"
echo "  BT_DIR=$BT_DIR  BT_PREFIX=$BT_PREFIX"
echo "  DATA_DIR=$DATA_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR   window=$START_YEAR-$START_YEAR m$START_MONTH-$END_MONTH"
echo "  start=$(date)"

srun -N 1 -n "$NTASKS" -c "$OMP_NUM_THREADS" --export=ALL \
    "$PY" -W ignore -u ./test.py \
      --checkpoint "$CKPT" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
      --start-year "$START_YEAR" --end-year "$END_YEAR" \
      --start-month "$START_MONTH" --end-month "$END_MONTH" \
      --force-reprocess
rc=$?
echo "=== done rc=$rc  $(date) ==="
find "$OUTPUT_DIR" -name '*_MCS_detection.nc' -printf '  %f\n'
exit $rc
