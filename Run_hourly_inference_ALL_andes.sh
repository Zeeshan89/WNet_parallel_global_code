#!/bin/bash -l
# Run_hourly_inference_ALL_andes.sh — full hourly MCS inference, ONE Andes job.
# Shards monthly files across many CPU tasks (test.py get_rank_files). Because the CPU
# forward pass barely scales with threads (~16 s at 4 or 32 threads) and MCS_ONLY skips
# the 93 GB wind load, we pack ~8 month-tasks/node (4 threads each, 32 cores) so ~8 months
# run concurrently per node. Resume-safe: months whose output exists are skipped (no
# --force-reprocess), so a re-run continues after any interruption.
#
# Defaults: 12 nodes x 8 tasks = 96 tasks over 2004-2018 (180 months) -> ~2 waves.
# Override: --export=NODES=..,TPN=..,START_YEAR=..,END_YEAR=..,OUTPUT_DIR=..,DATA_DIR=..
# Submit: sbatch -M andes -D <codedir> [--dependency=afterok:<buildjob>] Run_hourly_inference_ALL_andes.sh

#SBATCH -A cli138
#SBATCH -J wnet_hourly_ALL
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 12
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH -t 16:00:00
#SBATCH -o logs/hourlyALL-%j.txt
#SBATCH -e logs/hourlyALL-err-%j.txt

set -u
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
BASE=$CODE/ARs_Hurricanes_ETCs_MCS
CKPT=$BASE/Trail9/final_checkpoint_2006_2024.pth
PY=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python
cd "$CODE"; mkdir -p logs

# wiring (env-gated in test.py). Defaults = HOURLY; override BT_DIR/BT_PREFIX/
# MCS_MIN_LIFETIME_STEPS/DATA_DIR/OUTPUT_DIR via --export to run the 6-hourly baseline.
export MCS_ONLY=1
export BT_DIR="${BT_DIR:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly/}"
export BT_PREFIX="${BT_PREFIX:-era5_bt_hourly_}"
export MCS_MIN_LIFETIME_STEPS="${MCS_MIN_LIFETIME_STEPS:-6}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export LD_LIBRARY_PATH=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

DATA_DIR="${DATA_DIR:-/lustre/orion/cli138/proj-shared/z7a/Storms_data/data_hourly/}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/Outputs_hourly}"
START_YEAR="${START_YEAR:-2004}"; END_YEAR="${END_YEAR:-2018}"
mkdir -p "$OUTPUT_DIR"

echo "=== HOURLY MCS inference ALL ${START_YEAR}-${END_YEAR}  (MCS_ONLY, lifetime=6) ==="
echo "  nodes=$SLURM_NNODES tasks=$SLURM_NTASKS ($SLURM_CPUS_PER_TASK thr each)  start=$(date)"
echo "  DATA_DIR=$DATA_DIR  OUTPUT_DIR=$OUTPUT_DIR"

# NOTE: no --force-reprocess -> resume-skips months already written (safe to re-run)
# --kill-on-bad-exit=0 : a single task failing (e.g. a memory spike on one month) must
# NOT terminate the other tasks; the failed month is just redone on the next resume run.
srun -N "$SLURM_NNODES" -n "$SLURM_NTASKS" --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-8}" \
     -c "${SLURM_CPUS_PER_TASK:-4}" --kill-on-bad-exit=0 --export=ALL \
    "$PY" -W ignore -u ./test.py \
      --checkpoint "$CKPT" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
      --start-year "$START_YEAR" --end-year "$END_YEAR" \
      --start-month 1 --end-month 12
rc=$?
echo "=== inference done rc=$rc  $(date) ==="
echo "  MCS files: $(find "$OUTPUT_DIR" -name '*_MCS_detection.nc' | wc -l) (expect $(( (END_YEAR-START_YEAR+1)*12 )))"
exit $rc
