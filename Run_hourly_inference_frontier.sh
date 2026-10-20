#!/bin/bash -l
# Run_hourly_inference_frontier.sh — HOURLY MCS inference on FRONTIER GPUs.
# 4 nodes x 8 GPUs = 32 GPU-tasks, file-sharded (test.py get_rank_files) over the
# requested year range. MCS_ONLY skips the 93 GB wind load, so per-task RAM ~21 GB
# (512 GB/node -> no OOM). Resume-safe: months already written are skipped.
#
# Submit ONE job per year-chunk (you drive the 6 chunks), e.g.:
#   sbatch -M frontier -D <codedir> --export=START_YEAR=2004,END_YEAR=2006 Run_hourly_inference_frontier.sh
#   sbatch -M frontier -D <codedir> --export=START_YEAR=2007,END_YEAR=2008 Run_hourly_inference_frontier.sh
#   ... (2009-2010, 2011-2012, 2013-2015, 2016-2018)

#SBATCH --account=CLI138
#SBATCH -J wnet_hourly_fr
#SBATCH -p batch
#SBATCH -q normal
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:15:00
#SBATCH -o logs/hourly_fr-%j.txt
#SBATCH -e logs/hourly_fr-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs
unset SLURM_EXPORT_ENV

# ── Modules (mirror Run_test_distributed.sh) ──────────────────────────────────
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mypython=$(which python3)
export PYTHONNOUSERSITE=1

# ── MIOpen / ROCm ─────────────────────────────────────────────────────────────
export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# ── HOURLY wiring (env-gated in test.py; default 6-hourly untouched) ───────────
export MCS_ONLY=1
export BT_DIR="/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly/"
export BT_PREFIX="era5_bt_hourly_"
export MCS_MIN_LIFETIME_STEPS=6

# ── Config (override START_YEAR/END_YEAR per job via --export) ─────────────────
CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code
CHECKPOINT="${CHECKPOINT:-$CODE/ARs_Hurricanes_ETCs_MCS/Trail9/final_checkpoint_2006_2024.pth}"
DATA_DIR="${DATA_DIR:-/lustre/orion/cli138/proj-shared/z7a/Storms_data/data_hourly/}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE/ARs_Hurricanes_ETCs_MCS/Outputs_hourly}"
START_YEAR="${START_YEAR:-2004}"; END_YEAR="${END_YEAR:-2018}"
mkdir -p "$OUTPUT_DIR"

echo "=== HOURLY MCS (Frontier) years ${START_YEAR}-${END_YEAR}  job=${SLURM_JOB_ID}  $(date) ==="
echo "  nodes=$SLURM_NNODES tasks=$SLURM_NTASKS  DATA_DIR=$DATA_DIR  OUTPUT_DIR=$OUTPUT_DIR"

# --kill-on-bad-exit=0 : one bad month/task must NOT terminate the other 31 tasks.
# no --force-reprocess : resume-skips months already written (safe to re-run).
srun -N $SLURM_NNODES -n $SLURM_NTASKS -c 7 \
     --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=0 --export=ALL \
    $mypython -W ignore -u ./test.py \
        --checkpoint  "$CHECKPOINT" \
        --data-dir    "$DATA_DIR" \
        --output-dir  "$OUTPUT_DIR" \
        --start-year  "$START_YEAR" \
        --end-year    "$END_YEAR" \
        --start-month 1 --end-month 12
rc=$?
echo "=== done rc=$rc  $(date) ==="
echo "  Outputs_hourly MCS files now: $(find "$OUTPUT_DIR" -name '*_MCS_detection.nc' | wc -l)/180"
exit $rc
