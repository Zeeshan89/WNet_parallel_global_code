#!/bin/bash -l
# Run_test_distributed.sh — multi-GPU, file-sharded inference for test.py.
#
# Fans test.py out over many GPUs. Each srun task binds to one GPU and processes
# a disjoint round-robin shard of the monthly files (get_rank_files in test.py,
# same convention training uses). Inference is embarrassingly parallel, so there
# is NO torch.distributed / NCCL / MASTER_ADDR here.
#
# Defaults to the precipitation-based MCS run:
#   checkpoint  Trail_RegCM_PR_<MODEL>/final_checkpoint_1980_2004.pth
#   output      Outputs_RegCM_PR_<MODEL>/
#   years       2005-2014 (held out; training used 1980-2004)
#
# Submit:
#   sbatch --export=REGCM_MODEL=BCC Run_test_distributed.sh
#   sbatch Run_test_distributed.sh
#   sbatch --export=START_YEAR=2004,END_YEAR=2017 Run_test_distributed.sh
#   sbatch --export=CHECKPOINT=/path/to.pth,OUTPUT_DIR=/path/out Run_test_distributed.sh
#
# Scale wider by editing --nodes / --ntasks below (keep ntasks = nodes * 8).
# Sizing: 16 GPUs / ~70 files ≈ 4–5 files/GPU.

#SBATCH --account=CLI138
#SBATCH -J wnet_test_dist
#SBATCH -p batch
#SBATCH -q normal
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/test_dist-%j.txt
#SBATCH -e logs/test_dist-err-%j.txt
#SBATCH --export=NONE

mkdir -p logs
unset SLURM_EXPORT_ENV

# ── Modules (mirrors Run_gpu_frontier_script_distributed.sh) ──────────────────
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

# ── MIOpen ────────────────────────────────────────────────────────────────────
export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_LOG_LEVEL=0

# ── ROCm / HIP ────────────────────────────────────────────────────────────────
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# ── Config (override via --export=KEY=val,...) ────────────────────────────────
# RegCM model drives all input paths + the per-model stats inside test.py.
export REGCM_MODEL="${REGCM_MODEL:-ACC}"
REGCM_BASE="/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"

# Precipitation-based MCS run. These MUST stay on the _PR_ paths: this script
# passes --checkpoint/--output-dir explicitly, so it OVERRIDES test.py's defaults.
# Pointing at Trail_RegCM_<MODEL> would evaluate the BT-era model through the
# precip pipeline, and writing to Outputs_RegCM_<MODEL> would destroy the BT
# results we A/B against. Both are the baseline — read-only.
#   To evaluate the BT baseline instead, pass CHECKPOINT= and OUTPUT_DIR= explicitly.
TRAIL="/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_RegCM/ARs_Hurricanes_ETCs_MCS/Trail_RegCM_PR_${REGCM_MODEL}"

CHECKPOINT="${CHECKPOINT:-${TRAIL}/final_checkpoint_1980_2004.pth}"
DATA_DIR="${DATA_DIR:-${REGCM_BASE}/${REGCM_MODEL}/WNet_6hourly_latlon/}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_RegCM/ARs_Hurricanes_ETCs_MCS/Outputs_RegCM_PR_${REGCM_MODEL}}"
# Held-out test period. Training used 1980-2004, so 2005-2014 is the only truly
# unseen data — the default was 1980 despite the comment claiming otherwise,
# which both leaked the training years into "test" and tripled the runtime.
START_YEAR="${START_YEAR:-1980}"
END_YEAR="${END_YEAR:-2014}"
START_MONTH="${START_MONTH:-1}"    # full year by default (1-12)
END_MONTH="${END_MONTH:-12}"

# Resume by default: months whose 4 output NetCDFs already exist are skipped up-front
# (before any GPU work) and the remaining months are re-sharded across GPUs.
# Set FORCE_REPROCESS=1 to redo every month regardless of existing outputs.
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
FORCE_FLAG=""
if [[ "$FORCE_REPROCESS" == "1" ]]; then
    FORCE_FLAG="--force-reprocess"
fi

mkdir -p "$OUTPUT_DIR"

# Verify checkpoint exists before launching
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "       (training writes final_checkpoint_<START>_<END>.pth — check the year suffix)"
    exit 1
fi

# Guard the BT-era baseline: it holds the verified AR/Hurricane/ETC results and is
# the only reference to A/B the precip run against. Overwriting it is unrecoverable,
# so require an explicit opt-in rather than letting a stale export do it silently.
if [[ "$OUTPUT_DIR" == *"/Outputs_RegCM_${REGCM_MODEL}" && "${ALLOW_BASELINE_OVERWRITE:-0}" != "1" ]]; then
    echo "ERROR: OUTPUT_DIR points at the BT-era baseline: $OUTPUT_DIR"
    echo "       That directory is the A/B reference. Use Outputs_RegCM_PR_${REGCM_MODEL},"
    echo "       or set ALLOW_BASELINE_OVERWRITE=1 if you really mean to replace it."
    exit 1
fi

# ── Job info ──────────────────────────────────────────────────────────────────
JOB_START_TIME=$(date +%s)
echo "================================================================================"
echo " Distributed WNet Inference (file-sharded)"
echo "================================================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-(local)}"
echo "  Started at   : $(date)"
echo "  Nodes        : $SLURM_NODELIST"
echo "  Total tasks  : $SLURM_NTASKS  (GPUs)"
echo "  Checkpoint   : $CHECKPOINT"
echo "  Years        : ${START_YEAR}-${END_YEAR}"
echo "  Months       : ${START_MONTH}-${END_MONTH}"
echo "  Data dir     : $DATA_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "================================================================================"
rocm-smi

# ── Launch: each rank processes its own file shard ────────────────────────────
srun \
    -N $SLURM_NNODES \
    -n $SLURM_NTASKS \
    -c 7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --export=ALL \
    $mypython -W ignore -u ./test.py \
        --checkpoint  "$CHECKPOINT" \
        --data-dir    "$DATA_DIR" \
        --output-dir  "$OUTPUT_DIR" \
        --start-year  "$START_YEAR" \
        --end-year    "$END_YEAR" \
        --start-month "$START_MONTH" \
        --end-month   "$END_MONTH" \
        $FORCE_FLAG

RC=$?

# ── Summary ───────────────────────────────────────────────────────────────────
JOB_END_TIME=$(date +%s)
JOB_ELAPSED=$(( JOB_END_TIME - JOB_START_TIME ))

echo ""
echo "================================================================================"
echo "  Finished at  : $(date)"
echo "  Elapsed      : ${JOB_ELAPSED}s ($((JOB_ELAPSED/60))m / $((JOB_ELAPSED/3600))h)"
echo "  Exit code    : $RC"
echo "  NetCDFs      : $OUTPUT_DIR"
echo "================================================================================"

# Quick sanity report on what was written
echo ""
echo "Output file count per class:"
for cls in AR Hurricane ETCs MCS; do
    n=$(find "$OUTPUT_DIR" -name "*_${cls}_detection.nc" 2>/dev/null | wc -l)
    printf "  %-12s : %d files\n" "$cls" "$n"
done

exit $RC
