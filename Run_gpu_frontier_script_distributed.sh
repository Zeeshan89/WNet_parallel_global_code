#!/bin/bash -l
#SBATCH --account CLI138
#SBATCH -J Storms_dist_4node
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:55:00
#SBATCH -q normal
#SBATCH -o logs/output-%j.txt
#SBATCH -e logs/error-%j.txt
#SBATCH --export=NONE



# ═══════════════════════════════════════════════════════════════════
# TRAINING MODE — set before submitting:
#
#   MODE=single   → one job, all years at once (needs enough wall time)
#   MODE=chain    → auto-submit next job when this one finishes
#
# Usage examples:
#   sbatch --export=MODE=single Run_gpu_frontier_script_distributed.sh
#   sbatch --export=MODE=chain,START_YEAR=1980,END_YEAR=1986 Run_gpu_frontier_script_distributed.sh
#
# Time estimates (6 nodes / 48 GPUs, 100 epochs/file):
#   ~312 files total ÷ 48 GPUs = ~6.25 files/GPU
#   ~6 files × 9 min/file     = ~54 min/GPU
#   Single job needs: ~02:00:00 wall time
#   Chain job needs:  ~02:00:00 per chunk (7 years ≈ 84 files ÷ 48 = ~1.75 files/GPU ≈ 16 min)
#
# ═══════════════════════════════════════════════════════════════════

# ── Timing guide (ACTUAL measured numbers) ────────────────────────────────────
#  6 nodes / 48 GPUs:
#    312 files ÷ 48 GPUs = ~6.25 files/GPU
#    BUT with 2-hr wall limit: 120 min ÷ 33 min = ~3 files/GPU max
#    3 files × 32 GPUs = 96 files per job
#    312 files ÷ 96 = 4 jobs needed → auto-chained below
#
#  Job chain covers:
#    Job 1: 1980–1986  (~84 files,  ~3 files/GPU)  ← submit this one
#    Job 2: 1987–1993  (~84 files,  ~3 files/GPU)  ← auto-submitted
#    Job 3: 1994–2000  (~84 files,  ~3 files/GPU)  ← auto-submitted
#    Job 4: 2001–2005  (~60 files,  ~2 files/GPU)  ← auto-submitted
#
#  WITH tensor cache (after one-time precompute, see precompute_tensors.py):
#    Load/file       : ~5 sec
#    Training/file   : ~10 min
#    Total/file      : ~10.5 min
#    10 files × 10.5 min = 105 min → fits in 2:00:00 with all 312 files!
#    → Only 1 job needed per full training run
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
unset SLURM_EXPORT_ENV

# ── Modules ───────────────────────────────────────────────────────
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

# ── MIOpen ────────────────────────────────────────────────────────
export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_LOG_LEVEL=0

# ── ROCm / HIP ───────────────────────────────────────────────────
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# ── NCCL / RCCL / Gloo ───────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── MASTER_ADDR ───────────────────────────────────────────────────
if [[ $SLURM_NNODES -gt 1 ]]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" \
        | head -n 1 \
        | xargs -I{} getent hosts {} \
        | awk '{print $1}' \
        | grep -v ':' \
        | head -n 1)
else
    export MASTER_ADDR=$(hostname -i | tr ' ' '\n' | grep -v ':' | head -n 1)
fi
export MASTER_PORT=3442

# ── Dataset constants ─────────────────────────────────────────────
FINAL_START_YEAR=2006
FINAL_END_YEAR=2024
YEARS_PER_JOB=7        # used in chain mode only

# ── Mode defaults ─────────────────────────────────────────────────
MODE=${MODE:-chain}    # 'single' or 'chain'

if [[ "$MODE" == "single" ]]; then
    # ── SINGLE JOB: train all years at once ──────────────────────
    START_YEAR=${START_YEAR:-$FINAL_START_YEAR}
    END_YEAR=${END_YEAR:-$FINAL_END_YEAR}
    START_MONTH=${START_MONTH:-1}
    END_MONTH=${END_MONTH:-12}

    echo "=========================================="
    echo "MODE         : SINGLE JOB (all years)"
    echo "Range        : ${START_YEAR}/${START_MONTH} – ${END_YEAR}/${END_MONTH}"
    echo "Files est.   : ~$(( (END_YEAR - START_YEAR + 1) * 12 )) monthly files"
    echo "GPUs         : $SLURM_NTASKS"
    echo "Files/GPU    : ~$(( (END_YEAR - START_YEAR + 1) * 12 / SLURM_NTASKS ))"
    echo "=========================================="

elif [[ "$MODE" == "chain" ]]; then
    # ── CHAIN MODE: this job handles one chunk, submits next ──────
    START_YEAR=${START_YEAR:-$FINAL_START_YEAR}
    END_YEAR=${END_YEAR:-$((FINAL_START_YEAR + YEARS_PER_JOB - 1))}
    START_MONTH=${START_MONTH:-1}
    END_MONTH=${END_MONTH:-12}

    # Cap at final year
    if [[ $END_YEAR -gt $FINAL_END_YEAR ]]; then
        END_YEAR=$FINAL_END_YEAR
    fi

    # Pre-submit next chunk before training starts
    NEXT_START_YEAR=$((END_YEAR + 1))
    NEXT_END_YEAR=$((END_YEAR + YEARS_PER_JOB))

    if [[ $NEXT_START_YEAR -le $FINAL_END_YEAR ]]; then
        if [[ $NEXT_END_YEAR -gt $FINAL_END_YEAR ]]; then
            NEXT_END_YEAR=$FINAL_END_YEAR
        fi
        echo "Pre-submitting next chunk: years=${NEXT_START_YEAR}-${NEXT_END_YEAR}"
        sbatch \
            --dependency=afterany:${SLURM_JOB_ID} \
            --export=MODE=chain,START_YEAR=${NEXT_START_YEAR},END_YEAR=${NEXT_END_YEAR},START_MONTH=1,END_MONTH=12 \
            Run_gpu_frontier_script_distributed.sh
    else
        echo "Chain complete — final chunk: ${START_YEAR}-${END_YEAR}"
    fi

    echo "=========================================="
    echo "MODE         : CHAIN (chunk)"
    echo "This chunk   : ${START_YEAR}/${START_MONTH} – ${END_YEAR}/${END_MONTH}"
    echo "Files est.   : ~$(( (END_YEAR - START_YEAR + 1) * 12 )) monthly files"
    echo "GPUs         : $SLURM_NTASKS"
    echo "Files/GPU    : ~$(( (END_YEAR - START_YEAR + 1) * 12 / SLURM_NTASKS ))"
    echo "=========================================="

else
    echo "ERROR: Unknown MODE='$MODE'. Use MODE=single or MODE=chain."
    exit 1
fi

# ── Job info ──────────────────────────────────────────────────────
JOB_START_TIME=$(date +%s)
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_NODELIST"
echo "MASTER_ADDR  : $MASTER_ADDR"
echo "MASTER_PORT  : $MASTER_PORT"
echo "Total tasks  : $SLURM_NTASKS"
rocm-smi
echo "Job started at: $(date)"

# ── Launch ────────────────────────────────────────────────────────
srun \
    -N $SLURM_NNODES \
    -n $SLURM_NTASKS \
    -c 7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --export=ALL \
    $mypython -W ignore -u ./main.py \
        --start_year  $START_YEAR \
        --end_year    $END_YEAR \
        --start_month $START_MONTH \
        --end_month   $END_MONTH

TRAIN_EXIT=$?

# ── Summary ───────────────────────────────────────────────────────
JOB_END_TIME=$(date +%s)
JOB_ELAPSED=$(( JOB_END_TIME - JOB_START_TIME ))
JOB_MINS=$(echo "scale=1; $JOB_ELAPSED / 60" | bc)
JOB_HOURS=$(echo "scale=2; $JOB_ELAPSED / 3600" | bc)

echo ""
echo "=========================================="
echo "Job completed at  : $(date)"
echo "Total elapsed     : ${JOB_ELAPSED}s (${JOB_MINS}m / ${JOB_HOURS}h)"
echo "Mode              : ${MODE}"
echo "Training years    : ${START_YEAR}-${END_YEAR}"
echo "Exit code         : $TRAIN_EXIT"
echo "=========================================="

exit $TRAIN_EXIT