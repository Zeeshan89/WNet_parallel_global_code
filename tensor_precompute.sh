#!/bin/bash -l
#SBATCH --account CLI138
#SBATCH -J tensor_precompute
#SBATCH -p batch
#SBATCH --nodes=8                   # 8 nodes × 8 GPUs = 64 ranks
#SBATCH --ntasks=64                 # 1 rank per GPU
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7            # 7 CPU cores per rank for scipy
#SBATCH --time=02:00:00              # 312 files ÷ 64 ranks = ~4.9 files/rank
                                     # 5 files × 23 min = 115 min + buffer = 2 hrs
#SBATCH -q normal
#SBATCH -o logs/precompute-%j.txt
#SBATCH -e logs/precompute-err-%j.txt
#SBATCH -C nvme
#SBATCH --export=NONE

# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME JOB — run this ONCE before any training job
#
# What this does:
#   Precomputes all 14-channel tensors for every monthly file (1980–2005)
#   Saves them as .npy files to TENSOR_DIR on Lustre
#
# After this job completes:
#   Training loads tensors in ~5 sec instead of ~23 min per file
#   4 nodes / 32 GPUs covers all 312 files in ONE 2-hour training job
#
# Math:
#   312 files ÷ 64 ranks = ~4.9 files/rank → each rank does 4 or 5 files
#   5 files × 23 min/file = 115 min
#   + 20 min buffer (startup, Lustre, IBTrACS load)
#   = ~135 min → well within 2:00:00
#
# Disk space needed:
#   312 files × ~480 MB = ~150 GB in TENSOR_DIR
#   Make sure you have space:
#     lfs quota -u zaslam /lustre/orion/proj-shared/cli138/
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
unset SLURM_EXPORT_ENV

# ── Modules — exactly matching your working test script ───────────────────────
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# ── Conda env ─────────────────────────────────────────────────────────────────
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mypython=$(which python3)
export PYTHONNOUSERSITE=1

# ── MIOpen cache — per job ID to avoid cross-node collisions ──────────────────
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
export MIOPEN_FIND_MODE=1

# ── CPU optimization — scipy uses OpenMP internally ───────────────────────────
export OMP_NUM_THREADS=7             # matches --cpus-per-task
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

# ── MPI / NCCL settings (same as training, for rank detection) ────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export HSA_FORCE_FINE_GRAIN_PCIE=1

# ── Job info ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "TENSOR PRECOMPUTE JOB (ONE-TIME)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Nodes     : $SLURM_NODELIST"
echo "Ranks     : $SLURM_NTASKS  (64 ranks = 64 files in parallel)"
echo "CPUs/rank : $SLURM_CPUS_PER_TASK"
echo "Python    : $mypython"
echo ""
echo "Math:"
echo "  312 files ÷ 64 ranks = ~4.9 files/rank (each rank does 4 or 5)"
echo "  5 files × 23 min/file = 115 min + 20 min startup = ~135 min"
echo "  Wall time 2:00:00 → plenty of buffer"
echo ""
echo "Output:"
echo "  /lustre/orion/proj-shared/cli138/z7a/Storms_data/tensor_cache/"
echo "  ~150 GB total (312 × ~480 MB)"
echo ""
echo "After this job:"
echo "  Training loads in ~5 sec instead of ~23 min per file"
echo "  4 nodes / 32 GPUs covers ALL 312 files in ONE 2-hour training job"
echo "=========================================="

# ── Check disk space before starting ─────────────────────────────────────────
echo ""
echo "Checking disk space..."
lfs quota -u zaslam /lustre/orion/proj-shared/cli138/ 2>/dev/null || \
    df -h /lustre/orion/proj-shared/cli138/ 2>/dev/null || \
    echo "Could not check quota — proceeding anyway"
echo ""

START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

# ── Run precompute ────────────────────────────────────────────────────────────
# Note: --gpus-per-task=1 keeps the same MPI rank detection as training
# The GPUs themselves are NOT used — this is pure CPU scipy work
srun \
    -N  $SLURM_NNODES \
    -n  $SLURM_NTASKS \
    -c  7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    $mypython -u ./precompute_tensors.py \
        --start_year  1980 \
        --end_year    2024 \
        --start_month 1 \
        --end_month   12

EXIT_CODE=$?

# ── Timing summary ────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(echo "scale=1; $ELAPSED / 60"   | bc)
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=========================================="
echo "PRECOMPUTE JOB FINISHED"
echo "Exit code : $EXIT_CODE"
echo "Elapsed   : ${ELAPSED}s  (${MINS} min  /  ${HOURS} hrs)"
echo ""

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ SUCCESS"
    echo ""
    echo "Verify output:"
    echo "  ls -lh /lustre/orion/proj-shared/cli138/z7a/Storms_data/tensor_cache/ | head -20"
    echo "  ls    /lustre/orion/proj-shared/cli138/z7a/Storms_data/tensor_cache/ | wc -l"
    echo "  (expect 624 files: 312 × tensors_YYYYMM.npy + 312 × tensors_YYYYMM.meta.npy)"
    echo ""
    echo "Next step — submit training job:"
    echo "  sbatch run_frontier_4node_cached.sh"
    echo "  (4 nodes, 2 hrs, covers ALL 312 files in ONE job)"
else
    echo "❌ FAILED — check logs/precompute-err-${SLURM_JOB_ID}.txt"
    echo ""
    echo "Common causes:"
    echo "  - Disk quota exceeded (need ~150 GB free)"
    echo "  - Missing HDF5 cache files in CACHE_DIR"
    echo "  - Import error in precompute_tensors.py"
    echo ""
    echo "To retry only failed files (already-cached files are skipped):"
    echo "  sbatch run_precompute.sh"
fi
echo "=========================================="

exit $EXIT_CODE
