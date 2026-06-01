#!/bin/bash -l
#SBATCH --account CLI138
#SBATCH -J miopen_real_warmup
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:50:00
#SBATCH -q normal
#SBATCH -o logs/warmup-real-%j.txt
#SBATCH -e logs/warmup-real-%j.txt
#SBATCH -C nvme
#SBATCH --export=NONE
# ─────────────────────────────────────────────────────────────────────────────
# Real-data MIOpen warmup: train on Jan 1980 for 2 epochs with 1 GPU.
# Uses FIND_MODE=1 (exhaustive) so every conv config gets cached.
# After this, submit the real training with FIND_MODE=3 (HYBRID).
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
unset SLURM_EXPORT_ENV

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

# ── MIOpen: FAST mode — no kernel search, use defaults instantly ──────────────
export MIOPEN_FIND_MODE=2          # FAST: no search, instant start
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_LOG_LEVEL=0

# ── ROCm / HIP ────────────────────────────────────────────────────────────────
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# ── NCCL ──────────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(hostname -i | tr ' ' '\n' | grep -v ':' | head -n 1)
export MASTER_PORT=3442

echo "=========================================="
echo "MIOpen Real-Data Warmup"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Find mode : 2 (FAST — no kernel search)"
echo "Data      : Jan 1980 only, 2 epochs"
echo "=========================================="
echo "Starting at: $(date)"

srun \
    -N 1 \
    -n 1 \
    -c 7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --export=ALL \
    $mypython -W ignore -u ./main.py \
        --start_year  1980 \
        --end_year    1980 \
        --start_month 1 \
        --end_month   1 \
        --epochs      2

TRAIN_EXIT=$?

echo ""
echo "=========================================="
echo "Completed at : $(date)"
echo "Exit code    : ${TRAIN_EXIT}"
if [[ $TRAIN_EXIT -eq 0 ]]; then
    echo "✓ SUCCESS — now submit: sbatch Run_gpu_frontier_script_distributed.sh"
fi
echo "=========================================="
exit $TRAIN_EXIT
