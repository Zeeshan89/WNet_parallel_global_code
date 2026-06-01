#!/bin/bash -l
#SBATCH --account CLI138
#SBATCH -J miopen_warmup
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH -q normal
#SBATCH -o logs/warmup-%j.txt
#SBATCH -e logs/warmup-%j.txt
#SBATCH -C nvme
#SBATCH --export=NONE
# ─────────────────────────────────────────────────────────────────────────────
# Runs on 1 node / 8 GPUs with DDP so the distributed code paths
# (allreduce, barrier) are exercised — same as real training.
# Only needs to run ONCE. Re-run only if you change model architecture
# or input shapes.
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
export PYTHONNOUSERSITE=1

# ── MIOpen: NORMAL mode — exhaustive search builds complete Find-Db ───────────
MIOPEN_LUSTRE_CACHE="/lustre/orion/proj-shared/cli138/z7a/.miopen-cache"
rm -rf ${MIOPEN_LUSTRE_CACHE}/*          # clear stale partial cache
mkdir -p ${MIOPEN_LUSTRE_CACHE}

export MIOPEN_FIND_MODE=1                # NORMAL: full exhaustive search
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_LOG_LEVEL=0
export MIOPEN_LUSTRE_CACHE=${MIOPEN_LUSTRE_CACHE}

export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET

export MASTER_ADDR=$(hostname -i | tr ' ' '\n' | grep -v ':' | head -n 1)
export MASTER_PORT=3442

mypython=$(which python3)

echo "=========================================="
echo "MIOpen Full-Model Kernel Warmup"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "GPUs      : $SLURM_NTASKS (1 node × 8)"
echo "Cache dir : ${MIOPEN_LUSTRE_CACHE}"
echo "Find mode : 1 (NORMAL — exhaustive, builds complete Find-Db)"
echo "Model     : ImprovedWNet(in_channels=14, num_classes=5)"
echo "Input     : (2, 14, 192, 384) — exact training shape from main.py"
echo "Grad acc  : 3 — exact setting from main.py"
echo "Steps     : 40 — covers all spatial scales in WNet encoder/decoder"
echo "=========================================="
echo "Started: $(date)"
echo ""

srun \
    -N 1 \
    -n 8 \
    -c 7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --export=ALL,MIOPEN_LUSTRE_CACHE=${MIOPEN_LUSTRE_CACHE} \
    $mypython -W ignore -u ./warmup_only.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Warmup completed at : $(date)"
echo "Exit code           : ${EXIT_CODE}"
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ SUCCESS"
    echo "Cache contents:"
    ls -lh ${MIOPEN_LUSTRE_CACHE}/
    echo ""
    echo "Next step: sbatch Run_gpu_frontier_script_distributed.sh"
else
    echo "✗ FAILED — check logs above before submitting training"
fi
echo "=========================================="
exit $EXIT_CODE