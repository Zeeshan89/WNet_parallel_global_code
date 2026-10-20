#!/bin/bash -l
# Run_phase2_train.sh — Phase 2 multi-label W-Net training.
#
# Reads the self-contained Phase 1 split cache, so there is no ERA5 / IBTrACS /
# percentile loading at all — the job is pure GPU work after startup.
#
# Submit:
#   sbatch -q debug Run_phase2_train.sh                       # 1-node pilot
#   sbatch --export=EPOCHS=20,MONTHS=0 Run_phase2_train.sh     # full run

#SBATCH --account=CLI138
#SBATCH -J wnet_p2
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH -o logs/phase2-%j.txt
#SBATCH -e logs/phase2-err-%j.txt
#SBATCH --export=NONE

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

export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29511

EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-8}"
MONTHS="${MONTHS:-72}"          # 0 = use every training month
NCUT="${NCUT:-0.1}"
DISCRETIZE="${DISCRETIZE:-ste}"
WORKERS="${WORKERS:-4}"
ACCUM="${ACCUM:-2}"
LR="${LR:-0.001}"
WARMUP="${WARMUP:-3}"
TAG="${TAG:-}"
# Which label build to train on. Empty = the module default (tensor_cache_v3_split,
# original hurricane label). Set to tensor_cache_v4_split for the Tier B label
# (require_both=True, wind AND vorticity).
export SPLIT_CACHE_DIR="${SPLIT_CACHE_DIR:-}"
# Defaults are sweep config C: per-class alpha/beta/pos_weight scaled to base
# rate. A single global setting cannot serve classes whose base rates span
# 0.3% (hurricane eye) to 6.8% (AR) — config B took the best raw mean IoU while
# collapsing hurricane from 0.237 to 0.079.
ALPHA="${ALPHA:-0.5,0.3,0.4,0.45}"
BETA="${BETA:-0.5,0.7,0.6,0.55}"
PW="${PW:-8,50,25,15}"
CKPT="${CKPT:-/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2}"
EXTRA="${EXTRA:-}"

LIMIT_ARG=""
if [[ "$MONTHS" != "0" ]]; then LIMIT_ARG="--limit-train-months $MONTHS"; fi
TAG_ARG=""
if [[ -n "$TAG" ]]; then TAG_ARG="--tag $TAG"; fi

echo "================================================================================"
echo " Phase 2 training"
echo "   epochs=$EPOCHS batch=$BATCH months=$MONTHS ncut=$NCUT discretize=$DISCRETIZE"
echo "   alpha=$ALPHA beta=$BETA pos_weight=$PW workers=$WORKERS"
echo "   accum=$ACCUM base_lr=$LR warmup=$WARMUP  (nodes=$SLURM_NNODES tasks=$SLURM_NTASKS)"
echo "   ckpt=$CKPT $TAG_ARG"
echo "   started $(date)"
echo "================================================================================"

srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./main_phase2.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --ncut-weight "$NCUT" \
        --discretize "$DISCRETIZE" \
        --tversky-alpha "$ALPHA" \
        --tversky-beta "$BETA" \
        --max-pos-weight "$PW" \
        --num-workers "$WORKERS" \
        --grad-accum "$ACCUM" \
        --lr "$LR" \
        --warmup-epochs "$WARMUP" \
        --ckpt-dir "$CKPT" \
        $LIMIT_ARG $TAG_ARG $EXTRA

RC=$?
echo "Finished $(date), exit $RC"
exit $RC
