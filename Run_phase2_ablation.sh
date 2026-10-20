#!/bin/bash -l
# Run_phase2_ablation.sh — quantify what N-cut and the discretised bottleneck
# actually contributed to the Phase 2 result.
#
# Two runs, identical to C_full80 except for the one variable under test:
#   NCUT0  ncut_weight 0.1 -> 0     : does the N-cut regulariser do anything?
#   SOFT   discretize ste -> soft   : does the straight-through bottleneck matter,
#                                     or would the leaky soft code score the same?
#
# Both are claims the paper would have to defend, so they need numbers rather
# than an argument from first principles.
#
# Run sequentially in ONE job on purpose: the first run pays the 440 s cold Lustre
# read of the 144 GB working set, and the second then finds it in page cache and
# starts at 19 s/epoch. Two separate jobs would pay that cost twice and queue twice.
#
# Submit:  sbatch --nodes=8 --ntasks=64 --ntasks-per-node=8 --time=01:15:00 Run_phase2_ablation.sh

#SBATCH --account=CLI138
#SBATCH -J wnet_p2_abl
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:15:00
#SBATCH -o logs/p2abl-%j.txt
#SBATCH -e logs/p2abl-err-%j.txt
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
export MASTER_PORT=29517

EPOCHS="${EPOCHS:-80}"
BASE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_ablation

# tag | ncut_weight | discretize | rec_weight ("" = scheduled default)
#   NCUT0  drop the N-cut term
#   SOFT   leaky continuous bottleneck
#   REC0   no reconstruction decoder at all -> plain supervised U-Net.
#          This is the one that says whether the W-Net architecture is justified.
#   BOTH0  neither N-cut nor reconstruction: the pure supervised control
CONFIGS=(
  "REC0|0.1|ste|0.0"
  "BOTH0|0.0|ste|0.0"
)

echo "================================================================================"
echo " Phase 2 ablations — ${EPOCHS} epochs each, all 432 months, 64 GPUs"
echo " control is C_full80: ncut=0.1 discretize=ste  ->  AR .963 Hurr .477 ETC .738 MCS .797"
echo " started $(date)"
echo "================================================================================"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG NCUT DISC RECW <<< "$cfg"
    RECARG=""; if [[ -n "$RECW" ]]; then RECARG="--rec-weight $RECW"; fi
    echo ""
    echo "########## ABLATION $TAG  ncut=$NCUT discretize=$DISC rec=${RECW:-sched} ##########"
    srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
        $mypython -W ignore -u ./main_phase2.py \
            --epochs "$EPOCHS" --batch-size 8 --grad-accum 1 \
            --lr 0.0007 --warmup-epochs 4 \
            --ncut-weight "$NCUT" --discretize "$DISC" \
            --tversky-alpha 0.5,0.3,0.4,0.45 \
            --tversky-beta  0.5,0.7,0.6,0.55 \
            --max-pos-weight 8,50,25,15 \
            --num-workers 4 \
            $RECARG --ckpt-dir "$BASE" --tag "$TAG"
    echo "########## ABLATION $TAG done (exit $?) ##########"
done

echo ""
echo "Finished $(date)"
echo "Results under $BASE/*/history.json"
