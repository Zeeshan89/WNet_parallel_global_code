#!/bin/bash -l
# Run_phase2_sweep.sh — precision/recall rebalance sweep for Phase 2.
#
# The first pilot came out at recall ~0.99 with precision ~0.55 on AR/ETC/MCS:
# pos_weight and Tversky beta BOTH push recall and they compound. AR suffered
# most and landed below its own single-threshold baseline (0.937), because for a
# class whose optimal decision is a crisp threshold, over-prediction costs IoU
# directly.
#
# Four configs, run sequentially in one job (each ~4 min at 72 months / 6 epochs).
#
# Submit:  sbatch -q debug Run_phase2_sweep.sh

#SBATCH --account=CLI138
#SBATCH -J wnet_p2_sweep
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:25:00
#SBATCH -o logs/p2sweep-%j.txt
#SBATCH -e logs/p2sweep-err-%j.txt
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
export MASTER_PORT=29513

BASE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2_sweep
EPOCHS="${EPOCHS:-6}"
MONTHS="${MONTHS:-72}"

# tag | alpha | beta | max_pos_weight
#   A  reproduce the pilot, as the control
#   B  Dice-like, mild pos_weight — least recall pressure
#   C  per-class, scaled to base rate: AR (6.8% of globe) gets no push,
#      hurricane (0.3%) keeps a strong one, ETC/MCS in between
#   D  per-class but with the recall push removed from AR entirely
CONFIGS=(
  "A|0.3|0.7|50"
  "B|0.5|0.5|10"
  "C|0.5,0.3,0.4,0.45|0.5,0.7,0.6,0.55|8,50,25,15"
  "D|0.6,0.3,0.4,0.45|0.4,0.7,0.6,0.55|4,50,25,15"
)

echo "================================================================================"
echo " Phase 2 precision/recall sweep — ${#CONFIGS[@]} configs, ${EPOCHS} epochs, ${MONTHS} months"
echo " started $(date)"
echo "================================================================================"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG ALPHA BETA PW <<< "$cfg"
    echo ""
    echo "########## CONFIG $TAG  alpha=$ALPHA beta=$BETA pos_weight=$PW ##########"
    srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
        $mypython -W ignore -u ./main_phase2.py \
            --epochs "$EPOCHS" --batch-size 8 \
            --limit-train-months "$MONTHS" \
            --tversky-alpha "$ALPHA" --tversky-beta "$BETA" \
            --max-pos-weight "$PW" \
            --ckpt-dir "$BASE" --tag "$TAG"
    echo "########## CONFIG $TAG done (exit $?) ##########"
done

echo ""
echo "Finished $(date)"
echo "Results under $BASE/{A,B,C,D}/history.json"
