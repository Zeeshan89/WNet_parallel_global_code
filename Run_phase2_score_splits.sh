#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J p2_score
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH -o logs/p2score-%j.txt
#SBATCH -e logs/p2score-err-%j.txt
#SBATCH --export=NONE
#
# Score one Phase-2 checkpoint on the train / val / test splits with the trainer's
# own validation code (phase2_score_split.py). Produces three JSONs next to the
# checkpoint that the paper's fig 6 reads for its training-vs-validation panels.
#
#   sbatch Run_phase2_score_splits.sh                       # F_ibtracs_pwH50_e200/final.pth
#   sbatch --export=ALL,RUN=<run dir>,CKPT=best.pth Run_phase2_score_splits.sh
#
# GPU environment copied from Run_phase2_train.sh so the forward pass is identical.

mkdir -p logs; unset SLURM_EXPORT_ENV
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
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517

RUN=${RUN:-ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200}
CKPT=${CKPT:-final.pth}
STRIDE=${STRIDE:-4}
TAG=${CKPT%.pth}
echo "run=$RUN  ckpt=$CKPT  stride=$STRIDE  $(date)"

for SPLIT in val train test; do
  echo "############ $SPLIT ############"
  srun -n $SLURM_NTASKS -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL \
    $mypython -W ignore -u ./phase2_score_split.py \
      --checkpoint "$RUN/$CKPT" --split $SPLIT --stride $STRIDE \
      --out "$RUN/score_${TAG}_${SPLIT}_stride${STRIDE}.json"
  echo "  -> exit $?"
done
echo "done $(date)"; ls -la "$RUN"/score_*.json
