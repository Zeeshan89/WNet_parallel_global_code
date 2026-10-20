#!/bin/bash -l
# Run_phase2_fullres_smoke.sh — one short 0.25 deg training epoch on 2 train + 2 val months (1 node,
# debug): catches memory / shape / loader errors before the 8-node run.
#SBATCH --account=CLI138
#SBATCH -J fullres_smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH -o logs/fullres-smoke-%j.txt
#SBATCH -e logs/fullres-smoke-err-%j.txt
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1
export MIOPEN_FIND_MODE=2 MIOPEN_DISABLE_CACHE=0 HSA_FORCE_FINE_GRAIN_PCIE=1 OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29574
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export SPLIT_CACHE_DIR=$NB/tensor_cache_v11_fullres_smoke
srun -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL $(which python3) -W ignore -u ./main_phase2.py \
  --epochs 1 --batch-size ${BS:-2} --ncut-weight 0.1 --ncut-radius 15 --ncut-sigma1 15 --crop-rows 720 \
  --discretize ste --stats predictor_stats.json \
  --tversky-alpha 0.5,0.3,0.4,0.45 --tversky-beta 0.5,0.7,0.6,0.55 --max-pos-weight 8,50,25,15 \
  --num-workers 4 --grad-accum 1 --lr 0.0003 --warmup-epochs 0 --val-stride 8 --train-eval-stride 0 \
  --init-weights ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300/final.pth \
  --ckpt-dir /tmp/fullres_smoke_ckpt --tag smoke --baseline-json phase1_verify_tensor_cache_v11_fullres_baseline.json \
  --select-class ar hurricane etc mcs --select-metric iou recall iou iou --select-min-precision 0.60
echo "exit $? $(date)"
