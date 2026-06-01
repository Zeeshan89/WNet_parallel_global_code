#!/bin/bash
#SBATCH --account CLI138
#SBATCH -J ddp_env_test
#SBATCH -o logs/ddp_test-%j.o
#SBATCH -e logs/ddp_test-%j.e
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 1   # Start with 1 node (8 GPUs)

mkdir -p logs

# Load modules
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Activate environment
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env

# Set master address
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442

# MIOpen cache (avoids disk I/O errors)
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run: 1 node, 8 tasks (1 per GPU), 7 CPU cores per task
srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u test_ddp.py