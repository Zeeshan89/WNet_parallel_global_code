#!/bin/bash
#SBATCH --job-name=precompute_tensors
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --partition=batch
#SBATCH --account=cli138
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zaslam@vols.utk.edu

# ── Environment ───────────────────────────────────────────────────────────────
unset SLURM_EXPORT_ENV
module load DefApps
source activate my_env        # use 'source activate' not 'conda activate' in SLURM

# ── MPI / process config ──────────────────────────────────────────────────────
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS   # = nodes × ntasks-per-node = 32

# Andes has 32 cores per node — each task gets 1 core, 16 tasks per node
echo "Job started at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Total MPI ranks: $SLURM_NTASKS"
echo "Tasks per node:  $SLURM_NTASKS_PER_NODE"

# ── Run ───────────────────────────────────────────────────────────────────────
srun --ntasks=$SLURM_NTASKS \
     --cpus-per-task=1 \
     python precompute_tensors.py \
         --start_year 1980 \
         --end_year 2024 \
         --tensor_dir /lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_v2


echo "Finished at $(date)"