#!/bin/bash -l
# Run_build_fullres_cache.sh — build_fullres_cache.py: 2 months at a time per node, each month's time
# steps spread over 28 cores (label code is CPU; ~5 GB/month array plus sources). Resumable; resubmit with the same YEARS until all months exist.
#   sbatch -N 2 --export=ALL,Y0=1980,Y1=1984 Run_build_fullres_cache.sh
#SBATCH --account=CLI138
#SBATCH -J fullres_cache
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH -o logs/fullrescache-%j.txt
#SBATCH -e logs/fullrescache-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE FULLRES_NPROC=$SLURM_CPUS_PER_TASK
N=$SLURM_NTASKS
srun -N $SLURM_JOB_NUM_NODES -n $N -c $SLURM_CPUS_PER_TASK --export=ALL $(which python3) -W ignore -u build_fullres_cache.py \
  --years ${Y0:?} ${Y1:?} ${MONTHS:+--months $MONTHS} ${EXTRA:-} 2>&1 
echo "months: $(ls ../tensor_cache_v11_fullres/split_??????.npy 2>/dev/null | wc -l)/540  $(date)"
