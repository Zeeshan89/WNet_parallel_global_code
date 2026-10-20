#!/bin/bash -l
# Run_export_hourly_fullres.sh — hourly MCS detections from the 0.25 deg WNet (export_hourly_fullres.py), one block
# of years per job; 16 ranks/node share the 8 GPUs; contiguous months per rank. Resumable.
#   sbatch --export=ALL,Y0=1980,Y1=1994 Run_export_hourly_fullres.sh
#SBATCH --account=CLI138
#SBATCH -J exphr_fr
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH -o logs/exphrfr-%j.txt
#SBATCH -e logs/exphrfr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 HDF5_USE_FILE_LOCKING=FALSE
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
N=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))
srun -N $SLURM_JOB_NUM_NODES -n $N -c 3 --gpus-per-node=8 --export=ALL $(which python3) -W ignore -u \
  export_hourly_fullres.py --years $(seq ${Y0:?} ${Y1:?})
echo "files: $(ls $NB/Global_ARs_Hurr_ETCs_MCS_fullres025_hourly/MCS_WNet_final/*.nc 2>/dev/null | wc -l)/540  $(date)"
