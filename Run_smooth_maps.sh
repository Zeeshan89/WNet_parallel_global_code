#!/bin/bash -l
# Run_smooth_maps.sh — plot_smooth_vs_nn_maps.py on 1 CPU node.
#SBATCH --account=CLI138
#SBATCH -J smooth_maps
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:20:00
#SBATCH -o logs/smoothmaps-%j.txt
#SBATCH -e logs/smoothmaps-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
srun -n 1 -c 56 --export=ALL $(which python3) -W ignore -u ./plot_smooth_vs_nn_maps.py --workers 48
