#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_artmip
#SBATCH -p batch
#SBATCH --nodes=10
#SBATCH --ntasks=80
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH -o logs/arartmip-%j.txt
#SBATCH -e logs/arartmip-err-%j.txt
#SBATCH --export=NONE
#
# WNet ARs in the context of ARTMIP: six global AR masks (MARS, WNet, Guan-Waliser,
# Mundhenk, TempestLR, Reid500) on one grid, 6-hourly, 1980-2019. 80 ranks, one per
# half-year, about 5 minutes each; finished parts are skipped, so it can be resubmitted.
#
#   sbatch Run_ar_artmip_v2.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

srun -n $SLURM_NTASKS -c 7 --export=ALL \
  $(which python3) -W ignore -u ./phase4_ar_artmip_v2.py --outdir evaluations/ar_artmip_eval_v2
echo "exit $?"; ls evaluations/ar_artmip_eval_v2/part_*.npz 2>/dev/null | wc -l
