#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J etc_terrain
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:12:00
#SBATCH -o logs/etcterrain-%j.txt
#SBATCH -e logs/etcterrain-err-%j.txt
#SBATCH --export=NONE
#
# Share of track points over the Rockies / Sierra / Plains lee / Greenland for the reference
# catalogues, the shipped WNet tracker and the tracker-sweep variants (phase4_etc_terrain_share.py).
# A batch job because the NAECC csv is 1.9 GB. Writes phase4_results/etc_terrain_share.txt.
#
#   sbatch Run_etc_terrain_share.sh
#   sbatch --export=ALL,VARIANTS="S0_base S11_common_mslp_gap2_nogrow" Run_etc_terrain_share.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg ETC_VERSION=v2

srun -n 1 -c 16 --export=ALL $(which python3) -W ignore -u ./phase4_etc_terrain_share.py --years 2000 2010 ${VARIANTS:+--variants $VARIANTS}
echo "exit $?"
