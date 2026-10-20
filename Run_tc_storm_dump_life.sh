#!/bin/bash -l
# Per-storm diagnosis for the lifecycle model (I_tclife_scratch40), probabilities and
# tracks from the SAME checkpoint: US box all years + global held-out years.
#SBATCH --account=CLI138
#SBATCH -J tc_dump_life
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH -o logs/tcdumplife-%j.txt
#SBATCH -e logs/tcdumplife-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7 IBTRACS_WIND_SOURCE=any
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CKPT=ARs_Hurricanes_ETCs_MCS/Phase2/I_tclife_scratch40/best_hurricane.pth
TR="life=$NB/TC_candidates/life_scr40/tracks/hurricane"
srun -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL $(which python3) -W ignore -u \
  ./phase4_tc_storm_dump.py --years $(seq 1980 2024) --checkpoint $CKPT --tracks "$TR" \
  --outdir phase4_results/storm_dump_life_usbox
srun -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest --export=ALL $(which python3) -W ignore -u \
  ./phase4_tc_storm_dump.py --years 1983 1987 1994 1998 2003 2007 2012 2016 2021 --checkpoint $CKPT \
  --tracks "$TR" --box -180 180 -45 45 --outdir phase4_results/storm_dump_life_global_ho
echo "usbox $(ls phase4_results/storm_dump_life_usbox | wc -l)/45  global $(ls phase4_results/storm_dump_life_global_ho | wc -l)/9  $(date)"
