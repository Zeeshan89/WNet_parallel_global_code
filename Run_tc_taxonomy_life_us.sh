#!/bin/bash -l
# Miss taxonomy (phase4_tc_miss_taxonomy.py) for the lifecycle model, one rank per
# held-out year, production tracker env. Writes phase4_results/taxonomy_life_us/tax_Y.json.
#SBATCH --account=CLI138
#SBATCH -J tc_tax_us
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH -o logs/tctaxlifeus-%j.txt
#SBATCH -e logs/tctaxlifeus-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs phase4_results/taxonomy_life_us; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=6 IBTRACS_WIND_SOURCE=any
export TC_MAX_AREA_PX=2000 TC_MIN_DURATION_H=24
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export TC_DET_DIR=$NB/TC_candidates/life_scr40/Hurricanes_WNet_final
export TRK=$NB/TC_candidates/life_scr40/tracks
srun -n 9 -c 6 --export=ALL bash -c '
  Y=(1981 1985 1988 1989 1995 2000 2005 2010 2019); y=${Y[$SLURM_PROCID]}
  python3 -W ignore -u ./phase4_tc_miss_taxonomy.py --years $y --track-dir $TRK \
     --json phase4_results/taxonomy_life_us/tax_$y.json > logs/tctaxlifeus_$y.log 2>&1
  echo "$y rc=$?"'
