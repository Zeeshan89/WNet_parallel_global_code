#!/bin/bash -l
# Tracker ocean/genesis gate sweep on one candidate's masks (default life_scr40),
# held-out years only. Output: TC_candidates/$NAME/sweep/<cfg>/hurricane/CONUS_year_Y
#SBATCH --account=CLI138
#SBATCH -J tc_gates
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH -o logs/tcgates-%j.txt
#SBATCH -e logs/tcgates-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export NAME=${NAME:-life_scr40}
export DET=$NB/TC_candidates/$NAME/Hurricanes_WNet_final SW=$NB/TC_candidates/$NAME/sweep
export YEARS="${YEARS:-1983 1987 1994 1998 2003 2007 2012 2016 2021}"
# cfg name : env assignments separated by ";" (NOT ","; sbatch --export splits on commas,
# so pass CFGS through the environment: export CFGS=...; sbatch --export=ALL ...)
export CFGS="${CFGS:-base:X=0 frac0:TC_MIN_OCEAN_FRAC=0 gen0:TC_OCEAN_GENESIS=0 fg0:TC_MIN_OCEAN_FRAC=0;TC_OCEAN_GENESIS=0 fg0lat45:TC_MIN_OCEAN_FRAC=0;TC_OCEAN_GENESIS=0;TC_GENESIS_LAT=45}"
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  i=0
  for c in $CFGS; do for y in $YEARS; do
    if [ $((i % SLURM_NTASKS)) -eq $SLURM_PROCID ]; then
      n=${c%%:*}; e=${c#*:}; out=$SW/$n
      f=$out/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt
      [ -s "$f" ] || env TC_MAX_AREA_PX=2000 IBTRACS_WIND_SOURCE=any TC_MIN_DURATION_H=24 $(echo $e | tr ";" " ") \
          python3 -W ignore -u ./phase4_run_trackers.py --det-dir $DET --out-dir $out --years $y --which hurricane \
          > logs/tcgates_${n}_${y}.log 2>&1
    fi; i=$((i + 1)); done; done
'
for c in $CFGS; do n=${c%%:*}; echo "$n: $(ls $SW/$n/hurricane/CONUS_year_*/*detailed* 2>/dev/null | wc -l) years"; done
