#!/bin/bash -l
# Run_tc_track_fullres.sh — TC tracks from the 0.25 deg WNet masks (Global_ARs_Hurr_ETCs_MCS_fullres025/
# Hurricanes_WNet_final), fixed tracker, two configurations, then held-out scoring vs IBTrACS with
# hurricane_analysis/tc_track_compare.py next to the shipped union and its single-model counterpart.
#   prod    : TC_MAX_AREA_PX=2000 TC_MIN_DURATION_H=24 IBTRACS_WIND_SOURCE=any (production)
#   fg0_sp0 : prod + TC_MIN_OCEAN_FRAC=0 TC_OCEAN_GENESIS=0 TC_MIN_SPEED_MS=0 (the union's lead-member gates)
# One (config, year) per rank; finished years are skipped.
#SBATCH --account=CLI138
#SBATCH -J tctrk_fr
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=90
#SBATCH --ntasks-per-node=45
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH -o logs/tctrkfr-%j.txt
#SBATCH -e logs/tctrkfr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export FR=$NB/Global_ARs_Hurr_ETCs_MCS_fullres025
export DET=$FR/Hurricanes_WNet_final TC_MAX_AREA_PX=2000 TC_MIN_DURATION_H=24 IBTRACS_WIND_SOURCE=any
srun -n 90 -c 1 --export=ALL bash -c '
  cfgs=(prod fg0_sp0); c=${cfgs[$((SLURM_PROCID / 45))]}; y=$((1980 + SLURM_PROCID % 45))
  [ "$c" = fg0_sp0 ] && export TC_MIN_OCEAN_FRAC=0 TC_OCEAN_GENESIS=0 TC_MIN_SPEED_MS=0
  OUTD=$FR/tracks_tc_$c
  f="$OUTD/hurricane/CONUS_year_${y}/CONUS_hurricane_tracks_detailed_${y}.txt"
  [ -s "$f" ] && exit 0
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$OUTD" --years $y --which hurricane \
     > "logs/tctrkfr_${c}_${y}.log" 2>&1; echo "[$c $y] rc=$?"'
for c in prod fg0_sp0; do echo "$c: $(ls $FR/tracks_tc_$c/hurricane/CONUS_year_*/CONUS_hurricane_tracks_detailed_*.txt 2>/dev/null | wc -l)/45 years"; done
echo "=== held-out scoring vs IBTrACS ==="
cd $NB/hurricane_analysis
export TAG=_fullres WNET_PRODUCTS="shipped_union7=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final \
single_fg0=$NB/TC_candidates/life_td_final_fg0/tracks/hurricane single_fg0_sp0=$NB/TC_candidates/life_td_final_fg0/sweep/c_sp0/hurricane \
fullres_prod=$FR/tracks_tc_prod/hurricane fullres_fg0_sp0=$FR/tracks_tc_fg0_sp0/hurricane"
python3 -W ignore -u tc_track_compare.py 2>&1 | tail -80
echo "done $(date)"
