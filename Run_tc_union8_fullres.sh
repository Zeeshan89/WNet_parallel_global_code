#!/bin/bash -l
# Run_tc_union8_fullres.sh — union7 (shipped v2 TC tracks) + the 0.25 deg WNet tracks (fg0_sp0) as an 8th, lowest-priority
# member -> TC_candidates/union8_fullres, then held-out / all-years scoring vs IBTrACS next to union7.
#SBATCH --account=CLI138
#SBATCH -J tc_union8fr
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -o logs/tcunion8fr-%j.txt
#SBATCH -e logs/tcunion8fr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
C=$NB/TC_candidates
IN="$C/life_td_final_fg0/sweep/c_sp0/hurricane $C/life_td_final_fg0/tracks/hurricane $C/life_td_scr40_fg0/tracks/hurricane $C/life_scr40/sweep/fg0/hurricane $C/life_ft40/tracks/hurricane $NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2_TC_tracks_backup_20260924 $C/life_tdet_final_fg0/tracks/hurricane $NB/Global_ARs_Hurr_ETCs_MCS_fullres025/tracks_tc_fg0_sp0/hurricane"
cd $NB/hurricane_analysis
srun -n 1 -c 1 --export=ALL $(which python3) -W ignore -u merge_tc_catalogues.py --out $C/union8_fullres --inputs $IN | tail -3
echo "merged years: $(ls -d $C/union8_fullres/CONUS_year_* | wc -l)"
export TAG=_union8fr IBTRACS_WIND_SOURCE=any WNET_PRODUCTS="union7=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/Hurricanes_WNet_final union8_fullres=$C/union8_fullres fullres_fg0_sp0=$NB/Global_ARs_Hurr_ETCs_MCS_fullres025/tracks_tc_fg0_sp0/hurricane"
python3 -W ignore -u tc_track_compare.py 2>&1 | grep -A8 "=== .* ===" | grep -v "^--" 
echo "done $(date)"
