#!/bin/bash -l
# Run_etc_track_fullres.sh — ETC tracks from the 0.25 deg WNet masks with the S15 tracker settings (as the shipped
# product), then the NAECC regional scoring (etc_naec_common/etc_regional_gap.py) next to the shipped v2 union and
# the single-model H_final_e300 S15 tracks. One year per rank; finished years are skipped.
#SBATCH --account=CLI138
#SBATCH -J etctrk_fr
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=45
#SBATCH --ntasks-per-node=23
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o logs/etctrkfr-%j.txt
#SBATCH -e logs/etctrkfr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export FR=$NB/Global_ARs_Hurr_ETCs_MCS_fullres025
export DET=$FR/ETCs_WNet_final OUTD=$FR/tracks_etc_S15
export ETC_FILTER_PROFILE=common ETC_CENTER_MODE=mslp ETC_CC_DEPTH_PA=150 ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0
export ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0
env | grep "^ETC_" | sort
srun -n 45 -c 2 --export=ALL bash -c '
  y=$((1980 + SLURM_PROCID))
  [ -s "$OUTD/etc/ETC_tracks_${y}_meteorological_v2.txt" ] && exit 0
  python3 -W ignore -u ./phase4_run_trackers.py --det-dir "$DET" --out-dir "$OUTD" --years $y --which etc \
      > "logs/etctrkfr_y${y}.log" 2>&1; echo "[$y] rc=$?"'
echo "tracked: $(ls $OUTD/etc/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l)/45 years"
echo "=== NAECC regional scoring (1980-2020) ==="
cd $NB/etc_naec_common
export ETC_PRODUCTS="shipped_union3=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final \
single_S15=$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2/H_final_e300/tracks_v3_S15/etc \
fullres_S15=$OUTD/etc"
python3 -W ignore -u etc_regional_gap.py 2>&1 | tail -40
echo "done $(date)"
