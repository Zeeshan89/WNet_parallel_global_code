#!/bin/bash -l
# Run_etc_union4_fullres.sh — the shipped ETC union3 members + the 0.25 deg WNet S15 tracks as a 4th, lowest-priority member
# -> ETC_candidates/union4_fullres (etc_naec_common/merge_etc_catalogues.py), then the NAECC regional scoring
# (etc_regional_gap.py) for union3, union4 and the 0.25 deg tracks alone. The shipped v2 files are not touched.
#SBATCH --account=CLI138
#SBATCH -J etc_union4fr
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=45
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -o logs/etcunion4fr-%j.txt
#SBATCH -e logs/etcunion4fr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
OUT=$NB/ETC_candidates/union4_fullres
IN="$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2_ETC_tracks_backup_20260925 \
$NB/ETC_candidates/H_final_e300_best/tracks_cc100fb/etc \
$NB/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200/tracks_C200_S15_mslp150/etc \
$NB/Global_ARs_Hurr_ETCs_MCS_fullres025/tracks_etc_S15/etc"
for d in $IN; do echo "member $d: $(ls $d/ETC_tracks_*_meteorological_v2.txt 2>/dev/null | wc -l) years"; done
cd $NB/etc_naec_common
srun -n 45 -c 1 --export=ALL $(which python3) -W ignore -u merge_etc_catalogues.py --out $OUT --inputs $IN | tail -3
echo "merged: $(ls $OUT/ETC_tracks_*_meteorological_v2.txt | wc -l)/45 years"
export ETC_PRODUCTS="union3=$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ETCs_WNet_final union4_fullres=$OUT \
fullres_S15=$NB/Global_ARs_Hurr_ETCs_MCS_fullres025/tracks_etc_S15/etc"
python3 -W ignore -u etc_regional_gap.py 2>&1 | grep -v "ETC tracks [0-9]\|TOTAL:" | tail -30
echo "done $(date)"
