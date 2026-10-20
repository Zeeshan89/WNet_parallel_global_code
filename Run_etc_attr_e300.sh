#!/bin/bash -l
# ETC miss attribution (phase4_etc_miss_attribution.py) for the SHIPPED v2 ETC product
# (H_final_e300/best.pth masks + S15 tracker), 2000-2010, one rank per year.
#SBATCH --account=CLI138
#SBATCH -J etc_attr
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH -o logs/etcattr-%j.txt
#SBATCH -e logs/etcattr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 ETC_MOUNTAIN_FRAME=0 ETC_MOUNTAIN_TRACK_FRAC=1.0
# The track/fragment stages read the finished S15 tracks (--trk-dir). The centre stage calls the
# tracker without an MSLP field, so ETC_CENTER_MODE=mslp cannot be set here: it uses centroids.
export ETC_FILTER_PROFILE=common ETC_CC_DEPTH_PA=150 ETC_MAX_GAP_STEPS=2 ETC_GAP_RADIUS_GROWTH=0
srun -n 11 -c 4 --export=ALL python3 -W ignore -u ./phase4_etc_miss_attribution.py --years 2000 2010    --det-dir /lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/ETC_candidates/H_final_e300_best/ETCs_WNet_final    --trk-dir /lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/ETC_candidates/H_final_e300_best/tracks_S15/etc    --outdir phase4_results/etc_attr_e300_S15
python3 -W ignore -u ./phase4_etc_miss_attribution.py --summarize --outdir phase4_results/etc_attr_e300_S15
