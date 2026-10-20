#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_baseline
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH -o logs/arbaseline-%j.txt
#SBATCH -e logs/arbaseline-err-%j.txt
#SBATCH --export=NONE
#
# Trivial single-threshold floor of the NEW AR label. main_phase2.py picks best.pth on skill
# relative to the floors, and the shipped AR floor (0.937, a constant IVT cut reproducing the
# old IVT > 400 label) is wrong for a label with geometry and direction tests. Same two months
# (201401, 201407) as the shipped floors, so the numbers stay comparable. Writes
#   phase1_verify_<cache>.json           the full report
#   phase1_verify_<cache>_baseline.json  --baseline-json for main_phase2.py: AR floor measured
#                                        here, hurricane / etc / mcs kept at the shipped values
#
#   sbatch --dependency=afterok:<cache build> Run_ar_baseline.sh
#   sbatch --export=CACHE=tensor_cache_v6_arlit Run_ar_baseline.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CACHE="${CACHE:-tensor_cache_v6_arlit}"
CLASSES="${CLASSES:-ar}"            # classes whose floor is measured here; e.g. CLASSES="ar etc" for v7
for m in 201401 201407; do
  [ -s "$NB/$CACHE/split_$m.npy" ] || { echo "ERROR: $NB/$CACHE/split_$m.npy missing"; exit 1; }
done
srun -n 1 -c 8 --export=ALL $(which python3) -W ignore -u ./phase1_verify_split.py \
    --cache-dir "$NB/$CACHE" --out "phase1_verify_${CACHE}.json" --baseline-json-for $CLASSES
echo "exit $?"
