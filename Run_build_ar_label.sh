#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_cache
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -o logs/arcache-%j.txt
#SBATCH -e logs/arcache-err-%j.txt
#SBATCH --export=NONE
#
# Training cache with the literature-based AR label in channel 6 (phase4_build_ar_label.py).
# Only channel 6 differs from tensor_cache_v5_ibtracs; the 6 predictors and the TC / ETC / MCS
# labels are copied through byte for byte. 540 months over 192 ranks on four nodes (about 3 months
# per rank, 4 GB each, about 10 min); finished months are skipped, so the job can be resubmitted.
#
#   sbatch --nodes=1 --ntasks=4 --time=00:10:00 --export=VERIFY=1 Run_build_ar_label.sh
#                                                       L0 must reproduce the existing channel 6
#                                                       (agreement about 1.0) on 200309/200310
#                                                       (old file format) and 202409/202410 (new)
#   sbatch --export=SOURCE=L6 Run_build_ar_label.sh      build tensor_cache_v6_arlit
#   sbatch --export=SOURCE=L6,STATS=1 Run_build_ar_label.sh   positive rates only, nothing written

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
SOURCE="${SOURCE:-L6}"
OUTDIR="${OUTDIR:-$NB/tensor_cache_v6_arlit}"
PY="$(which python3) -W ignore -u ./phase4_build_ar_label.py --mode cache"

if [ "${VERIFY:-0}" = "1" ]; then
  echo "verification: L0 against the existing channel 6"
  srun -n 4 -c 1 --export=ALL $PY --source L0 --verify --year-list 2003 2024 --months 9 10
elif [ "${STATS:-0}" = "1" ]; then
  srun -n $SLURM_NTASKS -c 1 --export=ALL $PY --source "$SOURCE" --stats-only
else
  echo "building $OUTDIR from $SOURCE; $(date)"
  srun -n $SLURM_NTASKS -c 1 --export=ALL $PY --source "$SOURCE" --outdir "$OUTDIR"
  echo "months written: $(ls $OUTDIR/split_*.npy 2>/dev/null | wc -l) / 540; $(date)"
fi
echo "exit $?"
