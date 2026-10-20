#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J ar_labels
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH -o logs/arlabels-%j.txt
#SBATCH -e logs/arlabels-err-%j.txt
#SBATCH --export=NONE
#
# Choose the AR training label on evidence. The model reproduces its label, so a candidate
# label is scored against the established AR methods before any training:
#   1. five candidate labels (L0 shipped IVT > 400, L1-L4 literature based) at 0.25 deg for
#      the validation years, one month per rank                       -> AR_label_candidates/
#   2. each candidate through phase4_ar_artmip_v2.py in the "WNet" slot (5 x 8 half-years)
#                                                                     -> evaluations/ar_label_sweep/<Lx>/
#   3. table + AR-days-per-year figure                                -> evaluations/ar_label_sweep/
# One node on purpose; every stage skips what is finished, so the job can be resubmitted.
#
#   sbatch Run_ar_label_sweep.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

export YEARS="1987 1998 2007 2016"
export NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export PY="$(which python3) -W ignore -u"

srun -n 48 -c 1 --export=ALL $PY ./phase4_build_ar_label.py --mode native --year-list $YEARS || exit 1

# 56 ranks = 7 candidates x 8 half-years: each rank poses as rank (r % 8) of 8 for its candidate
srun -n 56 -c 1 --export=ALL bash -c '
  names=(L0 L1 L2 L3 L4 L5 L6); c=${names[$((SLURM_PROCID / 8))]}
  export SLURM_PROCID=$((SLURM_PROCID % 8)) SLURM_NTASKS=8
  $PY ./phase4_ar_artmip_v2.py --pred-dir $NB/AR_label_candidates/$c \
      --outdir evaluations/ar_label_sweep/$c --year-list $YEARS' || exit 1

srun -n 1 -c 1 --export=ALL $PY evaluations/ar_label_sweep/plot_ar_label_sweep.py
echo "exit $?"
