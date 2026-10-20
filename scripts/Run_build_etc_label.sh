#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J etc_label
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -o logs/etclabel-%j.txt
#SBATCH -e logs/etclabel-err-%j.txt
#SBATCH --export=NONE
#
# ETC label with closed-low centres (phase4_build_etc_label.py). Three uses:
#   sbatch --nodes=1 --ntasks=4 --time=00:12:00 --export=STAGE=verify Run_build_etc_label.sh
#        legacy path vs channel 8 of tensor_cache_v6_arlit on 200309/200310 (old file
#        format) and 202409/202410 (new); agreement must be about 1.0
#   sbatch --export=STAGE=native Run_build_etc_label.sh
#        native 0.25 deg masks of BOTH labels for 2000-2010 (the tracker-sweep years) into
#        ETC_label_candidates/{legacy,cc}/, for the label-coverage test and the label-ceiling
#        tracker run; 264 tasks on 64 ranks, about 20 min
#   sbatch --nodes=8 --ntasks=384 --ntasks-per-node=48 --cpus-per-task=1 --time=00:30:00 \
#          --export=STAGE=cache Run_build_etc_label.sh
#        tensor_cache_v7_arlit_etccc: copy of v6_arlit with channel 8 replaced (540 months)
# Every stage skips finished months, so a job can be resubmitted.

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
STAGE="${STAGE:-verify}"
OUTDIR="${OUTDIR:-$NB/tensor_cache_v7_arlit_etccc}"
PY="$(which python3) -W ignore -u ./phase4_build_etc_label.py"
echo "stage $STAGE; $(date)"
case $STAGE in
  verify) srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --export=ALL $PY --mode cache --source legacy --verify \
              --src-cache $NB/tensor_cache_v6_arlit --year-list 2003 2024 --months 9 10 ;;
  native) srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --export=ALL $PY --mode native --source cc legacy --years 2000 2010 ;;
  cache)  srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --export=ALL $PY --mode cache --source cc \
              --src-cache $NB/tensor_cache_v6_arlit --outdir "$OUTDIR"
          echo "months written: $(ls $OUTDIR/split_*.npy 2>/dev/null | grep -vc meta.npy) / 540" ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "exit $?  $(date)"
