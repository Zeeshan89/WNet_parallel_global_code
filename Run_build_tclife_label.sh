#!/bin/bash -l
# Run_build_tclife_label.sh — build tensor_cache_v8_tclife: the v7 cache (the one
# H_final_e300_single was trained on) with ONLY channel 7 replaced by the
# FULL-LIFECYCLE IBTrACS TC label (phase4_build_ibtracs_label.py --lifecycle).
#
# Why: the >= 34 kt label switches off where a storm weakens over land, so WNet
# learned to go silent at landfall. Measured in the advisor's US box (all years):
# coverage of < 34 kt land positions WNet 0.28 vs TempestExtremes 0.70, TRACK 0.98,
# while >= 34 kt land positions are covered at 0.91.
#
# Resumable per month; 2-node chunks. Resubmit the same script until 540/540.
#   sbatch Run_build_tclife_label.sh
#SBATCH --account=CLI138
#SBATCH -J tclife_label
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH -o logs/tclife-%j.txt
#SBATCH -e logs/tclife-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0; module load rocm/7.1.1; module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=7

NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
export SRC="${SRC:-$NB/tensor_cache_v7_arlit_etccc}"
export OUTDIR="${OUTDIR:-$NB/tensor_cache_v8_tclife}"
export EXTRA="${EXTRA:---lifecycle}"
export Y0="${Y0:-1980}"; export Y1="${Y1:-2024}"
mkdir -p "$OUTDIR"
export NM=$(( (Y1 - Y0 + 1) * 12 ))
echo "TC lifecycle label: $SRC -> $OUTDIR  ($EXTRA)  $NM months / $SLURM_NTASKS ranks  $(date)"

# A rank that runs out of walltime leaves a .tmp file, never a partial split_*.npy
# (the builder writes to tmp and renames), so a resubmission redoes that month only.
srun -n $SLURM_NTASKS -c 7 --export=ALL bash -c '
  r=$SLURM_PROCID; n=$SLURM_NTASKS; i=$r
  while [ "$i" -lt "$NM" ]; do
    y=$((Y0 + i / 12)); m=$((i % 12 + 1))
    if [ ! -s "$OUTDIR/split_$(printf %04d%02d $y $m).npy" ]; then
      python3 -W ignore -u ./phase4_build_ibtracs_label.py --year $y --month $m \
          --src-cache "$SRC" --outdir "$OUTDIR" $EXTRA \
          >> "logs/tclife_rank${r}.log" 2>&1 || echo "[rank $r] FAILED $y-$m"
    fi
    i=$((i + n))
  done
  echo "[rank $r] done"
'
rm -f "$OUTDIR"/*.tmp* 2>/dev/null
N=$(ls "$OUTDIR"/split_*.npy 2>/dev/null | grep -vc meta.npy)
echo "months built: $N / $NM   $(date)"
[ "$N" -ge "$NM" ] || echo "INCOMPLETE — resubmit: sbatch $0"
