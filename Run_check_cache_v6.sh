#!/bin/bash -l
#SBATCH --account=CLI138
#SBATCH -J chk_cache
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH -o logs/chkcache-%j.txt
#SBATCH -e logs/chkcache-err-%j.txt
#SBATCH --export=NONE
#
# tensor_cache_v6_arlit must differ from tensor_cache_v5_ibtracs in channel 6 ONLY: the six
# predictors, the TC / ETC / MCS labels and the time axis byte-identical. Six months, one per
# rank (both file formats, first and last year).
#
#   sbatch Run_check_cache_v6.sh

mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export NEW="${NEW:-tensor_cache_v6_arlit}" OLD="${OLD:-tensor_cache_v5_ibtracs}"

srun -n 6 -c 2 --export=ALL $(which python3) -W ignore -u - <<'PY'
import os, numpy as np
NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
months = ["198001", "199407", "200309", "201612", "202408", "202412"]
m = months[int(os.environ.get("SLURM_PROCID", 0))]
new = np.load(f"{NB}/{os.environ['NEW']}/split_{m}.npy", mmap_mode="r")
old = np.load(f"{NB}/{os.environ['OLD']}/split_{m}.npy", mmap_mode="r")
tn = np.load(f"{NB}/{os.environ['NEW']}/split_{m}.meta.npy", allow_pickle=True)
to = np.load(f"{NB}/{os.environ['OLD']}/split_{m}.meta.npy", allow_pickle=True)
same = [bool(np.array_equal(new[:, c], old[:, c])) for c in range(10)]
ar_new, ar_old = np.asarray(new[:, 6]) > 0.5, np.asarray(old[:, 6]) > 0.5
print(f"{m}: shape {new.shape} == {old.shape}: {new.shape == old.shape}; meta identical: {np.array_equal(tn, to)}; "
      f"channels identical: {''.join('=' if s else 'X' for s in same)} (expect =====X===  i.e. only channel 6 differs); "
      f"AR positive rate {ar_old.mean():.4f} -> {ar_new.mean():.4f}; label values in [0,1]: {float(new[:, 6].min()):.2f}..{float(new[:, 6].max()):.2f}; "
      f"finite: {bool(np.isfinite(np.asarray(new[:, 6])).all())}", flush=True)
PY
echo "exit $?"
