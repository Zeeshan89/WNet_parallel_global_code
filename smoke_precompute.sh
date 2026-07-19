#!/bin/bash
#SBATCH --job-name=mcs_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --account=cli138
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

# Stage 6.1 / 6.2 — build ONE month of precip-MCS tensors into a SCRATCH dir and
# check it, BEFORE committing to the full ~419-file cache regeneration.
#
#   sbatch --export=ALL,REGCM_MODEL=ACC smoke_precompute.sh    # 6.1
#   sbatch --export=ALL,REGCM_MODEL=BCC smoke_precompute.sh    # 6.2 (noleap path)
#
# Environment mirrors Run_python_script_cpu.sh (Andes). One file on one rank
# takes ~23 min (the cold path).
#
# NOTE --tensor_dir does NOT fully isolate this run. precompute_tensors.py saves
# to --tensor_dir, but dataset.py's own _precompute_all_channels_cached ALSO
# writes to its class-level TENSOR_CACHE_DIR (tensor_cache_RegCM_PR_<MODEL>), so
# the smoke month lands in the real cache too. That is harmless — it is the same
# verified tensor, byte-identical to the scratch copy, and the full run will
# simply find it already cached — but do not read this script as a dry run.
# It cannot corrupt the BT-era cache (tensor_cache_RegCM_<MODEL>), which is on a
# different path entirely.
#
# What the verifier proves (see verify_smoke_tensors.py):
#   * the new tensor is (N,14,192,384), finite, MCS channels non-empty
#   * ch13 == ch12 and ch11 superset of ch12
#   * channels 0-10 are BIT-IDENTICAL to the BT-era cache -> AR/Hurricane/ETC
#     are provably untouched (their dara inputs never changed)

set -u
unset SLURM_EXPORT_ENV
module load DefApps
module load miniforge3/23.11.0-0
source activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env

mypython=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

MODEL=${REGCM_MODEL:-ACC}
YEAR=${YEAR:-2000}
MONTH=${MONTH:-7}
export REGCM_MODEL=$MODEL

CODE=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_RegCM
SCRATCH=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_SMOKE_${MODEL}

mkdir -p "$SCRATCH"
cd "$CODE"

echo "=== smoke: $MODEL $YEAR-$(printf '%02d' "$MONTH") -> $SCRATCH ==="
echo "started $(date)"

srun --ntasks=1 --cpus-per-task=8 \
    $mypython -u ./precompute_tensors.py \
        --start_year "$YEAR" --end_year "$YEAR" \
        --start_month "$MONTH" --end_month "$MONTH" \
        --tensor_dir "$SCRATCH" --overwrite
rc=$?
echo "precompute rc=$rc  ($(date))"
[ $rc -ne 0 ] && exit $rc

echo
echo "=== verifying ==="
$mypython -u ./verify_smoke_tensors.py "$SCRATCH" "$(printf '%04d%02d' "$YEAR" "$MONTH")"
rc=$?
echo "verify rc=$rc  ($(date))"
exit $rc
