#!/bin/bash
#SBATCH --job-name=precompute_PR
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=batch
#SBATCH --account=cli138
#SBATCH --output=precompute_PR_%j.out
#SBATCH --error=precompute_PR_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zaslam@vols.utk.edu

# Stage 6.3 — full precip-MCS tensor cache regeneration.
#
#   sbatch --export=ALL,MODELS=ACC Run_precompute_PR.sh          # ACC first
#   sbatch --export=ALL,MODELS="BCC NOR CNR MPI MRI" Run_precompute_PR.sh
#
# Environment and rank layout mirror Run_python_script_cpu.sh (Andes, 2x16 = 32
# ranks). Writes to tensor_cache_RegCM_PR_<MODEL>/ (set in precompute_tensors.py);
# the BT-era tensor_cache_RegCM_<MODEL>/ is a DIFFERENT path and is never touched,
# so the A/B baseline and the verified AR/Hurricane/ETC weights stay intact.
#
# Timing: 419 files / 32 ranks ~= 13 files/rank x ~23 min ~= 5 h. The 10 h
# walltime has room. Resumable: without --overwrite, already-cached months are
# skipped, so a timeout just needs a resubmit.
#
# Models run SEQUENTIALLY on purpose. Two concurrent 32-rank jobs contend on
# Lustre and roughly halve throughput (that is what wall-killed CNR/MRI before).
#
# Expect exactly ONE skip per model: 198001 has no pr/prc (RegCM SRF output
# starts 1980-02), so the critical-aux gate drops it -> 419 of 420 months.
# Any OTHER "[SKIP] ... no precipitation data" line means a missing PR year file
# and must be investigated, not ignored.

set -u
unset SLURM_EXPORT_ENV
module load DefApps
module load miniforge3/23.11.0-0
source activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env

mypython=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

MODELS=${MODELS:-"ACC"}
START_YEAR=${START_YEAR:-1980}
END_YEAR=${END_YEAR:-2014}

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "Job started at $(date)"
echo "Nodes: $SLURM_NODELIST   ranks: $SLURM_NTASKS"
echo "Models (sequential): $MODELS   years: $START_YEAR-$END_YEAR"

for REGCM_MODEL in $MODELS; do
    export REGCM_MODEL
    echo ""
    echo "==================== $REGCM_MODEL  ($(date)) ===================="
    srun --ntasks=$SLURM_NTASKS --cpus-per-task=1 \
         $mypython -u ./precompute_tensors.py \
             --start_year "$START_YEAR" --end_year "$END_YEAR"
    rc=$?
    echo "==================== $REGCM_MODEL done (rc=$rc, $(date)) ===================="
    if [[ $rc -ne 0 ]]; then
        echo "WARNING: $REGCM_MODEL exited rc=$rc — continuing to the next model."
    fi

    # Per-model tally: 419 expected (198001 legitimately has no precip).
    cache=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_RegCM_PR_${REGCM_MODEL}
    n=$(ls "$cache"/tensors_*.npy 2>/dev/null | grep -c 'tensors_[0-9]\{6\}\.npy')
    echo "  $REGCM_MODEL cached months: $n  (expect 419; 198001 has no pr)"
    if [ -e "$cache/tensors_198001.npy" ]; then
        echo "  !! tensors_198001.npy EXISTS — the aux gate did not fire; its MCS"
        echo "     channels would be all-zero. Delete it before training."
    fi
done

echo ""
echo "Finished at $(date)"
