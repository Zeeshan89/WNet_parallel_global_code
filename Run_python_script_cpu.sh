#!/bin/bash
#SBATCH --job-name=precompute_tensors
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --account=cli138
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zaslam@vols.utk.edu

# ── Environment ───────────────────────────────────────────────────────────────
unset SLURM_EXPORT_ENV
module load DefApps
module load miniforge3/23.11.0-0   # provides conda so 'source activate' works in SLURM
source activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env   # 'source activate' avoids the 'conda init' requirement in SLURM

# Call the env's interpreter by absolute path — on Andes 'source activate' does not
# reliably put the right python on PATH, so pin it explicitly.
mypython=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

# RegCM-CMIP6 runs to process (REGCM_MODEL drives all paths in precompute_tensors.py).
# ACC is already done; loop over the remaining five below. Add ACC to the list to
# rebuild it too.
# 
MODELS=(ACC BCC NOR CNR MPI MRI)

# ── MPI / process config ──────────────────────────────────────────────────────
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS   # = nodes × ntasks-per-node = 32

# Andes has 32 cores per node — each task gets 1 core, 16 tasks per node
echo "Job started at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Total MPI ranks: $SLURM_NTASKS"
echo "Tasks per node:  $SLURM_NTASKS_PER_NODE"

# ── Run ───────────────────────────────────────────────────────────────────────
# srun --ntasks=$SLURM_NTASKS \
#      --cpus-per-task=1 \
#      python precompute_tensors.py \
#          --start_year 1980 \
#          --end_year 2024 \
#          --tensor_dir /lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_v2


# python -u build_iwv_climatology.py

# python -u apply_min_area.py 1995

# python -u mcs-wnet-newfinalcode.py

# Process each model sequentially. Each iteration re-exports REGCM_MODEL and
# launches a fresh srun; the new python processes read REGCM_MODEL at import, so
# every model gets its own input paths, tensor_cache_RegCM_<MODEL>/, and stats.
# for REGCM_MODEL in "${MODELS[@]}"; do
#     export REGCM_MODEL
#     echo ""
#     echo "==================== $REGCM_MODEL  ($(date)) ===================="
#     srun --ntasks=$SLURM_NTASKS --cpus-per-task=1 \
#          $mypython -u ./precompute_tensors.py --start_year 1980 --end_year 2014
#     rc=$?
#     echo "==================== $REGCM_MODEL done (rc=$rc, $(date)) ===================="
#     if [[ $rc -ne 0 ]]; then
#         echo "WARNING: $REGCM_MODEL exited with code $rc — continuing to next model."
#     fi
# done




# ── MCS WNet post-processing over all 6 RegCM models ─────────────────────────
# mcs-wnet-newfinalcode.py takes the model name as a positional arg and writes
# outputs_wnet_standard_<MODEL>/. Plain xarray/scipy script (no MPI). The full
# RegCM/AR domain is ~3x the cells of the old CONUS subset and much slower, so we
# launch the 6 models CONCURRENTLY in the background (each is single-threaded; 6
# fit easily in the node's cores/memory) and wait for all — wall time ~= one model
# instead of the sum, which keeps it inside the SLURM walltime. BCC CNR MPI MRI NOR
MCS_MODELS=( ACC BCC CNR MPI MRI NOR )
pids=()
for MODEL in "${MCS_MODELS[@]}"; do
    echo "==================== launching MCS $MODEL  ($(date)) ===================="
    $mypython -u mcs-wnet-newfinalcode.py "$MODEL" --start-year 2004 --end-year 2014 \
        > "mcs_${MODEL}.log" 2>&1 &
    pids+=($!)
done

# Wait for every model, reporting per-model success/failure (logs in mcs_<MODEL>.log)
for i in "${!MCS_MODELS[@]}"; do
    if wait "${pids[$i]}"; then
        echo "==================== MCS ${MCS_MODELS[$i]} done OK  ($(date)) ===================="
    else
        echo "WARNING: MCS ${MCS_MODELS[$i]} failed — see mcs_${MCS_MODELS[$i]}.log"
    fi
done

# python -u Hurricane_track.py --model ACC --start-year 1980 --end-year 2014
# python -u Hurricane_track.py --model BCC --start-year 1980 --end-year 2014
# python -u Hurricane_track.py --model CNR --start-year 1980 --end-year 2014
# python -u Hurricane_track.py --model MPI --start-year 1980 --end-year 2014


# $mypython -u updated_monthly_ETC_Tracker.py --model ACC --start-year 1980 --end-year 2014
# $mypython -u updated_monthly_ETC_Tracker.py --model BCC --start-year 1980 --end-year 2014
# $mypython -u updated_monthly_ETC_Tracker.py --model CNR --start-year 1980 --end-year 2014
# $mypython -u updated_monthly_ETC_Tracker.py --model MPI --start-year 1980 --end-year 2014
# $mypython -u updated_monthly_ETC_Tracker.py --model MRI --start-year 1980 --end-year 2014
# $mypython -u updated_monthly_ETC_Tracker.py --model NOR --start-year 1980 --end-year 2014


echo "Finished at $(date)"