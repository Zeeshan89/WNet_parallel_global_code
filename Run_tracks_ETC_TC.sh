#!/bin/bash
#SBATCH --job-name=tracks_etc_tc
#SBATCH -A cli138
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zaslam@vols.utk.edu

# ─────────────────────────────────────────────────────────────────────────────
# ONE job, ANDES. Submit from a Frontier login node with:
#
#     sbatch -M andes Run_tracks_ETC_TC.sh
#
# The local Slurm federates andes/dtn/frontier, so -M works without ssh. Do NOT
# submit this to Frontier: Frontier's submit filter rejects a >2 h walltime on a
# 1-node job, which surfaces only as the opaque
# "Batch job submission failed: Unspecified error".
#
# Everything runs in a single job because only one job slot is available, so the
# stages are sequenced here rather than split across submissions.
#
# STAGE 1  TC tracker, all six models, 1980-2014 (concurrent)
# STAGE 2  ETC tracker, all six models, 1980-2014 (concurrent)
# STAGE 3  ETC comparison figures
# STAGE 4  TC comparison figures
#
# WHY STAGE 1 EXISTS
#   TC intensity was NaN in the previously-written track files. This was NOT a code
#   bug: Hurricane_track.py was fixed at 2026-07-18 10:01 but the outputs on disk
#   were written at 09:45, i.e. they predate the fix. A 1-year ACC smoke run against
#   the current code yields finite Vorticity / MSLP / WindSpeed, so the outputs are
#   simply stale. The pre-existing files are backed up under
#   backup_tracks_pre_intensity_rerun/ in case the 10:01 edit also changed geometry.
#
# WHY BOTH TRACKERS ARE RE-RUN: THE DOMAIN MASK
#   ~28% of the lat/lon bounding box lies outside RegCM's curved Lambert domain and
#   is NaN in every driving field. The WNet still emitted detections there (the
#   tensors were built with fillna(0.0) and the convolutions zero-pad), and the
#   result dominated the TC dataset: 45% of ALL track points were pinned to the
#   single row lat=18.75, and only 34% of points had finite MSLP.
#
#   Both trackers now zero the detection field outside the domain BEFORE any blob is
#   formed, using the driving field's own NaN pattern as the mask (it follows the
#   true curved boundary exactly, with no arbitrary edge-row constant), plus a
#   2-cell erosion for the padding response that reaches inside. See
#   Hurricane_track._domain_mask and ETCTracker._domain_mask.
#
#   Note test.py::clean_edge_artifacts CANNOT fix this — it is written for a global
#   periodic field where the first/last rows are poles, so on this regional domain it
#   propagates the edge contamination outward instead of erasing it.
#
#   Verified on ACC: worst-case coordinate pinning 45% -> 5.7%, MSLP-finite
#   34% -> 100%, TC counts 59-71/yr -> 26-39/yr against ~20/yr in IBTrACS. ETC is
#   barely affected (98.9% of its points were already in-domain) but loses the
#   spurious ~1300 km end-of-track jumps onto the cell (-61.50, 28.00).
#   Pre-fix tracks are preserved in backup_tracks_pre_intensity_rerun/.
#
# COST (measured on a Frontier login node; Andes is slower, hence 6 h)
#   TC ~19 s per model-year -> ~11 min/model, ~25-40 min for six concurrent.
#   ETC ~4.6 s per model-year -> all six models in ~2 min concurrently.
#   Figures ~10-20 min each over the full 35-year record.
# ─────────────────────────────────────────────────────────────────────────────

unset SLURM_EXPORT_ENV
module load DefApps
module load miniforge3/23.11.0-0
source activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env

# Absolute interpreter path — on Andes 'source activate' does not reliably put the
# right python on PATH.
mypython=/lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env/bin/python

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Job started at $(date)"
echo "Running on: $SLURM_NODELIST"
$mypython -c "import sys; print('python:', sys.executable)"

# ── STAGE 1: TC tracker, six models concurrently ─────────────────────────────
# Each is single-threaded xarray/scipy at ~400 MB RSS, so six fit comfortably on
# one node and wall time is one model rather than the sum. This mirrors the MCS
# block in Run_python_script_cpu.sh, which completed all six in ~25 min. (The
# sequential-run warning in Run_precompute_PR.sh concerns Lustre contention between
# 32-rank MPI jobs and does not apply here.)
TC_MODELS=( ACC BCC CNR MPI MRI NOR )
pids=()
for MODEL in "${TC_MODELS[@]}"; do
    echo "==================== launching TC $MODEL  ($(date)) ===================="
    $mypython -u Hurricane_track.py --model "$MODEL" \
        --start-year 1980 --end-year 2014 \
        > "tc_track_${MODEL}.log" 2>&1 &
    pids+=($!)
done

tc_failed=0
for i in "${!TC_MODELS[@]}"; do
    if wait "${pids[$i]}"; then
        echo "==================== TC ${TC_MODELS[$i]} OK  ($(date)) ===================="
    else
        echo "WARNING: TC ${TC_MODELS[$i]} FAILED — see tc_track_${TC_MODELS[$i]}.log"
        tc_failed=1
    fi
done

# ── STAGE 2: ETC tracker, all six models, under the same domain mask ────────
# NOR's 2007-2014 gap was already filled by the previous run; this pass re-runs all
# six because the domain mask changes ETC results too. The effect is small (ACC 2005:
# 35 -> 34 tracks, since 98.9% of ETC points were already in-domain) but it removes
# the spurious ~1300 km end-of-track jumps onto the out-of-domain cell (-61.50, 28.00)
# that affected ~8% of tracks. Pre-mask tracks are in backup_tracks_pre_intensity_rerun/.
ETC_MODELS=( ACC BCC CNR MPI MRI NOR )
epids=()
for MODEL in "${ETC_MODELS[@]}"; do
    echo "==================== launching ETC $MODEL  ($(date)) ===================="
    $mypython -u updated_monthly_ETC_Tracker.py --model "$MODEL" \
        --start-year 1980 --end-year 2014 \
        > "etc_track_${MODEL}.log" 2>&1 &
    epids+=($!)
done
for i in "${!ETC_MODELS[@]}"; do
    if wait "${epids[$i]}"; then
        echo "==================== ETC ${ETC_MODELS[$i]} OK  ($(date)) ===================="
    else
        echo "WARNING: ETC ${ETC_MODELS[$i]} FAILED — see etc_track_${ETC_MODELS[$i]}.log"
    fi
done

# Inventory before plotting: a year with 0 tracks or a 10x outlier is a tracker
# problem, and it is much cheaper to see it here than to read it off a figure.
echo ""
echo "==================== track inventory ($(date)) ===================="
for M in "${TC_MODELS[@]}"; do
    ne=$(ls ARs_Hurricanes_ETCs_MCS/Outputs_RegCM_PR_${M}/ETC_tracks/ETC_tracks_${M}_*.txt 2>/dev/null | wc -l)
    nt=$(ls -d ARs_Hurricanes_ETCs_MCS/Outputs_RegCM_PR_${M}/Hurricane_tracks/CONUS_year_* 2>/dev/null | wc -l)
    echo "  ${M}:  ETC years=${ne}  TC years=${nt}   (expect 35 / 35)"
done
echo ""

# ── STAGE 3/4: comparison figures ────────────────────────────────────────────
# Run even if a tracker failed — partial figures over the models that did succeed
# are more useful than none, and available_models() discovers what exists.
echo "==================== ETC figures ($(date)) ===================="
$mypython -u plot_ETC_model_comparison.py \
    --start-year 1980 --end-year 2014 --example-year 2005 \
    > "plot_ETC.log" 2>&1
echo "  rc=$? (see plot_ETC.log)"

echo "==================== TC figures ($(date)) ===================="
$mypython -u plot_TC_model_comparison.py \
    --start-year 1980 --end-year 2014 --example-year 2005 \
    > "plot_TC.log" 2>&1
echo "  rc=$? (see plot_TC.log)"

echo ""
echo "Figures written to Figures_ETC_comparison/ and Figures_TC_comparison/"
[[ $tc_failed -eq 1 ]] && echo "NOTE: at least one TC tracker run failed — check the logs above."
echo "Finished at $(date)"
