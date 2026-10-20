#!/bin/bash -l
# Run_ar_eval_fullres.sh — the two AR evaluations of the shipped product (evaluations/ar_mars_eval_v2,
# evaluations/ar_artmip_eval_v2) for the WNet trained and run at 0.25 deg
# (Global_ARs_Hurr_ETCs_MCS_fullres025/ARs_WNet_final), same harnesses and plot scripts, new directories:
#   evaluations/ar_mars_eval_fullres/     WNet vs MARS 1980-2024: spatial, 2col, diagnostics, time series,
#                                         robustness, seasonal, regional (ocean + land), regional gap, objects
#   evaluations/ar_artmip_eval_fullres/   WNet among MARS, Guan-Waliser, Mundhenk, TempestLR, Reid500 (1980-2019)
#   evaluations/ar_learnability_fullres/  model vs its own label (L6), validation years
# Modelled on Run_ar_eval_arlit.sh (2 nodes, 12.5 min there). Every harness stage skips finished parts.
#SBATCH --account=CLI138
#SBATCH -J ar_eval_fr
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH -o logs/areval-fr-%j.txt
#SBATCH -e logs/areval-fr-err-%j.txt
#SBATCH --export=NONE
mkdir -p logs; unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.7.0; module load cpe/26.03
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CODE=$NB/WNet_parallel_global_code
PRODUCT=Global_ARs_Hurr_ETCs_MCS_fullres025
LABEL=fullres
PRED="$NB/$PRODUCT/ARs_WNet_final"
MARS_OUT="$CODE/evaluations/ar_mars_eval_$LABEL"
ART_OUT="$CODE/evaluations/ar_artmip_eval_$LABEL"
V2=$CODE/evaluations/ar_mars_eval_v2
PY="$(which python3) -W ignore -u"
n=$(ls $PRED/era5dara_*_AR_detection.nc 2>/dev/null | wc -l)
[ "$n" -eq 540 ] || { echo "ERROR: $PRED has $n/540 months"; exit 1; }
echo "product $PRED; outputs $MARS_OUT, $ART_OUT; $(date)"

echo "=== WNet vs MARS, 1980-2024 (32 ranks) ==="
srun -n 32 -c 1 --export=ALL $PY $CODE/phase4_ar_vs_mars_v2.py --pred-dir "$PRED" --outdir "$MARS_OUT" || exit 1
echo "=== ARTMIP intercomparison, 1980-2019 (80 ranks, one per half-year) ==="
srun -n 80 -c 1 --export=ALL $PY $CODE/phase4_ar_artmip_v2.py --pred-dir "$PRED" --outdir "$ART_OUT" || exit 1

echo "=== merge ==="
cd "$MARS_OUT"
$PY $V2/verify_ar_v2.py --cache-dir "$MARS_OUT"
$PY $V2/merge_ar_v2.py --cache-dir "$MARS_OUT" --label "$LABEL" --pred-name "$PRODUCT/ARs_WNet_final" || exit 1
ln -sf "ar_${LABEL}_merged.npz" "$MARS_OUT/ar_v2_merged.npz"     # the plot scripts read the v2 file name
$PY $V2/merge_ar_objects_v2.py --cache-dir "$MARS_OUT" --label "$LABEL" || echo "WARN objects merge failed"

echo "=== figures (ar_mars_eval_v2 set) ==="
# no v1 cache here, so the 3-row / diagnostics figures draw MARS and this product
$PY $V2/plot_ar_v2_main.py        --v2-dir "$MARS_OUT" --v1-dir /nonexistent --out "$MARS_OUT/ar_spatial_3row" || echo "WARN main"
$PY $V2/plot_ar_v2_2col.py        --v2-dir "$MARS_OUT" --out "$MARS_OUT/ar_spatial_2col"                         || echo "WARN 2col"
$PY $V2/plot_ar_v2_2col.py        --v2-dir "$MARS_OUT" --levels published --out "$MARS_OUT/ar_spatial_2col_pubscale" || echo "WARN 2col pub"
$PY $V2/plot_ar_v2_diagnostics.py --v2-dir "$MARS_OUT" --v1-dir /nonexistent --out "$MARS_OUT/ar_v2_diagnostics" || echo "WARN diagnostics"
$PY $V2/plot_ar_v2_timeseries.py  --v2-dir "$MARS_OUT" --out "$MARS_OUT/ar_v2_timeseries"                        || echo "WARN timeseries"
$PY $V2/plot_ar_v2_robustness.py  --v2-dir "$MARS_OUT" --out "$MARS_OUT/ar_v2_robustness"                        || echo "WARN robustness"
$PY $V2/plot_ar_v2_seasonal.py    --v2-dir "$MARS_OUT" --out "$MARS_OUT/ar_v2_seasonal"                          || echo "WARN seasonal"
$PY $V2/plot_ar_regional.py       --v2-dir "$MARS_OUT" --out "$MARS_OUT"                                         || echo "WARN regional"
$PY $V2/plot_ar_regional.py       --v2-dir "$MARS_OUT" --set land --out "$MARS_OUT/regional_coast_to_land"       || echo "WARN regional land"
cp -n $V2/plot_ar_regional_gap.py $V2/ar_v2_common.py "$MARS_OUT/" 2>/dev/null   # directory-relative scripts
$PY "$MARS_OUT/plot_ar_regional_gap.py" || echo "WARN regional gap"

echo "=== figures (ar_artmip_eval_v2 set) ==="
cp -n $CODE/evaluations/ar_artmip_eval_v2/plot_ar_paper_figs.py "$ART_OUT/"
cd "$ART_OUT" && $PY plot_ar_paper_figs.py || echo "WARN artmip figs"

echo "=== model vs its own label (L6), validation years ==="
cd $CODE && srun -n 48 -c 1 --export=ALL $PY ./phase4_ar_learnability.py --pred-dir "$PRED" \
    --label L6 --outdir "evaluations/ar_learnability_$LABEL"
$PY ./phase4_ar_learnability.py --summarize --outdir "evaluations/ar_learnability_$LABEL"
echo "done $(date)"
ls "$MARS_OUT"/*.png "$ART_OUT"/*.png 2>/dev/null
