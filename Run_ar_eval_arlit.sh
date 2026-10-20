#!/bin/bash -l
# Run_ar_eval_arlit.sh — score the AR product of the model retrained on the literature-based
# label with the two existing harnesses, unchanged, into NEW directories:
#
#   evaluations/ar_mars_eval_arlit/    WNet vs MARS (phase4_ar_vs_mars_v2.py): latitude bands,
#                                      land / ocean per coastal region, seasons, object counts
#   evaluations/ar_artmip_eval_arlit/  WNet among the five established methods
#                                      (phase4_ar_artmip_v2.py + plot_ar_paper_figs.py)
#   evaluations/ar_learnability_arlit/ model vs its OWN label by latitude band and land / ocean
#                                      (phase4_ar_learnability.py, the validation years)
#
# Before-numbers to beat (shipped product, evaluations/ar_mars_eval_v2 and ar_artmip_eval_v2):
# tropics |lat| < 15 BIAS 9.6, polar POD 0.08, land IoU 0.10-0.30 vs ocean 0.54-0.60,
# 60S-60N pattern r 0.40, 9.0 % of the area above 120 AR days per year.
# Two nodes, 80 ranks, about 15 min; every stage skips finished parts.
#
#   sbatch --dependency=afterok:<export job> Run_ar_eval_arlit.sh
#   sbatch --export=PRODUCT=Global_arlit,LABEL=arlit Run_ar_eval_arlit.sh

#SBATCH --account=CLI138
#SBATCH -J ar_eval_lit
#SBATCH -p batch
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH -o logs/areval-lit-%j.txt
#SBATCH -e logs/areval-lit-err-%j.txt
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
PRODUCT="${PRODUCT:-Global_arlit}"
LABEL="${LABEL:-arlit}"
PRED="$NB/$PRODUCT/ARs_WNet_final"
MARS_OUT="$CODE/evaluations/ar_mars_eval_$LABEL"
ART_OUT="$CODE/evaluations/ar_artmip_eval_$LABEL"
PY="$(which python3) -W ignore -u"
n=$(ls $PRED/era5dara_*_AR_detection.nc 2>/dev/null | wc -l)
[ "$n" -eq 540 ] || { echo "ERROR: $PRED has $n/540 months"; exit 1; }
echo "product $PRED; outputs $MARS_OUT, $ART_OUT; $(date)"

echo "=== WNet vs MARS, 1980-2024 (32 ranks) ==="
srun -n 32 -c 1 --export=ALL $PY $CODE/phase4_ar_vs_mars_v2.py --pred-dir "$PRED" --outdir "$MARS_OUT" || exit 1

echo "=== ARTMIP intercomparison, 1980-2019 (80 ranks, one per half-year) ==="
srun -n 80 -c 1 --export=ALL $PY $CODE/phase4_ar_artmip_v2.py --pred-dir "$PRED" --outdir "$ART_OUT" || exit 1

echo "=== merge and figures ==="
cd "$MARS_OUT"
V2=$CODE/evaluations/ar_mars_eval_v2
$PY $V2/verify_ar_v2.py --cache-dir "$MARS_OUT"
$PY $V2/merge_ar_v2.py --cache-dir "$MARS_OUT" --label "$LABEL" --pred-name "$PRODUCT/ARs_WNet_final" || exit 1
ln -sf "ar_${LABEL}_merged.npz" "$MARS_OUT/ar_v2_merged.npz"     # the plot scripts read the v2 file name
$PY $V2/plot_ar_regional.py --v2-dir "$MARS_OUT" --out "$MARS_OUT"
$PY $V2/plot_ar_regional.py --v2-dir "$MARS_OUT" --set land --out "$MARS_OUT/regional_coast_to_land"
$PY $V2/plot_ar_v2_seasonal.py --v2-dir "$MARS_OUT" --out "$MARS_OUT/ar_seasonal"
cp -n $CODE/evaluations/ar_artmip_eval_v2/plot_ar_paper_figs.py "$ART_OUT/"      # directory-relative script
cd "$ART_OUT" && $PY plot_ar_paper_figs.py
echo "=== model vs its own label (validation years) ==="
NATIVE="${NATIVE:-L6}"                      # the label the model was trained on, in AR_label_candidates/
cd $CODE && srun -n 48 -c 1 --export=ALL $PY ./phase4_ar_learnability.py --pred-dir "$PRED" \
    --label "$NATIVE" --outdir "evaluations/ar_learnability_$LABEL"
$PY ./phase4_ar_learnability.py --summarize --outdir "evaluations/ar_learnability_$LABEL"
# the same for the shipped product against the shipped label: the resolution ceiling both share
srun -n 48 -c 1 --export=ALL $PY ./phase4_ar_learnability.py \
    --pred-dir "$NB/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final" --label L0 \
    --outdir "evaluations/ar_learnability_shipped_L0"
$PY ./phase4_ar_learnability.py --summarize --outdir "evaluations/ar_learnability_shipped_L0"
echo "exit $?  $(date)"
ls "$MARS_OUT"/*.csv "$ART_OUT"/*.csv 2>/dev/null
