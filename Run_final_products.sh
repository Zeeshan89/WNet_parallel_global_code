#!/bin/bash
# Run_final_products.sh — after the final retrain (H_final_e300) has all its epochs: export the
# AR and ETC products, track the ETCs with the S15 configuration, and run every evaluation, as
# one SLURM dependency chain. Not a batch script itself: it submits the jobs.
#
#   bash Run_final_products.sh                      # tag H_final_e300, product dir Global_final
#   TAG=H_final_e300 OUT=Global_final bash Run_final_products.sh
#
# Chain:
#   1. export AR   (best_ar.pth,  16 GPU nodes, ~15 min)   -> $OUT/ARs_WNet_final
#   2. export ETC  (best.pth,     16 GPU nodes, ~15 min)   -> $OUT/ETCs_WNet_final
#   3. AR evaluation (Run_ar_eval_arlit.sh: vs MARS, ARTMIP, learnability)   after 1
#   4. ETC tracks, S15 tracker, 1980-2024 (Run_etc_tracks_variant.sh)       after 2
#   5. ETC paper analysis with suffix _final (Run_etc_paper_variant.sh)     after 4
#   6. global KDE of the new tracks (plot_etc_kde_global.py)                after 4
set -u
NB=/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks
CODE=$NB/WNet_parallel_global_code
TAG="${TAG:-H_final_e300}"
OUT="${OUT:-Global_final}"
CK="$CODE/ARs_Hurricanes_ETCs_MCS/Phase2/$TAG"
cd $CODE

n=$(python3 -c "import json; print(len(json.load(open('$CK/history.json')).get('epochs', [])))" 2>/dev/null || echo 0)
[ "$n" -ge 300 ] || { echo "ERROR: $TAG has $n/300 epochs; the chain is not finished"; exit 1; }
for f in best_ar.pth best.pth; do [ -s "$CK/$f" ] || { echo "ERROR: $CK/$f missing"; exit 1; }; done
python3 - <<PY
import json
h = json.load(open("$CK/history.json")); e = h["epochs"]
for c in ("ar", "hurricane", "etc", "mcs"):
    b = max(e, key=lambda r: r.get(f"val_{c}_iou", -1)); print(f"  {c:<10} best val IoU {b[f'val_{c}_iou']:.4f} at epoch {b['epoch']}")
PY

sub() { local out; for i in 1 2 3 4 5; do out=$(timeout 90 sbatch --parsable "$@" 2>&1) && case "$out" in *[!0-9]*) ;; *) echo "$out"; return 0;; esac; sleep 30; done; echo "SUBMIT-FAILED: $out" >&2; return 1; }

J1=$(sub --export=ALL,CK=$CK/best_ar.pth,OUT=$NB/$OUT Run_export_ar_label.sh) || exit 1
echo "1. export AR            $J1"
J2=$(sub --export=ALL,CK_ETC=$CK/best.pth,OUT=$NB/$OUT,ONLY_ETC=1 Run_export_etc_final.sh) || exit 1
echo "2. export ETC           $J2"
J3=$(sub --dependency=afterok:$J1 --export=ALL,PRODUCT=$OUT,LABEL=final,NATIVE=L6 Run_ar_eval_arlit.sh) || exit 1
echo "3. AR evaluation        $J3  (after $J1) -> evaluations/ar_{mars,artmip}_eval_final"
J4=$(sub --dependency=afterok:$J2 --export=ALL,TAG=final_S15,DET=$NB/$OUT/ETCs_WNet_final,ETC_FILTER_PROFILE=common,ETC_CENTER_MODE=mslp,ETC_CC_DEPTH_PA=150,ETC_MAX_GAP_STEPS=2,ETC_GAP_RADIUS_GROWTH=0 Run_etc_tracks_variant.sh) || exit 1
echo "4. ETC tracks (S15)     $J4  (after $J2) -> Phase2/F_ibtracs_pwH50_e200/tracks_final_S15/etc"
TRK=$CODE/ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200/tracks_final_S15/etc
J5=$(cd $NB/etc_naec_common && sub --dependency=afterok:$J4 --export=ALL,TAG=final_S15,ETC_DIR=$TRK Run_etc_paper_variant.sh) || exit 1
echo "5. ETC paper analysis   $J5  (after $J4) -> etc_naec_common/figures_final_S15"
echo "$J1 $J2 $J3 $J4 $J5" > logs/final_products_jobids.txt
echo "after $J4: cd etc_naec_common && ETC_VERSION=v2 python plot_etc_kde_global.py (edit CATALOGUES to $TRK) for the global KDE"
