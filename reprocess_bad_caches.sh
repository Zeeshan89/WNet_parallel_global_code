#!/usr/bin/env bash
# reprocess_bad_caches.sh — Delete bad .h5 caches and re-run preprocess_cache.py
# on ONLY those months. Reads the list from cache_audit_results.txt.bad_months.txt
# (the file audit_cache.py produces).
#
# Usage:
#   ./reprocess_bad_caches.sh                          # uses default bad-list path
#   ./reprocess_bad_caches.sh path/to/bad_months.txt   # explicit list
#   DRY_RUN=1 ./reprocess_bad_caches.sh                # show actions, do nothing
#   SKIP_VERIFIED=1 ./reprocess_bad_caches.sh          # probe each bad-list file first;
#                                                       # only delete/redo if STILL broken
#                                                       # (safe to re-run after partial recovery)
#
# After it finishes:
#   python audit_cache.py --months-from-file cache_audit_results.txt.bad_months.txt
#   → verifies only the reprocessed months without touching the other 529.

set -euo pipefail

# ── Config (matches the layout in main.py / preprocess_cache.py) ──────────────
BAD_LIST="${1:-cache_audit_results.txt.bad_months.txt}"
CACHE_DIR="/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/cache/preprocessed_v3"
PREPROCESS_DIR="/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection"
PREPROCESS_SCRIPT="${PREPROCESS_DIR}/preprocess_cache.py"
LOG_DIR="${PREPROCESS_DIR}/reprocess_logs"
DRY_RUN="${DRY_RUN:-0}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${BAD_LIST}" ]]; then
  echo "ERROR: bad-list file not found: ${BAD_LIST}"
  echo "Run audit_cache.py first to generate it."
  exit 1
fi
if [[ ! -f "${PREPROCESS_SCRIPT}" ]]; then
  echo "ERROR: preprocess script not found: ${PREPROCESS_SCRIPT}"
  exit 1
fi
mkdir -p "${LOG_DIR}"

# ── Read stems (strip any trailing tab-separated error type) ──────────────────
mapfile -t STEMS < <(awk 'NF && !/^#/ { print $1 }' "${BAD_LIST}")
if [[ ${#STEMS[@]} -eq 0 ]]; then
  echo "Bad-list is empty — nothing to do."
  exit 0
fi

# ── Optional pre-check: drop already-OK months from the redo list ─────────────
if [[ "${SKIP_VERIFIED:-0}" == "1" ]]; then
  echo "── Pre-check: probing each bad-list file; will skip ones that now read OK ──"
  RECHECK_PREFIX="/tmp/reprocess_recheck_$$"
  RECHECK_OUT="${RECHECK_PREFIX}.txt"
  if python audit_cache.py \
        --months-from-file "${BAD_LIST}" \
        --out "${RECHECK_OUT}" >/dev/null 2>&1; then
    if [[ -f "${RECHECK_OUT}.bad_months.txt" ]]; then
      mapfile -t STEMS < <(awk 'NF { print $1 }' "${RECHECK_OUT}.bad_months.txt")
    else
      STEMS=()   # nothing still bad → nothing to do
    fi
  else
    echo "  WARNING: pre-check audit failed — falling back to processing the full bad-list"
  fi
  rm -f "${RECHECK_OUT}" "${RECHECK_OUT}.bad_months.txt"
  echo "  After pre-check: ${#STEMS[@]} months still need redo."
  if [[ ${#STEMS[@]} -eq 0 ]]; then
    echo "All months in bad-list are now readable — nothing to do."
    exit 0
  fi
fi

echo "================================================================================"
echo " REPROCESS BAD CACHES"
echo "================================================================================"
echo "  Bad-list           : ${BAD_LIST}"
echo "  Cache dir          : ${CACHE_DIR}"
echo "  Preprocess script  : ${PREPROCESS_SCRIPT}"
echo "  Log dir            : ${LOG_DIR}"
echo "  Months to redo (${#STEMS[@]}) : ${STEMS[*]}"
[[ "${DRY_RUN}" == "1" ]] && echo "  DRY RUN — no files will be deleted or written."
echo

# ── Step 1: delete the bad .h5 files (preprocess_cache.py won't overwrite if --skip-existing) ──
echo "── Step 1: delete bad cache files ──"
for stem in "${STEMS[@]}"; do
  cache_file="${CACHE_DIR}/era5_preprocessed_${stem}.h5"
  if [[ -f "${cache_file}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY: rm ${cache_file}"
    else
      rm -v "${cache_file}"
    fi
  else
    echo "  (already absent) ${cache_file}"
  fi
done

# ── Step 2: group by year so preprocess_cache.py runs once per year ───────────
echo
echo "── Step 2: group months by year ──"
declare -A YEAR_MONTHS
for stem in "${STEMS[@]}"; do
  year="${stem:0:4}"
  month="${stem:4:2}"
  month_no_pad="$((10#${month}))"   # strip leading zero, base-10
  YEAR_MONTHS[$year]+="${month_no_pad} "
done
for year in $(echo "${!YEAR_MONTHS[@]}" | tr ' ' '\n' | sort); do
  echo "  ${year}: ${YEAR_MONTHS[$year]}"
done

# ── Step 3: invoke preprocess_cache.py per (year, month) ──────────────────────
echo
echo "── Step 3: reprocess each (year, month) ──"
total=0; ok=0; failed=0; failed_stems=()

for year in $(echo "${!YEAR_MONTHS[@]}" | tr ' ' '\n' | sort); do
  for month in ${YEAR_MONTHS[$year]}; do
    total=$((total + 1))
    stem=$(printf "%04d%02d" "${year}" "${month}")
    log_file="${LOG_DIR}/preprocess_${stem}.log"

    echo
    echo "[${total}/${#STEMS[@]}] year=${year} month=${month}  → log: ${log_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY: (cd ${PREPROCESS_DIR} && python preprocess_cache.py --year ${year} --month ${month})"
      continue
    fi

    if (cd "${PREPROCESS_DIR}" && \
        python preprocess_cache.py --year "${year}" --month "${month}") \
        >"${log_file}" 2>&1; then
      # Verify the new .h5 actually appeared
      new_cache="${CACHE_DIR}/era5_preprocessed_${stem}.h5"
      if [[ -f "${new_cache}" ]]; then
        size=$(stat -c %s "${new_cache}")
        size_gb=$(awk "BEGIN {printf \"%.2f\", ${size}/1024/1024/1024}")
        echo "  OK   wrote ${new_cache} (${size_gb} GB)"
        ok=$((ok + 1))
      else
        echo "  FAIL preprocess returned 0 but no cache file at ${new_cache}"
        echo "       check ${log_file}"
        failed=$((failed + 1))
        failed_stems+=("${stem}")
      fi
    else
      echo "  FAIL preprocess exited non-zero — check ${log_file}"
      failed=$((failed + 1))
      failed_stems+=("${stem}")
    fi
  done
done

echo
echo "================================================================================"
echo " SUMMARY"
echo "================================================================================"
echo "  Reprocessed OK : ${ok} / ${total}"
echo "  Failed         : ${failed} / ${total}"
if [[ ${failed} -gt 0 ]]; then
  echo "  Failed stems   : ${failed_stems[*]}"
  echo "  Inspect logs in ${LOG_DIR}/"
fi
echo
echo "Next step — verify ONLY these months are now readable:"
echo "  python audit_cache.py --months-from-file ${BAD_LIST}"
echo "(should now report all OK; any still-BAD ones need a fresh look at their preprocess log.)"
