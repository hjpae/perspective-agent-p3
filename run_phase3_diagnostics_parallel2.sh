#!/usr/bin/env bash
set -euo pipefail

# Run diagnose_body_field_conative.py over all cohort/seed directories, with up to MAX_JOBS in parallel.
# Assumes you run from repo root and PYTHONPATH=. works.
#
# Example:
#   BASE=outputs/p3_v3_cohorts SEED_START=0 SEED_END=29 MAX_JOBS=2 \
#     bash cear_pilot/scripts/run_phase3_diagnostics_parallel2.sh
#
# Dry run:
#   DRY_RUN=1 bash cear_pilot/scripts/run_phase3_diagnostics_parallel2.sh

BASE=${BASE:-outputs/p3_v3_cohorts}
COHORTS=${COHORTS:-full,no_conative,no_body_in_g}
SEED_START=${SEED_START:-0}
SEED_END=${SEED_END:-29}
MAX_JOBS=${MAX_JOBS:-2}
DEVICE=${DEVICE:-cuda}
LATE_EPISODES=${LATE_EPISODES:-80}
N_PER_ZONE=${N_PER_ZONE:-200}
DIAG_SCRIPT=${DIAG_SCRIPT:-cear_pilot/scripts/diagnose_body_field_conative.py}
DRY_RUN=${DRY_RUN:-0}

IFS=',' read -r -a COHORT_ARR <<< "$COHORTS"

wait_for_slot() {
  while true; do
    local n
    n=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "$n" -lt "$MAX_JOBS" ]]; then
      break
    fi
    sleep 2
  done
}

run_one() {
  local cohort=$1
  local seed=$2
  local outdir="$BASE/$cohort/s$seed"
  local log="$outdir/diagnostics/diagnose.log"
  local summary="$outdir/diagnostics/bodydecoder_counterfactual_summary.csv"

  if [[ ! -f "$outdir/traj.parquet" || ! -f "$outdir/ckpt_final.pt" ]]; then
    echo "[skip missing run] $outdir"
    return 0
  fi

  if [[ -f "$summary" ]]; then
    echo "[skip done] $cohort s$seed"
    return 0
  fi

  mkdir -p "$outdir/diagnostics"
  echo "[diag start] $cohort s$seed"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "PYTHONPATH=. python $DIAG_SCRIPT --outdir $outdir --device $DEVICE --late_episodes $LATE_EPISODES --n_per_zone $N_PER_ZONE"
  else
    PYTHONPATH=. python "$DIAG_SCRIPT" \
      --outdir "$outdir" \
      --device "$DEVICE" \
      --late_episodes "$LATE_EPISODES" \
      --n_per_zone "$N_PER_ZONE" \
      > "$log" 2>&1
    echo "[diag done] $cohort s$seed"
  fi
}

for cohort in "${COHORT_ARR[@]}"; do
  cohort=$(echo "$cohort" | xargs)
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    wait_for_slot
    run_one "$cohort" "$seed" &
  done
done

wait
echo "[all diagnostics complete]"
