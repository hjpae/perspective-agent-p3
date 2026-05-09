#!/usr/bin/env bash
set -euo pipefail

# Run Phase 3 frozen perturbation assays for 3 cohorts × 30 seeds × conditions.
# Default: two jobs in parallel on the visible GPU(s).
#
# Usage:
#   bash run_phase3_assay_3cohorts_parallel2.sh
#
# Dry run:
#   DRY_RUN=1 bash run_phase3_assay_3cohorts_parallel2.sh
#
# Custom range / parallelism:
#   SEED_START=0 SEED_END=4 MAX_JOBS=2 bash run_phase3_assay_3cohorts_parallel2.sh

BASE=${BASE:-outputs/p3_v3_cohorts}
ASSAY_BASE=${ASSAY_BASE:-outputs/p3_v3_assay}
SEED_START=${SEED_START:-0}
SEED_END=${SEED_END:-29}
MAX_JOBS=${MAX_JOBS:-2}
DEVICE=${DEVICE:-cuda}

STEPS=${STEPS:-160}
SHOCK_START=${SHOCK_START:-60}
SHOCK_DURATION=${SHOCK_DURATION:-20}
BODY_U_SHOCK_DELTA=${BODY_U_SHOCK_DELTA:--0.08}
ENV_PERTURB_SCALE=${ENV_PERTURB_SCALE:-0.20}

# Default conditions. Add env_shock if desired:
#   CONDITIONS="control body_shock env_shock" bash ...
CONDITIONS=${CONDITIONS:-"control body_shock"}
COHORTS=${COHORTS:-"full no_body_in_g no_conative"}

mkdir -p "${ASSAY_BASE}"
MANIFEST="${ASSAY_BASE}/assay_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "cohort,seed,condition,outdir,status" > "${MANIFEST}"
fi

wait_for_slot() {
  while [[ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]]; do
    sleep 2
  done
}

run_one() {
  local cohort="$1"
  local seed="$2"
  local condition="$3"

  local ckpt="${BASE}/${cohort}/s${seed}/ckpt_final.pt"
  local outdir="${ASSAY_BASE}/${cohort}/s${seed}/${condition}"
  local done_file="${outdir}/assay_summary.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[missing ckpt] ${ckpt}" >&2
    echo "${cohort},${seed},${condition},${outdir},missing_ckpt" >> "${MANIFEST}"
    return 0
  fi

  if [[ -f "${done_file}" ]]; then
    echo "[skip] ${cohort} s${seed} ${condition} already done"
    echo "${cohort},${seed},${condition},${outdir},skipped" >> "${MANIFEST}"
    return 0
  fi

  mkdir -p "${outdir}"
  local log="${outdir}/assay.log"

  local cmd=(
    python -m cear_pilot.scripts.run_phase3_perturb_assay
    --ckpt "${ckpt}"
    --condition "${condition}"
    --steps "${STEPS}"
    --shock_start "${SHOCK_START}"
    --shock_duration "${SHOCK_DURATION}"
    --body_u_shock_delta "${BODY_U_SHOCK_DELTA}"
    --env_perturb_scale "${ENV_PERTURB_SCALE}"
    --device "${DEVICE}"
    --outdir "${outdir}"
  )

  echo "[run] ${cohort} s${seed} ${condition}"
  echo "${cmd[*]}" > "${outdir}/command.txt"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: ${cmd[*]}"
    echo "${cohort},${seed},${condition},${outdir},dry_run" >> "${MANIFEST}"
  else
    PYTHONPATH=. "${cmd[@]}" > "${log}" 2>&1
    echo "${cohort},${seed},${condition},${outdir},done" >> "${MANIFEST}"
    echo "[done] ${cohort} s${seed} ${condition}"
  fi
}

for cohort in ${COHORTS}; do
  for seed in $(seq "${SEED_START}" "${SEED_END}"); do
    for condition in ${CONDITIONS}; do
      wait_for_slot
      run_one "${cohort}" "${seed}" "${condition}" &
    done
  done
done

wait

echo "[all done] assay outputs: ${ASSAY_BASE}"
echo "[manifest] ${MANIFEST}"
