#!/usr/bin/env bash
set -euo pipefail

# Run Phase 3 training cohorts with at most 2 concurrent jobs.
# Cohorts:
#   1) full:          body_in_g ON,  conative ON
#   2) no_conative:   body_in_g ON,  conative OFF
#   3) no_body_in_g:  body_in_g OFF, conative ON
#
# Usage examples:
#   bash run_phase3_3cohorts_parallel2.sh
#   MAX_JOBS=2 DEVICE=cuda SEED_START=0 SEED_END=29 bash run_phase3_3cohorts_parallel2.sh
#   DRY_RUN=1 bash run_phase3_3cohorts_parallel2.sh

MAX_JOBS="${MAX_JOBS:-2}"
DEVICE="${DEVICE:-cuda}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-29}"
BASE_OUTDIR="${BASE_OUTDIR:-outputs/p3_v3_cohorts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

# If you want to bind all runs to a specific GPU, launch with e.g.:
#   CUDA_VISIBLE_DEVICES=0 bash run_phase3_3cohorts_parallel2.sh
# If you want two runs on two different GPUs, this simple script does not
# assign GPUs round-robin; use DEVICE=cuda and set CUDA_VISIBLE_DEVICES upstream.

export PYTHONPATH="${PYTHONPATH:-.}"
mkdir -p "${BASE_OUTDIR}"

MANIFEST="${BASE_OUTDIR}/run_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "condition,seed,outdir,status,start_time,end_time" > "${MANIFEST}"
fi

COMMON_ARGS=(
  --episodes 180
  --warmup_episodes 30
  --max_steps 200
  --save_traj
  --device "${DEVICE}"
  --gating_mode metric_c1
  --silhouette_dim 4
  --silhouette_sigma 0.10
  --g_carry_decay 0.99
  --lambda_body 0.10
  --entropy_target_ratio 0.70
  --w_entropy_min 0.005
  --metabolic_cost 0.002
  --movement_cost 0.001
  --affordance_gain 0.05
  --body_trend_weight 0.5
  --body_tendency_horizon 5
  --conative_temperature 0.10
  --conative_trend_weight 1.0
  --conative_body_weight 0.25
  --print_every 10
)

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [[ "$(running_jobs)" -ge "${MAX_JOBS}" ]]; do
    wait -n || true
  done
}

run_one() {
  local condition="$1"
  local seed="$2"
  local outdir="${BASE_OUTDIR}/${condition}/s${seed}"
  local logfile="${outdir}/train.log"

  mkdir -p "${outdir}"

  if [[ -f "${outdir}/ckpt_final.pt" && -f "${outdir}/traj.parquet" ]]; then
    echo "[skip] ${condition} seed=${seed}: existing ckpt_final.pt and traj.parquet"
    return 0
  fi

  local extra_args=()
  case "${condition}" in
    full)
      extra_args=(--w_conative 0.1)
      ;;
    no_conative)
      extra_args=(--w_conative 0.0)
      ;;
    no_body_in_g)
      extra_args=(--w_conative 0.1 --no_body_in_g)
      ;;
    *)
      echo "Unknown condition: ${condition}" >&2
      return 1
      ;;
  esac

  local start_time
  start_time="$(date -Iseconds)"
  echo "[start] ${condition} seed=${seed} outdir=${outdir} ${start_time}"

  local cmd=(
    "${PYTHON_BIN}" -m cear_pilot.training.train_phase3_v3_conative
    "${COMMON_ARGS[@]}"
    --seed "${seed}"
    --outdir "${outdir}"
    "${extra_args[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  {
    echo "# condition=${condition} seed=${seed}"
    echo "# start_time=${start_time}"
    printf '# command: '
    printf '%q ' "${cmd[@]}"
    printf '\n\n'
    "${cmd[@]}"
  } 2>&1 | tee "${logfile}"

  local end_time
  end_time="$(date -Iseconds)"
  echo "${condition},${seed},${outdir},done,${start_time},${end_time}" >> "${MANIFEST}"
  echo "[done] ${condition} seed=${seed} ${end_time}"
}

trap 'echo "[interrupt] stopping launched jobs..."; jobs -pr | xargs -r kill; exit 130' INT TERM

conditions=(full no_conative no_body_in_g)

for condition in "${conditions[@]}"; do
  for seed in $(seq "${SEED_START}" "${SEED_END}"); do
    wait_for_slot
    run_one "${condition}" "${seed}" &
  done
done

wait

echo "[all done] outputs under ${BASE_OUTDIR}"
echo "[manifest] ${MANIFEST}"
