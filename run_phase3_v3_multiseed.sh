#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Phase 3 v3 multi-seed pilot.
#
# Configuration: body-coupled architecture, body PE → g GRU (III),
# silhouette OFF, metric_c1 gating. Matches the s0 single-seed
# verified setup.
#
# Concurrency: 2 seeds simultaneously (GPU memory budget).
# Each seed's stdout is written to its own log file. Failures in
# one seed do not block the others.
# ─────────────────────────────────────────────────────────────────
set -uo pipefail

N_SEEDS="${N_SEEDS:-20}"
N_PARALLEL="${N_PARALLEL:-2}"
EPISODES="${EPISODES:-150}"
WARMUP="${WARMUP:-30}"
MAX_STEPS="${MAX_STEPS:-200}"
DEVICE="${DEVICE:-cuda}"
GATING_MODE="${GATING_MODE:-metric_c1}"
OUTBASE="${OUTBASE:-outputs/p3_v3_multiseed_${GATING_MODE}}"

mkdir -p "$OUTBASE/logs"

echo "[multiseed] starting"
echo "[multiseed] seeds: 0..$((N_SEEDS - 1))  parallel: $N_PARALLEL"
echo "[multiseed] episodes: $EPISODES (warmup $WARMUP)  max_steps: $MAX_STEPS"
echo "[multiseed] gating_mode: $GATING_MODE"
echo "[multiseed] outbase: $OUTBASE"
echo

run_one_seed() {
    local seed=$1
    local out_dir="$OUTBASE/seed_${seed}"
    local log_file="$OUTBASE/logs/seed_${seed}.log"
    mkdir -p "$out_dir"
    echo "[seed $seed] starting → $log_file"
    python -m cear_pilot.training.train_phase3_v3 \
        --episodes "$EPISODES" \
        --warmup_episodes "$WARMUP" \
        --max_steps "$MAX_STEPS" \
        --save_traj \
        --seed "$seed" \
        --device "$DEVICE" \
        --gating_mode "$GATING_MODE" \
        --print_every 25 \
        --outdir "$out_dir" \
        > "$log_file" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[seed $seed] DONE"
    else
        echo "[seed $seed] FAILED (rc=$rc) — see $log_file"
    fi
    return $rc
}

T0=$(date +%s)
n_done=0
n_failed=0
failed_list=()

# Pool semantics: keep at most N_PARALLEL background jobs running.
seed=0
while (( seed < N_SEEDS )); do
    # While the pool is full, wait for any one job to finish.
    while (( $(jobs -rp | wc -l) >= N_PARALLEL )); do
        # `wait -n` returns when any background job exits.
        if wait -n; then
            n_done=$((n_done + 1))
        else
            n_done=$((n_done + 1))
            n_failed=$((n_failed + 1))
        fi
    done
    run_one_seed "$seed" &
    seed=$((seed + 1))
done

# Drain remaining jobs.
while (( $(jobs -rp | wc -l) > 0 )); do
    if wait -n; then
        n_done=$((n_done + 1))
    else
        n_done=$((n_done + 1))
        n_failed=$((n_failed + 1))
    fi
done

T1=$(date +%s)
ELAPSED=$((T1 - T0))
echo
echo "[multiseed] complete in ${ELAPSED}s"
echo "[multiseed] $n_done finished, $n_failed failed"

# Brief final report: which seeds produced ckpts
echo
echo "[multiseed] checkpoint summary:"
for s in $(seq 0 $((N_SEEDS - 1))); do
    if [[ -f "$OUTBASE/seed_${s}/ckpt_final.pt" ]]; then
        echo "  seed $s: OK"
    else
        echo "  seed $s: MISSING ckpt"
    fi
done
