#!/bin/bash
# run_phase3_pilot.sh
#
# Pilot multi-seed run with AAAI-style warm-up + actor objective.
# 5 seeds × 2 architectures (sigmoid, metric_c1), 250 episodes total
# (50 warmup episodes + 200 actor-active episodes).
#
# Goal: verify whether warm-up + actor objective stabilizes visit pattern
# across seeds. If it does → expand to full 20-seed run.
#
# Estimated wall time:
#   sigmoid:   5 seeds × ~7 min = ~35 min
#   metric_c1: 5 seeds × ~22 min = ~110 min
#   Parallel on shared GPU: dominated by metric (~110 min × 1.3 ≈ 2.5h)
#   Parallel on separate GPUs: max(35min, 110min) = ~110 min
#
# Usage:
#   bash run_phase3_pilot.sh                       # default: both, 5 seeds
#   bash run_phase3_pilot.sh sigmoid               # only sigmoid
#   bash run_phase3_pilot.sh metric_c1             # only metric_c1
#   bash run_phase3_pilot.sh both 0 5              # explicit
#   FORCE_GPU=0 bash run_phase3_pilot.sh           # pin GPU
#
# Output:
#   outputs/multiseed_pilot/{salience,metric_c1}/seed_NN/...
#
# Inspect progress:
#   tail -f outputs/multiseed_pilot/orchestrator.log
#   tail -f outputs/multiseed_pilot/salience.log
#   tail -f outputs/multiseed_pilot/metric_c1.log

set -uo pipefail

ARCHS_ARG="${1:-both}"
SEED_START="${2:-0}"
SEED_END="${3:-5}"
EPISODES="${EPISODES:-250}"
WARMUP_EPISODES="${WARMUP_EPISODES:-50}"
OUTROOT="${OUTROOT:-outputs/multiseed_pilot}"

case "$ARCHS_ARG" in
    sigmoid|salience) ARCHS=("salience") ;;
    metric_c1|metric) ARCHS=("metric_c1") ;;
    both) ARCHS=("salience" "metric_c1") ;;
    *) echo "ERROR: unknown arch '$ARCHS_ARG'. Use sigmoid|metric_c1|both"; exit 1 ;;
esac

mkdir -p "$OUTROOT"
GLOBAL_LOG="$OUTROOT/orchestrator.log"

# ------------------- GPU assignment --------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    N_GPU=0
fi

declare -A ARCH_GPU
if [[ -n "${FORCE_GPU:-}" ]]; then
    for ARCH in "${ARCHS[@]}"; do
        ARCH_GPU[$ARCH]="$FORCE_GPU"
    done
    GPU_MODE="forced (GPU $FORCE_GPU for all)"
elif [[ "$N_GPU" -ge 2 && "${#ARCHS[@]}" -eq 2 ]]; then
    ARCH_GPU[salience]=0
    ARCH_GPU[metric_c1]=1
    GPU_MODE="separate GPUs (sal=0, met=1)"
elif [[ "$N_GPU" -ge 1 ]]; then
    for ARCH in "${ARCHS[@]}"; do
        ARCH_GPU[$ARCH]=0
    done
    GPU_MODE="shared GPU 0"
else
    for ARCH in "${ARCHS[@]}"; do
        ARCH_GPU[$ARCH]=""
    done
    GPU_MODE="no GPU detected (using cuda env default)"
fi

{
    echo "[$(date '+%F %T')] === pilot multi-seed run starting ==="
    echo "  archs:           ${ARCHS[*]}"
    echo "  seeds:           [$SEED_START, $SEED_END)  (n=$((SEED_END - SEED_START)))"
    echo "  episodes:        $EPISODES (warmup $WARMUP_EPISODES)"
    echo "  GPU mode:        $GPU_MODE"
    echo "  N_GPU seen:      $N_GPU"
    echo "  output:          $OUTROOT"
} | tee -a "$GLOBAL_LOG"

# ------------------- per-arch worker --------------------
run_arch() {
    local ARCH="$1"
    local GPU="${ARCH_GPU[$ARCH]}"
    local ARCH_DIR="$OUTROOT/$ARCH"
    local ARCH_LOG="$OUTROOT/${ARCH}.log"
    mkdir -p "$ARCH_DIR"

    if [[ -n "$GPU" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU"
    fi

    _emit() {
        local msg="$1"
        echo "$msg" >> "$ARCH_LOG"
        echo "[$ARCH] $msg"
    }

    _emit "[$(date '+%F %T')] worker starting on GPU=${GPU:-default}"

    local TOTAL_RUNS=$(( SEED_END - SEED_START ))
    local RUN_IDX=0

    for (( SEED=SEED_START; SEED<SEED_END; SEED++ )); do
        RUN_IDX=$((RUN_IDX + 1))
        local SEED_PADDED
        SEED_PADDED=$(printf "%02d" "$SEED")
        local RUN_DIR="$ARCH_DIR/seed_${SEED_PADDED}"
        local PROBE_DIR="$RUN_DIR/probe"
        mkdir -p "$RUN_DIR" "$PROBE_DIR"

        local CKPT="$RUN_DIR/ckpt_final.pt"
        local TRAJ="$RUN_DIR/traj.parquet"
        local PROBE_RESULT="$PROBE_DIR/probe_results.parquet"

        _emit "[$(date '+%F %T')] [$RUN_IDX/$TOTAL_RUNS] seed=$SEED"

        if [[ -f "$CKPT" && -f "$TRAJ" ]]; then
            _emit "  [skip] training already done"
        else
            _emit "  [run]  training..."
            local START_T
            START_T=$(date +%s)
            if ! python -m cear_pilot.training.train_phase3 \
                    --episodes "$EPISODES" \
                    --warmup_episodes "$WARMUP_EPISODES" \
                    --save_traj \
                    --seed "$SEED" \
                    --device cuda \
                    --gating_mode "$ARCH" \
                    --outdir "$RUN_DIR" \
                    > "$RUN_DIR/train.log" 2>&1
            then
                _emit "  [FAIL] training crashed; see $RUN_DIR/train.log"
                continue
            fi
            local ELAPSED=$(( $(date +%s) - START_T ))
            _emit "  [ok]   training in ${ELAPSED}s"
        fi

        if [[ -f "$PROBE_RESULT" ]]; then
            _emit "  [skip] probe already done"
        else
            _emit "  [run]  probe analysis..."
            local START_T
            START_T=$(date +%s)
            if ! python -m cear_pilot.experiments.probe_phase3 \
                    --ckpt "$CKPT" \
                    --traj "$TRAJ" \
                    --outdir "$PROBE_DIR" \
                    --device cuda \
                    > "$RUN_DIR/probe.log" 2>&1
            then
                _emit "  [FAIL] probe crashed; see $RUN_DIR/probe.log"
                continue
            fi
            local ELAPSED=$(( $(date +%s) - START_T ))
            _emit "  [ok]   probe in ${ELAPSED}s"
        fi
    done

    _emit "[$(date '+%F %T')] worker finished"
}

# ------------------- Launch in parallel --------------------
declare -a PIDS
for ARCH in "${ARCHS[@]}"; do
    run_arch "$ARCH" &
    PIDS+=($!)
    {
        echo "[$(date '+%F %T')] launched arch=$ARCH pid=$!  GPU=${ARCH_GPU[$ARCH]:-default}"
    } | tee -a "$GLOBAL_LOG"
    sleep 2
done

EXIT_FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        EXIT_FAIL=1
        echo "[$(date '+%F %T')] worker pid=$PID exited with non-zero status" \
            | tee -a "$GLOBAL_LOG"
    fi
done

{
    echo ""
    echo "[$(date '+%F %T')] === pilot run completed ==="
    echo "Aggregate with:"
    echo "  python -m cear_pilot.experiments.aggregate_multiseed --root $OUTROOT"
} | tee -a "$GLOBAL_LOG"

exit "$EXIT_FAIL"
