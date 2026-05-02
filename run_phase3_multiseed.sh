#!/bin/bash
# run_phase3_multiseed.sh
#
# Run 20 seeds × 2 architectures (sigmoid salience, metric_c1) with the two
# architectures running in PARALLEL.
#
# GPU assignment:
#   - If multiple GPUs detected (nvidia-smi reports ≥2): each arch gets its own.
#   - If single GPU: both archs share the GPU concurrently.
#       Memory is fine (29k + 67k params is tiny). Throughput per process
#       drops ~30% due to CUDA scheduling, so total wall time is dominated
#       by metric_c1 alone (~10h × 1.3 ≈ 13h instead of serial 15h).
#
# Resumable: skips runs whose ckpt_final.pt and probe_results.parquet exist.
#
# Usage:
#   bash run_phase3_multiseed.sh                  # default: 20 seeds, both archs in parallel
#   bash run_phase3_multiseed.sh both 0 5         # seeds 0-4, both archs in parallel
#   bash run_phase3_multiseed.sh sigmoid          # only sigmoid (no parallelism)
#   bash run_phase3_multiseed.sh metric_c1        # only metric_c1
#   FORCE_GPU=0 bash run_phase3_multiseed.sh      # pin both archs to GPU 0
#
# Output structure: per-arch dirs (same as serial version).
#
# Inspect progress while running:
#   tail -f outputs/multiseed/orchestrator.log
#   tail -f outputs/multiseed/salience.log
#   tail -f outputs/multiseed/metric_c1.log

set -uo pipefail   # NOT -e — we want a failing arch not to kill the other

ARCHS_ARG="${1:-both}"
SEED_START="${2:-0}"
SEED_END="${3:-20}"   # exclusive
EPISODES="${EPISODES:-300}"
OUTROOT="${OUTROOT:-outputs/multiseed}"

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
    echo "[$(date '+%F %T')] === multi-seed parallel run starting ==="
    echo "  archs:      ${ARCHS[*]}"
    echo "  seeds:      [$SEED_START, $SEED_END)  (n=$((SEED_END - SEED_START)))"
    echo "  episodes:   $EPISODES"
    echo "  device:     cuda"
    echo "  GPU mode:   $GPU_MODE"
    echo "  N_GPU seen: $N_GPU"
    echo "  output:     $OUTROOT"
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

    {
        echo "[$(date '+%F %T')] [$ARCH] worker starting on GPU=${GPU:-default}"
    } | tee -a "$ARCH_LOG" "$GLOBAL_LOG" >/dev/null

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

        echo "[$(date '+%F %T')] [$ARCH] [$RUN_IDX/$TOTAL_RUNS] seed=$SEED" \
            | tee -a "$ARCH_LOG" >/dev/null

        if [[ -f "$CKPT" && -f "$TRAJ" ]]; then
            echo "  [skip] training already done" | tee -a "$ARCH_LOG" >/dev/null
        else
            echo "  [run]  training..." | tee -a "$ARCH_LOG" >/dev/null
            local START_T
            START_T=$(date +%s)
            if ! python -m cear_pilot.training.train_phase3 \
                    --episodes "$EPISODES" \
                    --save_traj \
                    --seed "$SEED" \
                    --device cuda \
                    --gating_mode "$ARCH" \
                    --outdir "$RUN_DIR" \
                    > "$RUN_DIR/train.log" 2>&1
            then
                echo "  [FAIL] training crashed; see $RUN_DIR/train.log" \
                    | tee -a "$ARCH_LOG" "$GLOBAL_LOG" >/dev/null
                continue
            fi
            local ELAPSED=$(( $(date +%s) - START_T ))
            echo "  [ok]   training in ${ELAPSED}s" | tee -a "$ARCH_LOG" >/dev/null
        fi

        if [[ -f "$PROBE_RESULT" ]]; then
            echo "  [skip] probe already done" | tee -a "$ARCH_LOG" >/dev/null
        else
            echo "  [run]  probe analysis..." | tee -a "$ARCH_LOG" >/dev/null
            local START_T
            START_T=$(date +%s)
            if ! python -m cear_pilot.experiments.probe_phase3 \
                    --ckpt "$CKPT" \
                    --traj "$TRAJ" \
                    --outdir "$PROBE_DIR" \
                    --device cuda \
                    > "$RUN_DIR/probe.log" 2>&1
            then
                echo "  [FAIL] probe crashed; see $RUN_DIR/probe.log" \
                    | tee -a "$ARCH_LOG" "$GLOBAL_LOG" >/dev/null
                continue
            fi
            local ELAPSED=$(( $(date +%s) - START_T ))
            echo "  [ok]   probe in ${ELAPSED}s" | tee -a "$ARCH_LOG" >/dev/null
        fi
    done

    echo "[$(date '+%F %T')] [$ARCH] worker finished" \
        | tee -a "$ARCH_LOG" "$GLOBAL_LOG" >/dev/null
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
    echo "[$(date '+%F %T')] === multi-seed run completed ==="
    echo "Aggregate with:"
    echo "  python -m cear_pilot.experiments.aggregate_multiseed --root $OUTROOT"
} | tee -a "$GLOBAL_LOG"

exit "$EXIT_FAIL"
