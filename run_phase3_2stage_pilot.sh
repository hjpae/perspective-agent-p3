#!/bin/bash
# run_phase3_2stage_pilot.sh
#
# Two-stage training pilot. 5 seeds × 2 archs × 2 protocols = 20 runs.
#
# Stage 1 (80 ep): ablate_g=True, fixed alpha, all params trainable,
#                  actor active after warmup. perturbation OFF.
# Stage 2 (170 ep): backbone frozen, gating + AlphaNet trainable, actor OFF.
#                   perturbation ON (P4) or OFF (P3).
#
# Estimated wall time:
#   sigmoid:   5 seeds × ~7 min = ~35 min  per protocol
#   metric_c1: 5 seeds × ~22 min = ~110 min per protocol
#   Parallel on shared GPU: ~110 min × 2 protocols ≈ 3.7h
#   Parallel on separate GPUs: max(35min, 110min) × 2 = ~3.7h
#   Sequential protocols: ~3.7h × 2 = ~7.4h
#
# Output:
#   outputs/multiseed_2stage/{P3,P4}/{salience,metric_c1}/seed_NN/
#
# Usage:
#   bash run_phase3_2stage_pilot.sh                      # both protocols, both archs
#   bash run_phase3_2stage_pilot.sh P3                   # only P3
#   bash run_phase3_2stage_pilot.sh P4 sigmoid           # P4 + sigmoid only
#   bash run_phase3_2stage_pilot.sh both both 0 5        # explicit
#   FORCE_GPU=0 bash run_phase3_2stage_pilot.sh          # pin GPU

set -uo pipefail

PROTOCOL_ARG="${1:-both}"
ARCHS_ARG="${2:-both}"
SEED_START="${3:-0}"
SEED_END="${4:-5}"
STAGE1_EPS="${STAGE1_EPS:-80}"
STAGE2_EPS="${STAGE2_EPS:-170}"
WARMUP_EPISODES="${WARMUP_EPISODES:-20}"
SIGMA_LEFT="${SIGMA_LEFT:-0.40}"
SIGMA_RIGHT="${SIGMA_RIGHT:-0.05}"
N_PERTURBATIONS="${N_PERTURBATIONS:-4}"
OUTROOT="${OUTROOT:-outputs/multiseed_2stage}"

case "$PROTOCOL_ARG" in
    P3) PROTOCOLS=("P3") ;;
    P4) PROTOCOLS=("P4") ;;
    both) PROTOCOLS=("P3" "P4") ;;
    *) echo "ERROR: unknown protocol '$PROTOCOL_ARG'. Use P3|P4|both"; exit 1 ;;
esac

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
    echo "[$(date '+%F %T')] === two-stage pilot run starting ==="
    echo "  protocols:       ${PROTOCOLS[*]}"
    echo "  archs:           ${ARCHS[*]}"
    echo "  seeds:           [$SEED_START, $SEED_END)  (n=$((SEED_END - SEED_START)))"
    echo "  stage1_eps:      $STAGE1_EPS  (warmup $WARMUP_EPISODES)"
    echo "  stage2_eps:      $STAGE2_EPS"
    echo "  sigma:           $SIGMA_LEFT → $SIGMA_RIGHT"
    echo "  n_perturb (P4):  $N_PERTURBATIONS"
    echo "  GPU mode:        $GPU_MODE"
    echo "  output:          $OUTROOT"
} | tee -a "$GLOBAL_LOG"

# ------------------- per-arch worker --------------------
run_arch() {
    local PROTOCOL="$1"
    local ARCH="$2"
    local GPU="${ARCH_GPU[$ARCH]}"
    local ARCH_DIR="$OUTROOT/$PROTOCOL/$ARCH"
    local ARCH_LOG="$OUTROOT/${PROTOCOL}_${ARCH}.log"
    mkdir -p "$ARCH_DIR"

    if [[ -n "$GPU" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU"
    fi

    _emit() {
        local msg="$1"
        echo "$msg" >> "$ARCH_LOG"
        echo "[$PROTOCOL/$ARCH] $msg"
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
            if ! python -m cear_pilot.training.train_phase3_2stage \
                    --stage1_episodes "$STAGE1_EPS" \
                    --stage2_episodes "$STAGE2_EPS" \
                    --warmup_episodes "$WARMUP_EPISODES" \
                    --perturb_protocol "$PROTOCOL" \
                    --sigma_left "$SIGMA_LEFT" \
                    --sigma_right "$SIGMA_RIGHT" \
                    --n_perturbations "$N_PERTURBATIONS" \
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

# ------------------- Launch protocols sequentially, archs in parallel --------------------
EXIT_FAIL=0
for PROTOCOL in "${PROTOCOLS[@]}"; do
    {
        echo "[$(date '+%F %T')] === starting protocol $PROTOCOL ==="
    } | tee -a "$GLOBAL_LOG"

    declare -a PIDS
    PIDS=()
    for ARCH in "${ARCHS[@]}"; do
        run_arch "$PROTOCOL" "$ARCH" &
        PIDS+=($!)
        {
            echo "[$(date '+%F %T')] launched protocol=$PROTOCOL arch=$ARCH pid=$!  GPU=${ARCH_GPU[$ARCH]:-default}"
        } | tee -a "$GLOBAL_LOG"
        sleep 2
    done

    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            EXIT_FAIL=1
            echo "[$(date '+%F %T')] worker pid=$PID exited with non-zero status" \
                | tee -a "$GLOBAL_LOG"
        fi
    done

    {
        echo "[$(date '+%F %T')] === protocol $PROTOCOL done ==="
    } | tee -a "$GLOBAL_LOG"
done

{
    echo ""
    echo "[$(date '+%F %T')] === all protocols completed ==="
    echo "Aggregate per protocol with:"
    for PROTOCOL in "${PROTOCOLS[@]}"; do
        echo "  python -m cear_pilot.experiments.aggregate_multiseed --root $OUTROOT/$PROTOCOL"
    done
} | tee -a "$GLOBAL_LOG"

exit "$EXIT_FAIL"
