#!/usr/bin/env bash
#
# Standardized Full-Scale Benchmark Harness for Undertow
# Evaluates 235B - 671B MoE models across 4 operating conditions:
#   1. Cold Start (clean cache, no pre-pinned experts)
#   2. Warm Unpinned (LRU cache after initial decode)
#   3. Pinned Working Set (Top 25% hottest experts pinned)
#   4. MTP Speculative Decoding (native draft head verification)
#
set -euo pipefail

MODEL_PATH="${1:-}"
OUTPUT_DIR="${2:-./bench_results}"
PROMPT="${3:-Explain the architectural trade-offs between dense model scaling and sparse Mixture-of-Experts inference when DRAM capacity is constrained by cost.}"
MAX_NEW="${4:-256}"
BUDGET_MB="${5:-32768}" # 32 GB RAM budget for expert cache by default

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 <model_dir> [output_dir] [prompt] [max_new] [cache_budget_mb]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
PROFILE_PATH="$OUTPUT_DIR/hot_experts.json"
RESULTS_FILE="$OUTPUT_DIR/benchmark_summary.md"

echo "======================================================================"
echo "Undertow Full-Scale Benchmark Protocol"
echo "Model:         $MODEL_PATH"
echo "Output Dir:    $OUTPUT_DIR"
echo "Max Tokens:    $MAX_NEW"
echo "Cache Budget:  ${BUDGET_MB} MB"
echo "Date:          $(date -u)"
echo "======================================================================"

drop_caches() {
    echo "Flushing OS filesystem caches..."
    if [[ "$(uname)" == "Darwin" ]]; then
        sudo purge 2>/dev/null || true
    elif [[ -f /proc/sys/vm/drop_caches ]]; then
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    fi
}

run_phase() {
    local phase_name="$1"
    local extra_args="$2"
    local log_file="$OUTPUT_DIR/${phase_name}.log"

    echo ""
    echo "----------------------------------------------------------------------"
    echo ">>> Running Phase: $phase_name"
    echo "----------------------------------------------------------------------"

    # Run undertow with time measurement
    /usr/bin/time -l cargo run --release -p undertow-cli -- run \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-new "$MAX_NEW" \
        --cache-budget-mb "$BUDGET_MB" \
        --stats \
        $extra_args 2>&1 | tee "$log_file"
}

# 1. Cold Start
drop_caches
run_phase "01_cold_start" "--profile-out $PROFILE_PATH"

# 2. Warm Unpinned
run_phase "02_warm_unpinned" ""

# 3. Pinned Working Set (Top 25% hot experts pinned into memory)
run_phase "03_pinned_working_set" "--profile $PROFILE_PATH --cache-policy weighted"

# 4. MTP Speculative Decoding
if cargo run --release -p undertow-cli -- run --model "$MODEL_PATH" --help | grep -q "\-\-mtp"; then
    run_phase "04_mtp_speculative" "--profile $PROFILE_PATH --cache-policy weighted --mtp"
fi

echo ""
echo "======================================================================"
echo "Benchmark Suite Completed Successfully."
echo "Results logged in: $OUTPUT_DIR"
echo "======================================================================"
