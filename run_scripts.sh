#!/usr/bin/env bash
set -Eeuo pipefail

# --- repo root ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- Make third_party importable (so PaTAT_piped resolves) ---
export PYTHONPATH="$ROOT_DIR/third_party:${PYTHONPATH:-}"

# Ensure PaTAT_piped is a package and create a shim for 'synthesizer' imports if needed
test -f "$ROOT_DIR/third_party/PaTAT_piped/__init__.py" || touch "$ROOT_DIR/third_party/PaTAT_piped/__init__.py"
[ -e "$ROOT_DIR/third_party/synthesizer" ] || ln -sfn "$ROOT_DIR/third_party/PaTAT_piped" "$ROOT_DIR/third_party/synthesizer"

set -x

# --- Read config values (trim whitespace) ---
environment="$(awk -F= '/^[[:space:]]*environment[[:space:]]*=/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' config.ini || true)"
openai_key="$(awk -F= '/^[[:space:]]*openai_api[[:space:]]*=/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' config.ini || true)"
echo "Environment: ${environment:-<none>}"

# --- Housekeeping ---
rm -f user_checkpoints/test.json

# --- ensure checkpoint for current dataset exists ---
mkdir -p user_checkpoints
stem="$(awk -F= '/^[[:space:]]*data_file[[:space:]]*=/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); sub(/\.csv$/,"",$2); print $2}' config.ini)"
[ -f "user_checkpoints/${stem}.json" ] || echo '{}' > "user_checkpoints/${stem}.json"

# --- Helper: prefer conda env if it exists, else use current python ---
run_py () {
  local script="$1"
  if [[ -n "${environment:-}" ]] && command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "$environment"; then
      echo "▶ conda run -n $environment python $script"
      conda run -n "$environment" python "$script"
      return
    fi
  fi
  echo "python $script"
  python "$script"
}

# --- Pipeline ---

# 01 - Pattern discovery / annotation (uses PaTAT)
run_py 01_data_formatting.py

# 02 - Counterfactual generation (requires real OpenAI key)
if [[ -n "${openai_key:-}" && "${openai_key}" != SK-PLACEHOLDER* && "${openai_key}" != "" && "${openai_key}" != "<<API_KEY>>" ]]; then
  run_py 02_counterfactual_over_generation.py
else
  echo "Skipping 02_counterfactual_over_generation.py (no real OpenAI key in config.ini)"
fi

# 03 - Symbolic filtering (PaTAT again)
run_py 03_counterfactual_filtering.py

# 04 - Fine-tuning (optional)
# run_py 04_fine_tuning.py

# 05 - Active Learning experiments (optional)
run_py 05_AL_testing_BERT.py
run_py 05_AL_testing.py