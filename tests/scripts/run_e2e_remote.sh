#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-testplan"
PYTHON_BIN="${VENV_DIR}/bin/python"

cd "$ROOT_DIR"

if [[ "${E2E_LOAD_ENV:-1}" != "0" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

OUT_DIR="${E2E_OUT_DIR:-e2e_runs/remote}"
# The harness accepts --out-dir overrides; report discovery must follow the
# same final value (including --out-dir=PATH and paths containing spaces).
EXPECT_OUT_DIR=0
for ARG in "$@"; do
  if [[ "$EXPECT_OUT_DIR" -eq 1 ]]; then
    OUT_DIR="$ARG"
    EXPECT_OUT_DIR=0
  elif [[ "$ARG" == "--out-dir" ]]; then
    EXPECT_OUT_DIR=1
  elif [[ "$ARG" == --out-dir=* ]]; then
    OUT_DIR="${ARG#--out-dir=}"
  fi
done
if [[ "$OUT_DIR" = /* ]]; then
  OUT_DIR_PATH="$OUT_DIR"
else
  OUT_DIR_PATH="${ROOT_DIR}/${OUT_DIR}"
fi

if [[ -z "${BASE_URL:-}" ]]; then
  echo "FAIL: set BASE_URL to the deployed service URL before running remote e2e tests" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is not installed or not on PATH" >&2
  exit 1
fi

uv venv --allow-existing "$VENV_DIR"
uv pip install --python "$PYTHON_BIN" -r requirements-api.txt pytest pytest-asyncio

echo
echo "================================================================================"
echo "Running remote e2e harness against ${BASE_URL}"
echo "Posts WhatsApp webhook payloads to the deployed /message endpoint."
echo "Remote structured verdicts require ENABLE_E2E_DEBUG=true on the service."
echo "Remote media cases require expected.json media_id values."
echo "The deployed service downloads media through the WhatsApp Graph API."
# Do not enable local-only auth bypass, outbound suppression, or media stubs.
# Remote service configuration controls authentication and WhatsApp behavior.
echo "================================================================================"
echo

CMD=("$PYTHON_BIN" scripts/e2e_cases.py --base-url "$BASE_URL" --debug-tools --out-dir "$OUT_DIR")
if [[ -n "${WEBHOOK_SECRET:-}" ]]; then
  CMD+=(--secret "$WEBHOOK_SECRET")
fi
CMD+=("$@")

latest_results_json() {
  if [[ ! -d "$OUT_DIR_PATH" ]]; then
    return 0
  fi
  find "$OUT_DIR_PATH" -mindepth 2 -maxdepth 2 -type f -name results.json -print \
    | sort \
    | tail -n 1
}

PREVIOUS_RESULTS_JSON="$(latest_results_json)"

if "${CMD[@]}"; then
  E2E_STATUS=0
else
  E2E_STATUS=$?
fi

RESULTS_JSON="$(latest_results_json)"
MARKDOWN_STATUS=0
if [[ -n "$RESULTS_JSON" && "$RESULTS_JSON" != "$PREVIOUS_RESULTS_JSON" ]]; then
  echo
  echo "Generating clean Markdown report from ${RESULTS_JSON}"
  if "$PYTHON_BIN" tests/scripts/clean_results_to_markdown.py "$RESULTS_JSON"; then
    MARKDOWN_STATUS=0
  else
    MARKDOWN_STATUS=$?
  fi
else
  echo "FAIL: the E2E run did not produce a new results.json under ${OUT_DIR_PATH}" >&2
  MARKDOWN_STATUS=1
fi

if [[ "$E2E_STATUS" -ne 0 ]]; then
  exit "$E2E_STATUS"
fi
exit "$MARKDOWN_STATUS"
