#!/bin/bash
# One-shot: train splatfacto on the bonsai COLMAP model, then evaluate and
# render held-out views so results can actually be looked at. Not baked into
# the image's CMD (that's the base image's own /start.sh, for SSH/Jupyter) --
# run this manually over SSH once the pod is up. See SETUP-PLAN.md.
set -euo pipefail

BONSAI_DATA="${BONSAI_DATA:-/workspace/data/bonsai}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"

# nerfstudio's colmap dataparser defaults to <data>/colmap/sparse/0, but the
# official Mip-NeRF 360 zip (SETUP-PLAN.md's fallback data source) ships the
# sparse model directly at <data>/sparse/0 -- detect which layout we've got.
if [[ -f "${BONSAI_DATA}/colmap/sparse/0/cameras.bin" ]]; then
    COLMAP_PATH="colmap/sparse/0"
elif [[ -f "${BONSAI_DATA}/sparse/0/cameras.bin" ]]; then
    COLMAP_PATH="sparse/0"
else
    echo "error: no COLMAP sparse model found under ${BONSAI_DATA} (checked colmap/sparse/0 and sparse/0) -- see SETUP-PLAN.md step 3" >&2
    exit 1
fi

echo "=== Training splatfacto on ${BONSAI_DATA} (colmap-path: ${COLMAP_PATH}) ==="
ns-train splatfacto \
    --data "${BONSAI_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --viewer.quit-on-train-completion True \
    colmap --colmap-path "${COLMAP_PATH}"

CONFIG="$(find "${OUTPUT_DIR}" -path '*/splatfacto/*/config.yml' -print0 | xargs -0 ls -t | head -n1)"
if [[ -z "${CONFIG}" ]]; then
    echo "error: no config.yml found under ${OUTPUT_DIR} after training" >&2
    exit 1
fi
echo "=== Trained config: ${CONFIG} ==="

echo "=== Evaluating + rendering held-out test views ==="
mkdir -p "${RESULTS_DIR}"
ns-eval \
    --load-config "${CONFIG}" \
    --output-path "${RESULTS_DIR}/psnr.json" \
    --render-output-path "${RESULTS_DIR}/renders"

echo "=== Done ==="
echo "Metrics:        ${RESULTS_DIR}/psnr.json"
echo "Held-out views: ${RESULTS_DIR}/renders/"
python3 - "${RESULTS_DIR}/psnr.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
r = d["results"]
print(f"PSNR: {r.get('psnr')}  SSIM: {r.get('ssim')}  LPIPS: {r.get('lpips')}")
PY

echo ""
echo "To view interactively instead: ns-viewer --load-config ${CONFIG}"
echo "then forward pod port 7007 (RunPod: expose as a TCP port in the pod's HTTP/TCP config, or 'ssh -L 7007:localhost:7007 <pod>')."
