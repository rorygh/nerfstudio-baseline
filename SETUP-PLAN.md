# Setup Plan

## 0. Pod

Original plan was to reuse DynamicReconstruction's exact pod (RTX 2000 Ada,
16GB, CUDA 12.4) to hold hardware constant across the comparison. In
practice, the pod actually used for the confirmed-working run below is a
different one: a generic RunPod PyTorch template, RTX 3090 (24GB), driver
CUDA 13.0 / nvcc 12.8, Ubuntu 24.04, Python 3.12, torch 2.8.0+cu128
pre-installed -- not built from this repo's `Dockerfile.runpod` (no Docker
available on the pod itself). Notably *not* the same GPU/CUDA as
DynamicReconstruction's pod, so hardware is no longer held constant between
the two -- worth keeping in mind if results are borderline instead of
clearly good/bad. No `tiny-cuda-nn` involved either way (splatfacto only), so
the usual `TCNN_CUDA_ARCHITECTURES`-vs-compute-capability gotcha doesn't
apply.

## 1. Image

Build via this repo's `docker-publish.yml` (`workflow_dispatch` only --
trigger manually from the Actions tab or `gh workflow run docker-publish.yml`
once `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets are set on *this* repo --
they don't carry over from DynamicReconstruction). Deploy the pod from
`rorygh/nerfstudio-baseline:latest`, same as DynamicReconstruction's pattern.

## 2. Environment setup -- verified issues and fixes

Confirmed on the live pod described in step 0. All three bit regardless of
which exact base image is used, since the root cause in each case is
`pyproject.toml`'s dependencies being lower-bound-only:

- **`pip install -e .` silently upgrades torch.** Base deps say
  `torch>=1.13.1` with no upper bound, so pip grabbed the newest torch
  available (a different CUDA major version than the pod's pre-installed,
  driver-matched one), which broke `torchaudio` and left an unverified
  torch/CUDA combo in place. Fix (baked into `Dockerfile.runpod`): capture
  the base image's existing torch/torchvision/torchaudio versions into a
  pip constraints file *before* running `pip install -e .`, then install
  against that constraint so torch never moves.
- **Newer Pillow breaks `pil_to_numpy`.** Base deps say `Pillow>=10.3.0`
  with no upper bound; pip grabbed the latest, whose C-internals no longer
  match what `nerfstudio/data/utils/data_utils.py`'s `pil_to_numpy` hooks
  into (`Image._getencoder(...).setimage(...)` signature changed) --
  `TypeError: function takes exactly 2 arguments (1 given)` on the very
  first image load. Fix: pin `Pillow==10.4.0` (confirmed working) in the
  same constraints file.
- **`gsplat==1.4.0` is a pure-Python wheel**, not a prebuilt CUDA binary --
  it JIT-compiles its CUDA kernels via torch's `cpp_extension` loader the
  first time you actually train, keyed off whatever torch is active *then*.
  So the install-time torch/CUDA match matters less than expected, but the
  JIT-compile step (logged right after "Caching / undistorting train
  images") is still worth watching on a freshly built image. Confirmed
  working against torch 2.8.0+cu128 / RTX 3090.
- **`torch.load` default flip in torch 2.6** (`weights_only` now defaults to
  `True`) broke checkpoint loading in `nerfstudio/utils/eval_utils.py` and
  `nerfstudio/engine/trainer.py` (`ns-eval` and checkpoint-resume both
  failed with `UnpicklingError` on a `numpy._core.multiarray.scalar`
  global). Fixed in this repo by passing `weights_only=False` explicitly at
  all three call sites -- safe since these are always our own
  self-generated checkpoints, never third-party ones.

## 3. Get bonsai data onto the pod

Preferred: reuse the bonsai COLMAP sparse model DynamicReconstruction already
generated (same camera poses/points that produced the low-PSNR result) --
`scp`/`runpodctl send` it over rather than re-running SfM. This holds SfM
constant and isolates the training loop specifically, per the isolation-test
reasoning in DynamicReconstruction's `docs/pipeline-alternatives.md`.

Fallback if that model isn't available: download the `bonsai` scene from the
[Mip-NeRF 360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
(~12GB zip, extract just `bonsai/`) directly onto the pod. **Used for the
confirmed-working run below** -- DynamicReconstruction's pod/data wasn't
reachable from this one. Its sparse model lands at `<data>/sparse/0`, not
nerfstudio's default `<data>/colmap/sparse/0` -- `run_bonsai.sh` detects
which layout is present and passes the right `--colmap-path` automatically.

## 4. Train, evaluate, and render

```bash
BONSAI_DATA=<bonsai_dir> runpod/run_bonsai.sh
```

Runs `ns-train splatfacto ... colmap` (defaults only, per MISSION.md's "don't
overengineer"), then `ns-eval` against the held-out test split, writing
`results/psnr.json` (PSNR/SSIM/LPIPS) and `results/renders/` (the actual
held-out-view images -- this is what "visibly sharp" in MISSION.md's Done
criterion is judged against, not just the PSNR number).

Both `ns-train` and `ns-eval`/`ns-render` pick up CUDA automatically when
available -- no device flag needed, nothing here should silently fall back to
CPU on this GPU pod.

## 5. View the results

Two ways, not mutually exclusive:

- **Pull the renders down**: `runpodctl send results/` on the pod, then
  `runpodctl receive <code>` locally (same tool used for the bonsai data in
  step 3) -- or `scp -r`. Then just open `results/renders/` and
  `results/psnr.json` locally.
- **Interactive viewer**: `ns-viewer --load-config <config.yml>` on the pod
  (path printed at the end of `run_bonsai.sh`), then either expose pod port
  `7007` as a TCP port in the RunPod dashboard, or `ssh -L 7007:localhost:7007
  <pod>` and open `http://localhost:7007`.

## 6. Compare

Held-out PSNR/SSIM/LPIPS (step 4) and the rendered views (step 5) against
DynamicReconstruction's `core/splat.py` output on the same scene. Good PSNR
+ visibly sharp renders here = bug is in DynamicReconstruction's training
code. Still bad = bug is upstream of training (data prep/conversion).
