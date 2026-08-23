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

## 2b. New-scene COLMAP path (`ns-process-data`) -- verified issues and fixes

Not needed for the bonsai mission itself (its sparse model ships pre-built --
step 3), but validated separately since a future scene may not have one.
Installed COLMAP via `Dockerfile.runpod`'s exact micromamba recipe
(`colmap=4.0.4` from conda-forge) on the same pod as step 0, then ran
`ns-process-data images --data <raw_photos> --output-dir <out>` end to end.
Two real bugs found and fixed, both a mismatch between this pinned COLMAP
version and what `nerfstudio/process_data/colmap_utils.py` was written
against:

- **CLI flag rename.** `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu`
  don't exist in COLMAP 4.0.4 (`unrecognised option`) -- renamed to
  `--FeatureExtraction.use_gpu` / `--FeatureMatching.use_gpu` (both default
  to GPU-on already). Fixed by probing the installed `colmap`'s own `-h`
  output at runtime rather than hardcoding a version cutoff, since the exact
  COLMAP release that renamed them isn't known -- see `_use_gpu_flag()` in
  `colmap_utils.py`. Note COLMAP prints help/usage to stderr, not stdout.
- **Default matching method (`vocab_tree`) is broken on this COLMAP
  version.** It downloads a legacy flann-based vocab tree index, but COLMAP
  switched to faiss-based indexes in May 2025 -- the pinned 4.0.4 build
  aborts trying to read it (`Check failed: file_version == 1 || file_version
  == 2`), and this build doesn't even ship `vocab_tree_upgrader` to convert
  it. Workaround: pass `--matching-method exhaustive` (fine for
  test/small-scene image counts; reconsider if a future scene has enough
  images that exhaustive's O(n^2) matching gets slow -- `sequential` is the
  other option that doesn't need a vocab tree).

Confirmed working end to end (GPU feature extraction + GPU exhaustive
matching + mapper + intrinsics refinement, `colmap/sparse/0` +
`transforms.json` produced) on 35 images with real overlap -- 35/35
registered. (A first attempt using every-8th-frame from bonsai's orbit
capture only registered 2/35 -- not a bug, just insufficient overlap between
such widely-spaced views; contiguous frames fixed it.)

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
- **Interactive viewer**: `TORCHDYNAMO_DISABLE=1 ns-viewer --load-config
  <config.yml>` on the pod (path printed at the end of `run_bonsai.sh`), then
  either expose pod port `7007` as a TCP port in the RunPod dashboard, or
  `ssh -L 7007:localhost:7007 <pod>` and open `http://localhost:7007`. The
  `TORCHDYNAMO_DISABLE=1` is required: `splatfacto.py`'s `get_viewmat` is
  `@torch_compile()`-wrapped, and under torch 2.8 the viewer's
  render-cancellation exception (thrown mid-compile whenever the camera
  moves, to abort a stale in-flight render) surfaces as an uncaught
  `InternalTorchDynamoError` instead of propagating normally -- every render
  after the first crashes and the viewer looks frozen. Training itself never
  hits this path (no camera-interrupt during training), so this only affects
  the standalone viewer.

  Two harmless cosmetic bugs to expect, both because standalone `ns-viewer`
  never attaches a live `Trainer` (unlike the viewer during an active
  `ns-train` run): the step counter stays at 0 regardless of which
  checkpoint step is actually loaded (`viewer.py`'s `update_scene` returns
  before pushing the step to the GUI, since it's called once before any
  client connects), and the "Pause Training" button is always shown
  (`train_btn_state` is derived from `self.trainer is None`, which is always
  true here, defaulting to `"training"`). Neither affects correctness --
  confirmed separately via `ns-eval`'s PSNR/rendered images that the correct
  checkpoint is what's actually loaded and rendered.

## 6. Compare

Held-out PSNR/SSIM/LPIPS (step 4) and the rendered views (step 5) against
DynamicReconstruction's `core/splat.py` output on the same scene. Good PSNR
+ visibly sharp renders here = bug is in DynamicReconstruction's training
code. Still bad = bug is upstream of training (data prep/conversion).
