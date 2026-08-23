# Setup Plan (not yet executed -- run on the pod, not before)

## 0. Pod

Reuse the *exact same* RunPod pod already deployed for DynamicReconstruction
(RTX 2000 Ada, 16GB VRAM, CUDA 12.4 driver -- see that repo's Dockerfile.runpod
and its confirmed-working `nvidia-smi` output) rather than deploying a new
one. Reasoning:

- No `tiny-cuda-nn` in this mission (splatfacto only), so nerfstudio's usual
  "match `TCNN_CUDA_ARCHITECTURES` to your GPU" gotcha doesn't apply here --
  compute capability is a non-issue.
- 16GB comfortably covers a single-object scene at `bonsai`'s scale for
  `splatfacto` (same class of workload DynamicReconstruction already ran on
  this hardware).
- Reusing the same pod holds GPU/driver/CUDA fixed as a variable in the
  comparison -- a difference in results can't be blamed on different
  hardware.

Only spin up a separate pod if VRAM turns out insufficient once actually
running (unlikely at this scene scale).

## 1. Image

Build via this repo's `docker-publish.yml` (`workflow_dispatch` only --
trigger manually from the Actions tab or `gh workflow run docker-publish.yml`
once `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets are set on *this* repo --
they don't carry over from DynamicReconstruction). Deploy the pod from
`rorygh/nerfstudio-baseline:latest`, same as DynamicReconstruction's pattern.

## 2. Known open risk -- verify first, before anything else

`pyproject.toml` pins `gsplat==1.4.0`. The base image (`runpod/pytorch:2.4.0-
py3.11-cuda12.4.1-devel-ubuntu22.04`) ships torch 2.4.1+cu124. Unknown until
tested: whether a `gsplat==1.4.0` wheel exists for that exact torch/CUDA
combo, or whether it needs a source build (slower, needs nvcc -- the base
image is a `-devel` image so nvcc is present, but untested here). Check
`pip install -e .`'s output during the image build (or interactively on the
pod) before assuming the image built cleanly.

## 3. Get bonsai data onto the pod

Preferred: reuse the bonsai COLMAP sparse model DynamicReconstruction already
generated (same camera poses/points that produced the low-PSNR result) --
`scp`/`runpodctl send` it over rather than re-running SfM. This holds SfM
constant and isolates the training loop specifically, per the isolation-test
reasoning in DynamicReconstruction's `docs/pipeline-alternatives.md`.

Fallback if that model isn't available: download the `bonsai` scene from the
[Mip-NeRF 360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
(~12GB zip, extract just `bonsai/`) directly onto the pod.

## 4. Train

```bash
ns-train splatfacto --data <bonsai_dir> colmap   # verify exact dataparser invocation/flags on pod --
                                                   # nerfstudio's colmap dataparser expects
                                                   # images/ + sparse/0/{cameras,images,points3D}.bin
```

No custom config -- `splatfacto` defaults only, per MISSION.md's "don't
overengineer."

## 5. Compare

Render a held-out view, compare against DynamicReconstruction's `core/splat.py`
output on the same scene. Good PSNR here = bug is in DynamicReconstruction's
training code. Still bad = bug is upstream of training (data prep/conversion).
