# Pipeline walkthrough

File/function references for the train -> eval -> view pipeline, as
confirmed working on the live pod (SETUP-PLAN.md). Assumes a COLMAP sparse
model already exists (`{cameras,images,points3D}.bin`); doesn't cover
generating one.

## Training -- `ns-train splatfacto --data <dir> [--output-dir ...] colmap
[--colmap-path ...]`

Entry point: `nerfstudio/scripts/train.py`.

- `entrypoint()` (:269) -> `main(config)` (:228) -> `launch(...)` (:161) ->
  `train_loop(...)` (:90), which builds a `Trainer` and calls
  `trainer.setup()` then `trainer.train()`.
- `Trainer.setup()` (`nerfstudio/engine/trainer.py:149`) builds a
  `VanillaPipeline` (`nerfstudio/pipelines/base_pipeline.py:223`) wrapping:
  - **Datamanager**: `FullImageDatamanager`
    (`nerfstudio/data/datamanagers/full_images_datamanager.py:95`) -- whole
    rendered images vs. whole ground-truth images, not per-ray sampling.
  - **Dataparser**: `ColmapDataParser`
    (`nerfstudio/data/dataparsers/colmap_dataparser.py:108`).
    `_generate_dataparser_outputs()` (:257) reads
    `data / colmap_path` (field default `Path("colmap/sparse/0")`, :99) and
    asserts it exists -- pass `--colmap-path` explicitly if your sparse
    model isn't at that default location (`run_bonsai.sh` auto-detects
    this for bonsai's actual layout).
  - **Model**: `SplatfactoModel` (`nerfstudio/models/splatfacto.py:171`).
    `get_outputs()` calls `get_viewmat()` (:66, `@torch_compile()`-wrapped)
    and gsplat's `rasterization()`.
- Defaults: `nerfstudio/configs/method_configs.py`,
  `method_configs["splatfacto"]` (:592) -- `max_num_iterations=30000`,
  `steps_per_save=2000`, `steps_per_eval_image=100`,
  `steps_per_eval_all_images=1000`. Nothing overridden here (MISSION.md:
  defaults-only).
- `Trainer.train()` (:233) loops `train_iteration()` ->
  `pipeline.get_train_loss_dict()`, saving `step-XXXXXXXXX.ckpt` +
  `config.yml` under
  `<output-dir>/<experiment>/splatfacto/<timestamp>/nerfstudio_models/`.
- Checkpoint load (`Trainer._load_checkpoint`, :420) uses
  `torch.load(..., weights_only=False)` -- fixed this session (torch 2.6
  default flip; safe since these are always our own checkpoints).

Confirmed: 30,000 iterations in ~13 min on an RTX 3090 (torch 2.8.0+cu128).

## Evaluation + renders -- `ns-eval --load-config <config.yml>
--output-path <json> --render-output-path <dir>`

Entry point: `nerfstudio/scripts/eval.py`, class `ComputePSNR` (:34).

- `main()` (:44) -> `eval_setup(load_config)`
  (`nerfstudio/utils/eval_utils.py:71`) rebuilds the same pipeline as
  training, then `eval_load_checkpoint()` (:33) loads the latest
  `step-*.ckpt` (same `weights_only=False` fix).
- `pipeline.get_average_eval_image_metrics(output_path=render_output_path,
  get_std=True)` runs the held-out test split, computing PSNR/SSIM/LPIPS
  and, if `render_output_path` is set, writing each rendered/ground-truth
  pair as a PNG.
- Result: `{experiment_name, method_name, checkpoint, results: {psnr, ssim,
  lpips, ...}}` JSON.

Confirmed on bonsai: PSNR 31.5 / SSIM 0.936 / LPIPS 0.139.

## Interactive viewer -- `ns-viewer --load-config <config.yml>`

Entry point: `nerfstudio/scripts/viewer/run_viewer.py`, class `RunViewer`.

- `main()` calls the same `eval_setup()` as `ns-eval`, then
  `_start_viewer(...)`, which wraps a `viser.ViserServer` on
  `--viewer.websocket-port` (default 7007, `EXPOSE`d in
  `Dockerfile.runpod`).
- Per-client renders go through `render_state_machine.py` ->
  `pipeline.model.get_outputs_for_camera()` -- same code path as
  training/eval, for whatever pose the client is at.
- **Requires `TORCHDYNAMO_DISABLE=1`** in the environment: `get_viewmat()`'s
  `@torch_compile()` doesn't tolerate the viewer's render-cancellation
  exception under torch 2.8 (full trace: SETUP-PLAN.md section 5).
- Two cosmetic-only quirks (standalone viewer never attaches a live
  `Trainer`): step counter always reads 0, "Pause Training" always shown.
  Doesn't affect what's rendered.

## Orchestration -- `runpod/run_bonsai.sh`

Detects `--colmap-path`, runs `ns-train splatfacto ...
--viewer.quit-on-train-completion True colmap --colmap-path <detected>`,
finds the resulting `config.yml`, then runs `ns-eval` into
`results/psnr.json` + `results/renders/`. Viewer is a separate manual step
(long-running/interactive, not scripted).

## Config/install notes

- `pyproject.toml` pins `gsplat==1.4.0` -- pure-Python wheel that
  JIT-compiles its CUDA kernels via torch's `cpp_extension` loader on first
  `rasterization()` call, keyed to whatever torch is active *then* (not at
  `pip install` time). Logged right after "Caching / undistorting train
  images".
- `Dockerfile.runpod` installs against a pip constraints file (base image's
  existing torch/torchvision/torchaudio + `Pillow==10.4.0`) instead of a
  bare `pip install -e .`, since `pyproject.toml`'s base deps
  (`torch>=1.13.1`, `Pillow>=10.3.0`) are lower-bound-only and pip
  otherwise silently upgrades both, breaking CUDA match and a
  Pillow-internals hook nerfstudio's image loader relies on. Full story:
  SETUP-PLAN.md section 2.

## Timings, vs. published numbers

All "ours" figures are from this session's actual runs (RTX 3090,
torch 2.8.0+cu128) -- see SETUP-PLAN.md for the raw logs.

| Step | Ours | Published reference | Source |
|---|---|---|---|
| COLMAP feature extraction (35 img, GPU) | ~7s | GPU SIFT extraction cited at ~45k keypoints/sec/image vs. ~2k/sec on CPU -- consistent with ours being fast, though not a direct apples-to-apples number | [pycolmap 2026 benchmarks](https://johal.in/colmap-python-bindings-feature-matching-bundle-adjustment-2026/) |
| COLMAP matching (35 img, exhaustive, GPU) | ~23s | no published small-scene (<50 img) benchmark found | -- |
| COLMAP bundle adjustment + refine (35 img) | ~6m29s (the long pole: ~70% of total SfM time) | not separately benchmarked at this scale in what's published | -- |
| splatfacto training, 30k iterations | ~13 min (292 images, RTX 3090) | RTX 3060: ~25 min; RTX 3090: ~20 min (100 photos, 1080p); RTX 2080 Ti: ~10-15 min; RTX 4090: ~8 min | [3DGS/nerfstudio 2026 field guide](https://www.youngju.dev/blog/culture/2026-05-16-3d-gaussian-splatting-photogrammetry-2026-nerfstudio-postshot-luma-genie-polycam-apple-object-capture-realitycapture-deep-dive.en) |
| Original 3DGS reference impl (INRIA, not gsplat), Mip-NeRF360 | n/a -- splatfacto uses gsplat, not the original CUDA code | 18m40s | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| gsplat vs. original 3DGS impl | consistent with the faster time above | gsplat claims up to 4x less training memory and up to 15% less training time on Mip-NeRF360 vs. the official implementation | community writeups citing the gsplat project |

Two things stand out:

- **Ours (~13 min, 292 images) beats the closest published RTX 3090 number
  (~20 min, only 100 images)** despite using ~3x more images. Plausible
  contributors, not fully isolated: the COLMAP dataparser auto-picked a 2x
  image downscale factor for bonsai (`Using image downscale factor of 2` in
  the training log -- see SETUP-PLAN.md), which directly cuts
  per-iteration rasterization cost; gsplat has also kept improving since
  whatever version that blog benchmark used. Not a controlled comparison,
  just the two closest numbers available.
- **COLMAP's bundle adjustment dominates SfM time** even at just 35 images
  (~70% of the ~7 min total) -- feature extraction and matching are both
  GPU-accelerated and fast, but the mapper's incremental bundle adjustment
  is CPU-bound (Ceres solver) and doesn't benefit from the GPU at all. Worth
  planning around if a future scene has hundreds of images instead of 35 --
  this step, not GPU matching, is what will actually dominate wall time.
