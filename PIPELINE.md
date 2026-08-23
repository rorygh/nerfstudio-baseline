# Pipeline walkthrough

What actually runs, file by file and function by function, for the workflow
this repo exists to do (see MISSION.md). Written from the confirmed-working
run on the live pod (SETUP-PLAN.md) -- every path/function name below has
been exercised directly, not read off documentation.

Two independent paths feed into the same training step:

```
A) have a reconstruction already      B) don't have one yet
   (bonsai: ships with the dataset)      (ns-process-data)
              |                                    |
              v                                    v
     <scene>/{colmap/}sparse/0/          ns-process-data images
     {cameras,images,points3D}.bin              |
              |                                  v
              |                        colmap/sparse/0/*.bin + transforms.json
              |                                  |
              +----------------+-----------------+
                                v
                      ns-train splatfacto ... colmap
                                |
                    +-----------+-----------+
                    v                       v
                ns-eval                ns-viewer
        (PSNR/SSIM/LPIPS + renders)  (interactive, port 7007)
```

## A. Reconstruction path (when you don't have one) -- `ns-process-data`

Entry point: `nerfstudio/scripts/process_data.py`, CLI `ns-process-data
images --data <raw_photos> --output-dir <out> [--matching-method exhaustive]`.

- Dispatches (via tyro subcommand) to `ImagesToNerfstudioDataset.main()`
  in `nerfstudio/process_data/images_to_nerfstudio_dataset.py:36`.
- That calls `self._run_colmap()`, defined on the shared base class
  `ColmapConverterToNerfstudioDataset` in
  `nerfstudio/process_data/colmap_converter_to_nerfstudio_dataset.py:183`,
  which calls `colmap_utils.run_colmap(...)`.
- `run_colmap()` in `nerfstudio/process_data/colmap_utils.py:92` shells out
  to the `colmap` binary in sequence:
  1. `colmap feature_extractor` -- SIFT features per image, GPU by default
     (`gpu: bool = True` param). Flag name is version-probed at runtime via
     `_use_gpu_flag()` (`colmap_utils.py:92`, added this session) because
     COLMAP renamed `--SiftExtraction.use_gpu` -> `--FeatureExtraction.use_gpu`
     at some point and the pinned COLMAP 4.0.4 (conda-forge, see
     `Dockerfile.runpod`) only accepts the new name.
  2. `colmap {matching_method}_matcher` -- default `vocab_tree`, but that's
     currently broken on this COLMAP build (legacy flann index vs. COLMAP's
     May-2025 switch to faiss -- see SETUP-PLAN.md section 2b). Pass
     `--matching-method exhaustive` (or `sequential`) to avoid it.
  3. `colmap mapper` -- bundle adjustment, produces `colmap/sparse/0/*.bin`.
  4. `colmap bundle_adjuster` -- intrinsics refinement (`refine_intrinsics`,
     default on).
- Back in `colmap_converter_to_nerfstudio_dataset.py:136`,
  `colmap_utils.colmap_to_json(...)` converts the COLMAP binary model into
  nerfstudio's own `transforms.json` (camera poses + intrinsics in
  nerfstudio's convention) plus `sparse_pc.ply` (point cloud, used for
  splatfacto's initial Gaussian positions).

Output layout: `<out>/images{,_2,_4,_8}/`, `<out>/colmap/sparse/0/*.bin`,
`<out>/transforms.json`, `<out>/sparse_pc.ply`. This is nerfstudio's default
colmap-path (`colmap/sparse/0`), which is why a scene reconstructed this way
needs no `--colmap-path` override at train time (see B below).

Verified: 35 real photos with actual frame-to-frame overlap -> 35/35
registered, GPU-accelerated extraction + matching, ~7 min total.

## B. Existing-reconstruction path (bonsai's actual path)

No script needed -- the Mip-NeRF 360 zip already ships
`bonsai/sparse/0/{cameras,images,points3D}.bin` (COLMAP binary format
directly, no `transforms.json`). This is *not* nerfstudio's default
colmap-path (`<data>/colmap/sparse/0`), so `--colmap-path sparse/0` must be
passed explicitly -- `runpod/run_bonsai.sh` detects which of the two layouts
is present and sets this automatically (checks for
`colmap/sparse/0/cameras.bin` first, falls back to `sparse/0/cameras.bin`).

## Training -- `ns-train splatfacto --data <dir> [--output-dir ...] colmap
[--colmap-path ...]`

Entry point: `nerfstudio/scripts/train.py`.

- `entrypoint()` (:269) -> `main(config)` (:228) -> `launch(...)` (:161) ->
  `train_loop(local_rank, world_size, config)` (:90), which builds a
  `Trainer` from `config` and calls `trainer.setup()` then `trainer.train()`.
- `Trainer.setup()` (`nerfstudio/engine/trainer.py:149`) builds the
  `Pipeline` via `config.pipeline.setup(device=..., ...)`. For splatfacto
  this instantiates a `VanillaPipeline` (`nerfstudio/pipelines/base_pipeline.py:223`)
  wrapping:
  - **Datamanager**: `FullImageDatamanager`
    (`nerfstudio/data/datamanagers/full_images_datamanager.py:95`) --
    splatfacto trains on whole rendered images compared against whole
    ground-truth images (rasterization, not per-ray sampling like NeRF
    methods), hence "full image".
  - **Dataparser**: `ColmapDataParser`
    (`nerfstudio/data/dataparsers/colmap_dataparser.py:108`).
    `_generate_dataparser_outputs()` (:257) reads
    `self.config.data / self.config.colmap_path` (default
    `Path("colmap/sparse/0")`, field at :99) and asserts it exists -- this is
    the exact assertion that fails without the right `--colmap-path` (path
    B above).
  - **Model**: `SplatfactoModel` (`nerfstudio/models/splatfacto.py:171`),
    config `SplatfactoModelConfig` (:85). `get_outputs()` calls
    `get_viewmat()` (:66, `@torch_compile()`-wrapped -- the source of the
    viewer's dynamo crash, SETUP-PLAN.md) and gsplat's `rasterization()` to
    render Gaussians into an image.
- Method defaults: `nerfstudio/configs/method_configs.py`,
  `method_configs["splatfacto"]` (:592) -- `max_num_iterations=30000`,
  per-parameter-group optimizers (means/scales/quats/opacities/features,
  camera_opt, bilateral_grid), `steps_per_save=2000`,
  `steps_per_eval_image=100`, `steps_per_eval_all_images=1000`. Nothing
  overridden by this repo -- MISSION.md is explicit about defaults-only.
- `Trainer.train()` (:233) loops `train_iteration()`, periodically calling
  `pipeline.get_train_loss_dict()`
  (`VanillaPipeline`, backed by `SplatfactoModel.get_outputs()` +
  `get_loss_dict()`), saving checkpoints (`step-XXXXXXXXX.ckpt` under
  `<output-dir>/<experiment>/splatfacto/<timestamp>/nerfstudio_models/`) and
  a `config.yml` snapshot alongside.
- Checkpoint load/resume (`Trainer._load_checkpoint`, :420) uses
  `torch.load(..., weights_only=False)` -- fixed this session (torch 2.6
  flipped the default; these are always our own checkpoints, never
  third-party, so this is safe).

Confirmed: 30,000 iterations in ~13 min on an RTX 3090 (torch 2.8.0+cu128),
steady-state ~13ms/iter after densification kicks in.

## Evaluation + held-out renders -- `ns-eval --load-config <config.yml>
--output-path <json> --render-output-path <dir>`

Entry point: `nerfstudio/scripts/eval.py`, class `ComputePSNR` (:34).

- `main()` (:44) calls `eval_setup(self.load_config)`
  (`nerfstudio/utils/eval_utils.py:71`), which: loads the saved `config.yml`,
  rebuilds the same `Pipeline` as training (`config.pipeline.setup(...)`,
  `pipeline.eval()`), then `eval_load_checkpoint(config, pipeline)` (:33)
  finds the latest `step-*.ckpt` in `config.get_checkpoint_dir()` and loads
  it with `torch.load(..., weights_only=False)` (fixed this session, same
  reason as training's checkpoint load).
- `pipeline.get_average_eval_image_metrics(output_path=render_output_path,
  get_std=True)` runs the model against the held-out test split (COLMAP
  dataparser's train/test image split), computing PSNR/SSIM/LPIPS per image
  and averaging; if `render_output_path` is set, also writes each rendered
  vs. ground-truth image pair as a PNG there.
- Result written to `<output-path>` as JSON:
  `{experiment_name, method_name, checkpoint, results: {psnr, ssim, lpips,
  ...}}`.

Confirmed result on bonsai: PSNR 31.5 / SSIM 0.936 / LPIPS 0.139 (30k-step
checkpoint), visually confirmed sharp against the published ~27-30dB range.

## Interactive viewer -- `ns-viewer --load-config <config.yml>`

Entry point: `nerfstudio/scripts/viewer/run_viewer.py`, class `RunViewer`.

- `main()` calls the same `eval_setup()` as `ns-eval` (loads config +
  checkpoint), then `_start_viewer(config, pipeline, step)`, which
  constructs a `Viewer` (`nerfstudio/viewer/viewer.py`) wrapping a
  `viser.ViserServer` bound to `--viewer.websocket-port` (default 7007,
  `EXPOSE`d in `Dockerfile.runpod`).
- `viewer_state.init_scene(train_dataset, train_state="completed",
  eval_dataset)` (:432) draws camera-frustum thumbnails for the training
  images in 3D; clicking one snaps the viewport to that exact capture pose.
- Per-client renders are driven by `render_state_machine.py`, which calls
  `pipeline.model.get_outputs_for_camera()` -> `SplatfactoModel.get_outputs()`
  (same code path as training/eval) for whatever camera pose the client is
  currently at.
- **Requires `TORCHDYNAMO_DISABLE=1`** in the environment (not a code
  change): `get_viewmat()`'s `@torch_compile()` doesn't tolerate the
  viewer's render-cancellation exception under torch 2.8 -- see
  SETUP-PLAN.md section 5 for the full trace.
- Two cosmetic-only quirks (standalone viewer never attaches a live
  `Trainer`, unlike the viewer embedded in an active `ns-train` run): the
  step counter always reads 0, and "Pause Training" is always shown
  regardless of state. Neither affects what's actually rendered.

## Orchestration script -- `runpod/run_bonsai.sh`

Ties training + eval together for the one-command case: detects
`--colmap-path` (path A vs. B above), runs `ns-train splatfacto ...
--viewer.quit-on-train-completion True colmap --colmap-path <detected>`,
finds the resulting `config.yml`, then runs `ns-eval` against it into
`results/psnr.json` + `results/renders/`. Doesn't touch the viewer -- that's
a separate manual step (see above), since it's a long-running interactive
process rather than a one-shot command.

## Config/install notes worth knowing

- `pyproject.toml` pins `gsplat==1.4.0` -- it's a pure-Python wheel that
  JIT-compiles its actual CUDA kernels via torch's `cpp_extension` loader
  the first time `rasterization()` is called, keyed to whatever torch is
  active *then* (not at `pip install` time). Watch for this the first time
  a new image trains -- logged right after "Caching / undistorting train
  images".
- `Dockerfile.runpod` installs against a pip constraints file (captures the
  base image's existing torch/torchvision/torchaudio versions, adds
  `Pillow==10.4.0`) instead of a bare `pip install -e .`, because
  `pyproject.toml`'s base deps (`torch>=1.13.1`, `Pillow>=10.3.0`) are
  lower-bound-only and pip will otherwise silently upgrade both, breaking
  the base image's CUDA match and a Pillow-internals hook nerfstudio's
  image loader relies on. Full story in SETUP-PLAN.md section 2.
