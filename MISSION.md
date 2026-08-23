# Mission

Vendored copy of [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
(upstream on the `upstream` remote), deployed on RunPod for one purpose:
**produce a known-good Gaussian Splatting reconstruction of the Mip-NeRF 360
`bonsai` scene using `splatfacto`.**

Why: [DynamicReconstruction](https://github.com/rorygh/DynamicReconstruction)'s
own `core/splat.py` produced very low PSNR on the same scene. `splatfacto`
uses the same `gsplat` rasterizer DynamicReconstruction already depends on,
and has widely-reproduced published numbers on this exact scene (~27-30dB) --
running it here, on the same sparse COLMAP model, isolates whether the bug is
in DynamicReconstruction's training-loop code or somewhere upstream in data
prep.

## Scope -- deliberately minimal

- One scene (`bonsai`), one method (`splatfacto`), one pod, one comparison.
- No custom training config beyond `splatfacto` defaults.
- No CI beyond building and pushing the RunPod image on demand.
- Not a base for ongoing nerfstudio development -- upstream's own test suite,
  docs pipeline, and PyPI publish workflow were removed from `.github/workflows/`
  on purpose (see commit history). Pull from `upstream` if a newer nerfstudio
  fix is needed; don't build features here.

## Done when

`ns-train splatfacto --data <bonsai colmap dir>` completes on the pod and a
rendered held-out view is visibly sharp / in the expected PSNR range --
proving (or disproving) that the bug lives in DynamicReconstruction's own
code rather than the data or the general COLMAP -> Gaussian Splatting
approach.
