# Fifteenth pass (declared in LOG.md before the run)

## A. Causal test on ETH3D with the DTU-fitted directions (nine scenes, five adjacent views)

p: exact paired sign-flip over the nine scenes, support direction ablated against that direction.

| backbone | condition | head AUROC | native AUROC | AbsRel supported | AbsRel single-view | ablate - this: head AUROC (p) | AbsRel single-view (p) |
|---|---|---|---|---|---|---|---|
| vggt | none | 0.981 | 0.615 | 0.0105 | 0.0283 | | |
| vggt | ablate | 0.829 | 0.559 | 0.0647 | 0.0879 | | |
| vggt | rand | 0.981 | 0.614 | 0.0107 | 0.0289 | | |
| vggt | pc0 | 0.948 | 0.545 | 0.0295 | 0.0385 | -0.119 (0.0703) | +0.0494 (0.0156) |
| vggt | pc1 | 0.970 | 0.578 | 0.0362 | 0.0724 | -0.141 (0.0234) | +0.0155 (0.7383) |
| vggt | pc2 | 0.946 | 0.525 | 0.0534 | 0.1070 | -0.117 (0.0195) | -0.0191 (0.5859) |
| pi3x | none | 0.970 | 0.871 | 0.0095 | 0.0216 | | |
| pi3x | ablate | 0.829 | 0.778 | 0.0156 | 0.0295 | | |
| pi3x | rand | 0.969 | 0.871 | 0.0096 | 0.0217 | | |
| pi3x | pc0 | 0.971 | 0.866 | 0.1361 | 0.2095 | -0.142 (0.0039) | -0.1800 (0.0039) |
| pi3x | pc1 | 0.969 | 0.868 | 0.0151 | 0.0317 | -0.140 (0.0078) | -0.0022 (0.6484) |
| pi3x | pc2 | 0.946 | 0.856 | 0.0809 | 0.1593 | -0.117 (0.0195) | -0.1297 (0.0039) |
| pi3 | none | 0.953 | 0.786 | 0.0105 | 0.0264 | | |
| pi3 | ablate | 0.874 | 0.691 | 0.0599 | 0.0905 | | |
| pi3 | rand | 0.953 | 0.786 | 0.0105 | 0.0264 | | |
| pi3 | pc0 | 0.948 | 0.771 | 0.0129 | 0.0307 | -0.074 (0.0469) | +0.0598 (0.0039) |
| pi3 | pc1 | 0.938 | 0.779 | 0.0193 | 0.0447 | -0.064 (0.0703) | +0.0457 (0.0039) |
| pi3 | pc2 | 0.911 | 0.816 | 0.0244 | 0.0439 | -0.037 (0.3594) | +0.0465 (0.0039) |

**Stated criterion not met:** the support direction lowers the head's AUROC more than the principal direction at p < 0.05 in 6 of 9 comparisons (needed: all nine).

## B. HAD wrapper diagnosis (scan 37, five views; `results/P22-had-diag/diag.log`)

- Checkpoint keys: model 365, checkpoint 365, missing 0, unexpected 0, shape mismatch 0.
- Render PSNR, half / full precision: 408x512 9.08 / 9.08 dB; 512x512 8.87 / 8.87; 288x512 9.23 / 9.23; 256x320 9.13 / 9.13.
- Identity (target is one of the references): 24.91 dB.
- Pose convention and scene scale factor (0.1, 0.25, 0.5, 1.0, 1.35): camera-to-world, OpenCV axes (the wrapper's) 8.65, 10.38, 8.72, 9.63, 9.08 dB; world-to-camera 8.21, 8.50, 7.78, 6.11, 5.70; OpenGL axes 8.72, 9.14, 7.94, 8.47, 8.42.
- Camera centres up to 872 mm apart, median depth 738 mm.
- Conclusion: no fault found in the wrapper; the 15 dB threshold is not reached; no HAD rows are produced.
