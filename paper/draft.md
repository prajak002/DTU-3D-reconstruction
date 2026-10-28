# Observe, Imagine, Abstain: Hallucination-Aware 3D Reconstruction with Discrete Diffusion and Continual Adaptation

Draft v0.1, 2026-10-28. Anonymous submission, CVPR 2027.

Status key: every number marked `TBD(run-id)` is filled only from the CSV produced by that run (see `EXPERIMENTS.md`). No number in this draft is estimated or invented. Numbers that already exist come from `results/synthetic/` and are labelled as synthetic.

## Abstract

Feed-forward 3D foundation models reconstruct a scene from unposed photos in a single pass. Where no input view observes a surface, these models fill the gap from learned priors, and they often assign the invented geometry a confidence similar to that of observed geometry. We call this geometric hallucination and give it an operational definition: a predicted point is hallucinated if it lies on a surface that no input view shows. We build HalluBench-3D, a benchmark whose labels are free by construction: a 3D region on the ground-truth surface is hidden in every input view, so every point inside it is unsupported, and every point far from it is supported. We measure whether a model's native confidence separates the two (AUROC) and how much unsupported geometry survives a confidence threshold that keeps 90% of supported geometry (Confident-Unsupported Rate, CUR). Across five models (VGGT, Pi3, Pi3X, MapAnything, Depth Anything 3) on 18 DTU scans with 5 mask placements, 2 region sizes and 3 fill types, native confidence lets between 24% (VGGT) and 70% (Depth Anything 3) of unsupported pixels through a threshold that keeps 90% of supported pixels, with AUROC from 0.90 down to 0.71. A cheap alternative, variance over perturbed inputs, is worse than native confidence on both backbones tested. We then propose a method that does not ban imagination but makes it explicit: a frozen 3D backbone with a small discrete diffusion head draws K geometry samples; agreement marks observed geometry, disagreement marks imagined geometry, and the remainder abstains. On HalluBench-3D, K-sample disagreement reaches AUROC TBD(P3) and CUR TBD(P3) versus TBD(P3) for native confidence at equal compute. A LoRA adapter with a small replay buffer lets the head adapt to a stream of new sites with TBD(P4) forgetting. Code and benchmark will be released.

## 1. Introduction

Figure 1 (candidate: `results/P2-stress/ex_P2-stress_inpaint10/vggt_scan17_r0.18.png`): left, DTU scan 17 with one region hidden in all ten views and inpainted so no visible seam remains; middle, VGGT depth, which reconstructs a plausible facade inside the hidden region, and VGGT confidence, which drops only partially there; right, TBD(P3-main) our observed / imagined / abstain labels on the same pixels.

Feed-forward 3D reconstruction has moved from pairwise pointmaps (DUSt3R) to models that process dozens of unposed images in one pass (VGGT, Pi3, MapAnything, Depth Anything 3). These models are trained to output dense geometry for every pixel, including pixels whose true surface is not observed by any view. The training objective rewards a plausible guess, so the models learn to guess. The guess is often good, and that is useful. The problem is that the model's confidence output does not tell the user which parts were observed and which were guessed. A conservator deciding whether a crack in a dome is real cannot act on a guess.

We make three contributions.

1. **A definition and a benchmark.** We define hallucinated geometry as predicted points on surfaces that no input view shows. HalluBench-3D produces these labels by 3D-consistent masking of ground-truth surfaces on DTU, with controls on the same pixels under clean input. We report AUROC, CUR, and CUR-err for TBD(P2) models and three stress tests (view dropping, low parallax, distractor images).
2. **A method that separates observed from imagined geometry.** A frozen backbone plus frozen language-aligned ViT features feed a uniform-state discrete diffusion head over FSQ-quantized geometry tokens, trained self-supervised by masking regions and predicting their tokens. At test time K samples give geometry, a three-way label, and an uncertainty from sample disagreement.
3. **Continual adaptation.** LoRA on the diffusion head with a small replay buffer adapts to a stream of new sites; we report new-site accuracy and old-site forgetting against naive fine-tuning.

The headline claim we test, and do not assume, is: native confidence fails to flag unsupported geometry across the evaluated models, and K-sample disagreement from the discrete diffusion head flags it substantially better at equal compute. Whether this claim holds is decided by Gate 1 and Gate 2 in `EXPERIMENTS.md`.

## 2. Related work

Based on the P0 novelty search of 2026-10-28 (details in `LOG.md`). Re-run the search just before submission; uncertainty papers for feed-forward 3D have appeared roughly monthly since May 2026.

**Feed-forward 3D reconstruction.** DUSt3R and MASt3R predict pointmaps for image pairs. VGGT, Pi3 and Pi3X, MapAnything (arXiv 2509.13414), VGGT-Omega (arXiv 2605.15195) and Depth Anything 3 extend this to many unposed views in one pass. Each exposes a per-pixel confidence trained as a loss weight on observed data; none is trained to distinguish observed from unobserved surfaces. Pi3X replaces Pi3's near-binary confidence with continuous quality levels, so confidence semantics differ even within one family. MapAnything additionally offers a multi-view consistency confidence, which we include as a baseline.

**Hallucination in feed-forward 3D.** Bratulic et al. (arXiv 2512.11508) probe DUSt3R, VGGT and DA3 and show that attention heads keep matching correspondences after small patches are occluded, concluding that these models hallucinate missing correspondences from learned priors; they measure attention and Sampson error on ShapeNet pairs, not confidence or 3D labels. Paul et al. (arXiv 2605.18754) benchmark hallucination at the input level: unrelated scenes, repeated images and noise produce dense geometry, scored by COLMAP-based consistency. 3D-Mirage (arXiv 2512.15423) studies monocular depth under visual illusions. Our definition is per point, inside a real scene, with exact supported and unsupported labels, and it tests whether native confidence separates them.

**Uncertainty for feed-forward 3D.** ReVar3R (arXiv 2610.07883) aligns perturbed runs to a common frame and takes per-point variance; it beats native confidence on AUSE in most conditions but, by its authors' account, misses stable systematic bias, which is exactly what a confidently filled region is. Trust3R (arXiv 2605.19539) fits an evidential head on a frozen backbone. GeoCond (arXiv 2609.18465) adds pose-level uncertainty and a refinement gate. Deng et al. (arXiv 2608.29705) audit calibration of seven backbones and find implied uncertainty at least 2.4x too small. All of these score uncertainty against ground-truth error; they conflate confident errors on observed surfaces with confident predictions on surfaces no view saw. Our Confident-Unsupported Rate separates the second case. Free Geometry (arXiv 2604.14048) and TTT3R (arXiv 2509.26645) adapt at test time, Free Geometry with LoRA and hidden frames, but neither labels unsupported regions.

**Generative and multi-sample 3D.** Z3D (arXiv 2609.04174) decodes hidden surfaces from frozen VGGT features with linear probes and diffusion, evidence that unobserved geometry is present in the features. VolFill (arXiv 2605.31466) reports sample variation about 1.3x larger in occluded regions for single-view amodal reconstruction, the same signal we use in the multi-view setting. Spherical Occupancy Profiles (arXiv 2608.23206) add a generative branch to a feed-forward model. Diffusion depth ensembles (Marigold, Perfecting Depth, arXiv 2506.04612) give sample-disagreement uncertainty but require aligning samples for scale and shift before taking variance; we do the same. URSA (arXiv 2510.24717) gives the uniform-state discrete diffusion sampler we build on; we found no prior use of it for 3D.

**Hallucination in vision-language models.** Object hallucination benchmarks (POPE and successors) label statements unsupported by the image. Ours is the geometric analogue.

## 3. Defining geometric hallucination

Given input views I_1..I_N and a predicted depth map D_i with confidence C_i for view i, a pixel p in view i is **supported** if its ground-truth surface point is visible in at least one input view, and **unsupported** if it is visible in none. A model hallucinates at p if it outputs geometry at p with high confidence while p is unsupported.

Why confidence is the wrong tool. Native confidence is trained as a per-pixel loss weight on observed data. It learns to be low where the regression target is noisy (edges, specular surfaces) and high where it is smooth. An unobserved region that the prior fills smoothly therefore receives high confidence by construction. Section 4 measures how large this effect is.

Metrics, computed per (model, scan, seed, radius, fill):

- **AUROC**: P(C_S > C_U), with ties counted half. 1.0 means the model ranks every supported pixel above every unsupported one. 0.5 means confidence carries no information about support.
- **CUR**: choose tau so that 90% of supported pixels have C >= tau; CUR is the fraction of unsupported pixels with C >= tau. This is the hallucination rate a user would experience after confidence filtering.
- **CUR-err**: fraction of unsupported pixels above tau whose AbsRel exceeds 0.1. Confident and wrong.
- **AbsRel_U, delta<1.25_U**: depth error on unsupported pixels after per-sequence median scaling fitted on supported pixels only.
- **Controls**: the same pixels under clean input (AbsRel_U_ctrl, AUROC_ctrl, confidence ratio).

## 4. HalluBench-3D

**Protocol.** For each DTU scan we take 10 views spread evenly over the 49-camera rig, at 420x518 model input. A sphere is centred on a random ground-truth surface point in the reference view, with radius 0.10 or 0.18 of the object bounding-box diagonal, resampled until it covers 2 to 35% of the reference view. Every pixel in every view whose ground-truth point lies inside the sphere is hidden. Hidden pixels are filled with gray, a blur of the original, or uniform noise; fill type is an ablation so that results are not an artefact of a gray patch. Labels: U = hidden in all views, S = valid and outside a 6-pixel dilation of the hidden region, ring = ignored. Five mask seeds per scan. Masks are deterministic in (scan, seed, radius).

**Correctness guards.** Before any model is evaluated we check (a) ground-truth depth reprojection agreement between view pairs and (b) that unsupported points from view i, when visible in view j, land in view j's hidden mask. On the synthetic sanity scene these are 0.952 and 0.996. On real DTU (scans 3 and 5): reprojection agreement 0.796 and 0.824, U-transfer 0.961 to 0.991 (`results/P1-check/`). The 1% depth tolerance is strict for real DTU ground truth, which has inter-view occlusion and sensor noise; a camera error would give agreement near zero. We keep the tolerance and pass the guard.

**Pipeline validation (synthetic, real numbers).** On a synthetic DTU-format scene with two fake models, from `results/synthetic/synthetic_results.csv`:

| model | fill | AUROC | CUR | CUR-err | AbsRel_U | AbsRel_U ctrl |
|---|---|---|---|---|---|---|
| oracle (knows what is hidden) | gray | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| oracle | noise | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| blind (guesses, flat confidence) | gray | 0.500 | 0.902 | 0.893 | 0.149 | 0.000 |
| blind | noise | 0.504 | 0.898 | 0.890 | 0.149 | 0.000 |

The metrics behave as intended: an oracle scores AUROC 1 / CUR 0 and a blind guesser scores AUROC 0.5 / CUR 0.9 (the 10% of supported pixels dropped by tau is the expected floor). These numbers validate the code, not any model.

**Models.** VGGT-1B (depth, depth_conf), Pi3X and Pi3 (local point z, sigmoid of conf), then MapAnything, VGGT-Omega, Depth Anything 3 after their APIs are verified from source. Same views, resolution, and fp16 budget for all.

**Main benchmark table (Table 1).** 18 DTU validation scans x 5 seeds x 2 radii x 3 fills per model. Mean with 95% cluster bootstrap CI over scans. Full configuration: TBD(P1-full, running on Kaggle 2026-10-28).

**Table 1a, smoke configuration (real, runs P1-smoke, P2-smoke, P2-smoke-b, 2026-10-28, Tesla T4, fp16).** 18 scans, 1 seed, radius 0.15, gray fill, 10 views at 420x518. Mean [95% cluster bootstrap CI over scans]. Sources: `results/P1-smoke/`, `results/P2-smoke/`, `results/P2-smoke-b/`.

| model | confidence | AUROC | AUROC ctrl | CUR | CUR-err | AbsRel_S | AbsRel_U | AbsRel_U ctrl | conf ratio U |
|---|---|---|---|---|---|---|---|---|---|
| VGGT-1B | depth_conf | 0.991 [0.986, 0.995] | 0.411 | 0.019 [0.007, 0.036] | 0.000 | 0.007 | 0.021 | 0.005 | 0.011 |
| Pi3 | sigmoid(conf) | 0.984 [0.973, 0.993] | 0.528 | 0.046 [0.015, 0.084] | 0.000 | 0.016 | 0.022 | 0.014 | 0.024 |
| Pi3X | sigmoid(conf) | 0.940 [0.902, 0.971] | 0.477 | 0.207 [0.089, 0.343] | 0.001 | 0.019 | 0.024 | 0.016 | 0.272 |
| MapAnything | conf (learned) | 0.880 [0.824, 0.930] | 0.411 | 0.329 [0.202, 0.459] | 0.205 [0.111, 0.302] | 0.103 | 0.260 | 0.124 | 0.594 |
| MapAnything | multi-view conf | 0.875 [0.824, 0.920] | 0.441 | 0.514 [0.310, 0.717] | 0.330 [0.181, 0.489] | 0.103 | 0.260 | 0.124 | 0.048 |
| Depth Anything 3 (Large 1.1) | conf | 0.675 [0.581, 0.761] | 0.450 | 0.772 [0.651, 0.877] | 0.002 | 0.016 | 0.017 | 0.012 | 0.747 |

All AUROC values differ from 0.5 (sign-flip p <= 0.0026). Paired sign-flip tests over the 18 scans: Pi3X vs VGGT, AUROC p = 0.0045, CUR p = 0.0068; DA3 vs MapAnything, AUROC p = 0.00025, CUR p = 0.00005.

Reading of Table 1a. The five models span the full range. VGGT and Pi3 already separate supported from unsupported pixels almost perfectly on a gray fill and let 2 to 5% of unsupported pixels through. Pi3X, which replaced Pi3's near-binary confidence with continuous quality levels, lets 21% through. Depth Anything 3 is close to blind: AUROC 0.675, with 77% of unsupported pixels above the threshold and a median confidence inside the hidden region at 75% of its clean-input value. Its depth there is still accurate (AbsRel_U 0.017), so DA3 imagines well and reports high confidence for it: a confident hallucination by our definition, even though it is a correct one on DTU. MapAnything is the only model that is confident and wrong (CUR-err 0.21 to 0.33), and swapping its learned confidence for its multi-view consistency confidence makes leakage worse (CUR 0.51), which is a first negative result for a cheap training-free detector.

Caveats on Table 1a. (1) MapAnything's AbsRel on supported pixels is 0.10, ten times VGGT's, and its control-input error on the hidden region is 0.12; at its native 392x518 input the numbers are unchanged (AbsRel_S 0.095, AUROC 0.875, CUR 0.343; `results/P2-mapanything-res/`), so the error is the model's behaviour on DTU in image-only metric mode, not an input-size artefact, and the hallucination metrics are stable across input size. (2) A gray patch is a visible textureless cue, so these are lower bounds on hallucination; Table 1b with blur and noise fills is the headline. (3) VGGT-Omega is gated and not yet run.


**Table 1, full configuration, five models (real; runs P1-full and P2-full, 2026-10-28, Tesla T4, fp16).** 18 scans x 5 mask seeds x 2 radii (0.10, 0.18) x 3 fills = 540 masked forwards per model, pooled over radii. Mean [95% cluster bootstrap CI over 18 scans]. Merged CSV: `results/table1_all_models.csv`; per-cell table: `results/table1_by_model_fill.csv`.

| model | fill | AUROC | CUR | CUR-err | AbsRel_U | AbsRel_U ctrl | conf ratio U |
|---|---|---|---|---|---|---|---|
| VGGT-1B | gray | 0.973 [0.955, 0.987] | 0.067 [0.031, 0.111] | 0.000 | 0.018 | 0.006 | 0.051 |
| VGGT-1B | blur | 0.886 [0.843, 0.922] | 0.263 [0.203, 0.324] | 0.000 | 0.011 | 0.006 | 0.241 |
| VGGT-1B | noise | 0.919 [0.892, 0.945] | 0.222 [0.145, 0.304] | 0.000 | 0.014 | 0.006 | 0.192 |
| Pi3 | gray | 0.981 [0.972, 0.988] | 0.053 [0.025, 0.085] | 0.001 | 0.020 | 0.014 | 0.057 |
| Pi3 | blur | 0.923 [0.895, 0.947] | 0.236 [0.156, 0.326] | 0.001 | 0.018 | 0.014 | 0.261 |
| Pi3 | noise | 0.902 [0.865, 0.934] | 0.325 [0.219, 0.438] | 0.002 | 0.020 | 0.014 | 0.411 |
| Pi3X | gray | 0.940 [0.902, 0.971] | 0.202 [0.085, 0.337] | 0.002 | 0.022 | 0.017 | 0.335 |
| Pi3X | blur | 0.826 [0.782, 0.865] | 0.512 [0.397, 0.630] | 0.002 | 0.020 | 0.017 | 0.739 |
| Pi3X | noise | 0.801 [0.734, 0.864] | 0.551 [0.390, 0.711] | 0.004 | 0.022 | 0.017 | 1.091 |
| MapAnything | gray | 0.861 [0.812, 0.905] | 0.374 [0.272, 0.480] | 0.195 [0.125, 0.269] | 0.222 | 0.111 | 0.650 |
| MapAnything | blur | 0.773 [0.720, 0.825] | 0.580 [0.482, 0.672] | 0.277 [0.197, 0.361] | 0.135 | 0.111 | 0.677 |
| MapAnything | noise | 0.712 [0.645, 0.773] | 0.664 [0.566, 0.758] | 0.295 [0.211, 0.390] | 0.129 | 0.111 | 0.709 |
| Depth Anything 3 | gray | 0.723 [0.664, 0.780] | 0.703 [0.596, 0.804] | 0.003 | 0.017 | 0.013 | 0.804 |
| Depth Anything 3 | blur | 0.682 [0.611, 0.744] | 0.711 [0.619, 0.796] | 0.003 | 0.016 | 0.013 | 0.800 |
| Depth Anything 3 | noise | 0.740 [0.686, 0.793] | 0.690 [0.578, 0.793] | 0.001 | 0.017 | 0.013 | 0.855 |

Headline numbers, pooled over the two natural-statistics fills (blur and noise):

| model | AUROC | CUR | CUR-err |
|---|---|---|---|
| VGGT-1B | 0.903 [0.873, 0.929] | 0.242 [0.181, 0.304] | 0.000 |
| Pi3 | 0.912 [0.881, 0.940] | 0.281 [0.189, 0.379] | 0.002 |
| Pi3X | 0.813 [0.762, 0.860] | 0.532 [0.403, 0.663] | 0.003 |
| MapAnything | 0.743 [0.687, 0.795] | 0.622 [0.534, 0.707] | 0.286 [0.206, 0.370] |
| Depth Anything 3 | 0.711 [0.653, 0.763] | 0.701 [0.614, 0.784] | 0.002 |

Pairwise CUR differences on blur and noise, paired sign-flip over 18 scans with Holm correction over the 10 pairs (adjusted p): VGGT and Pi3 are indistinguishable (0.41); both leak less than Pi3X (0.0015 and 0.0005), MapAnything (0.0005) and DA3 (0.0005). Pi3X vs DA3 (0.052), Pi3X vs MapAnything (0.19) and MapAnything vs DA3 (0.19) are not separated after correction. Within every model, blur and noise raise CUR over gray (for VGGT and Pi3X all eight tests p = 0.00005; DA3 is the exception: its CUR is 0.70 on every fill, i.e. it does not even react to the gray patch).

Reading of Table 1. Three tiers emerge. VGGT and Pi3 drop confidence inside the hidden region (median confidence at 19 to 41% of its clean value) but still let one in four unsupported pixels through a threshold that keeps 90% of supported ones. Pi3X, which replaced Pi3's near-binary confidence with continuous quality levels, leaks one in two, and under a noise fill is as confident about geometry it cannot see as about geometry it can (ratio 1.09). MapAnything and Depth Anything 3 are close to blind to non-observation (AUROC 0.71 to 0.74, CUR 0.62 to 0.70). Confidence inside the hidden region in DA3 stays at 80 to 86% of its clean value for every fill, so DA3's confidence encodes something other than observation.

Confident and wrong. On DTU the imagined depth is accurate for four of the five models (AbsRel_U 0.011 to 0.022, about 1.3 to 2x the clean value), so CUR-err stays near zero: the failure is that the user cannot tell imagined from observed, not that the geometry is wrong. MapAnything is the exception, with 29% of unsupported pixels both confident and wrong by more than 10% AbsRel, on top of a 10% error on supported pixels in image-only metric mode (also seen at its native input size, `results/P2-mapanything-res/`). CUR and CUR-err are expected to separate for the other models on scenes with less predictable hidden structure (CO3Dv2, heritage), which is why both are reported.

Radius has a small effect (per-radius cells in `results/*/P*-full_summary.csv`; VGGT CUR under blur is 0.261 at r = 0.10 and 0.264 at r = 0.18).

**Gate 1 verdict (2026-10-28): PASS.** CUR is above zero with CIs excluding zero for all five models on every fill, and AUROC is well below 1 on the natural-statistics fills for all five. The effect is not an artefact of a gray patch, and it is strongly model dependent.

**Stress tests (Table 2, real; run P2-stress, 2026-10-28, 3157 s on T4).** 18 scans x 2 mask seeds, radius 0.18. Reference rows are the matching cells (blur fill, 10 spread views, seeds 0 and 1) of Table 1. Mean [95% CI over scans]; the last column is the paired CUR difference against the reference with a sign-flip p over scans. Source: `results/P2-stress/stress_vs_reference.csv`.

| model | test | AUROC | CUR | CUR-err | AbsRel_U | CUR vs ref |
|---|---|---|---|---|---|---|
| VGGT-1B | reference | 0.902 [0.868, 0.930] | 0.259 [0.175, 0.349] | 0.000 | 0.010 | |
| VGGT-1B | inpaint fill | 0.867 [0.821, 0.907] | 0.347 [0.255, 0.443] | 0.001 | 0.018 | +0.088 (p = 0.031) |
| VGGT-1B | 5 views | 0.902 | 0.269 [0.173, 0.377] | 0.000 | 0.011 | +0.010 (p = 0.63) |
| VGGT-1B | 3 views | 0.899 | 0.259 [0.167, 0.361] | 0.000 | 0.013 | +0.001 (p = 0.98) |
| VGGT-1B | low parallax (adjacent 10) | 0.911 | 0.214 [0.110, 0.334] | 0.000 | 0.021 | -0.045 (p = 0.35) |
| VGGT-1B | + distractor image | 0.905 | 0.257 [0.179, 0.340] | 0.000 | 0.011 | -0.002 (p = 0.90) |
| Pi3X | reference | 0.828 [0.781, 0.870] | 0.521 [0.390, 0.650] | 0.002 | 0.024 | |
| Pi3X | inpaint fill | 0.786 [0.731, 0.836] | 0.589 [0.473, 0.702] | 0.004 | 0.022 | +0.068 (p = 0.005) |
| Pi3X | 5 views | 0.822 | 0.539 [0.411, 0.661] | 0.006 | 0.024 | +0.018 (p = 0.44) |
| Pi3X | 3 views | 0.840 | 0.497 [0.373, 0.618] | 0.032 | 0.051 | -0.024 (p = 0.64) |
| Pi3X | low parallax | 0.812 | 0.523 [0.376, 0.671] | 0.040 | 0.050 | +0.002 (p = 0.98) |
| Pi3X | + distractor | 0.829 | 0.510 [0.368, 0.650] | 0.002 | 0.024 | -0.011 (p = 0.66) |
| DA3 | reference | 0.712 [0.632, 0.783] | 0.679 [0.569, 0.791] | 0.006 | 0.018 | |
| DA3 | inpaint fill | 0.726 | 0.688 [0.605, 0.774] | 0.009 | 0.018 | +0.008 (p = 0.87) |
| DA3 | 5 views | 0.697 | 0.785 [0.703, 0.865] | 0.005 | 0.019 | +0.106 (p = 0.008) |
| DA3 | 3 views | 0.729 | 0.667 [0.560, 0.770] | 0.018 | 0.031 | -0.012 (p = 0.81) |
| DA3 | low parallax | 0.863 [0.785, 0.925] | 0.372 [0.230, 0.520] | 0.008 | 0.029 | -0.307 (p = 0.0003) |
| DA3 | + distractor | 0.660 [0.580, 0.734] | 0.775 [0.680, 0.868] | 0.003 | 0.019 | +0.096 (p = 0.0004) |

Reading of Table 2. (1) Fill realism: the inpainted fill, which is edge-continuous and has no texture discontinuity, raises leakage further for VGGT and Pi3X, so Table 1 is a lower bound on what a natural-looking occluder produces, and the headline is not an artefact of a visible patch. (2) Evidence quantity: dropping from 10 to 3 views leaves CUR unchanged for all three models; the leakage is not reduced by having fewer views to cross-check, which indicates that native confidence is driven by single-view appearance rather than multi-view agreement. (3) Geometry quality under stress: with 3 views or low parallax, Pi3X's imagined depth degrades (AbsRel_U 0.05, CUR-err up to 0.04) while its confidence does not, so the confident-and-wrong fraction becomes visible on DTU under harder capture conditions. (4) DA3 behaves differently: a distractor image raises its leakage and low-parallax input halves it, which is consistent with its confidence tracking something like inter-view consistency of appearance rather than observation of the surface; this is an observation, not an explanation, and is flagged for the discussion.



**CO3Dv2 subset (Table 3).** 18 scenes from the CO3Dv2 single-sequence subset: the two depth-carrying sequences of each of nine categories (apple, ball, bench, hydrant, mouse, teddybear, toytrain, toytruck, vase), with the official COLMAP depth as ground truth. Views are 10 frames spread over the full orbit, so the hidden region is seen from the back as well as the front. Cameras are converted from the PyTorch3D convention and were checked by projecting the official point cloud (median depth error 0.7 to 2.5% per view) and by an occlusion-aware reprojection test (0.92 to 1.00 agreement, the same test gives 0.99 on DTU). Images are 640x360 phone frames; models run at 518x294 instead of 518x420. Everything else, including the masking protocol and all thresholds, is unchanged from the DTU freeze. Scenes are listed in `results/P2-co3d/P2-co3d_consistency.json`.

**Table 3, CO3Dv2 subset, five models (real; runs P2-co3d and P2-co3d-b, 2026-10-28, Tesla T4, fp16, 1832 s + 5967 s).** 18 scenes x 5 mask seeds x 2 radii (0.10, 0.18) x 3 fills = 540 masked forwards per model, pooled over radii. Mean [95% cluster bootstrap CI over 18 scenes]. Merged CSV: `results/table3_all_models.csv`; per-cell table: `results/table3_by_model_fill.csv`.

| model | fill | AUROC | CUR | CUR-err | AbsRel_S | AbsRel_U | AbsRel_U ctrl | conf ratio U |
|---|---|---|---|---|---|---|---|---|
| VGGT-1B | gray | 0.747 [0.692, 0.802] | 0.565 [0.453, 0.677] | 0.029 | 0.029 | 0.031 | 0.027 | 0.634 |
| VGGT-1B | blur | 0.772 [0.717, 0.825] | 0.534 [0.416, 0.652] | 0.034 | 0.029 | 0.031 | 0.027 | 0.603 |
| VGGT-1B | noise | 0.590 [0.517, 0.656] | 0.767 [0.693, 0.839] | 0.029 | 0.029 | 0.029 | 0.027 | 0.805 |
| Pi3 | gray | 0.930 [0.894, 0.958] | 0.196 [0.118, 0.283] | 0.002 | 0.015 | 0.016 | 0.013 | 0.351 |
| Pi3 | blur | 0.903 [0.860, 0.939] | 0.251 [0.170, 0.334] | 0.002 | 0.015 | 0.016 | 0.013 | 0.389 |
| Pi3 | noise | 0.880 [0.813, 0.929] | 0.293 [0.193, 0.405] | 0.002 | 0.015 | 0.015 | 0.013 | 0.472 |
| Pi3X | gray | 0.863 [0.825, 0.899] | 0.329 [0.248, 0.408] | 0.001 | 0.017 | 0.022 | 0.014 | 0.749 |
| Pi3X | blur | 0.720 [0.661, 0.777] | 0.584 [0.500, 0.668] | 0.002 | 0.017 | 0.018 | 0.014 | 0.877 |
| Pi3X | noise | 0.743 [0.680, 0.798] | 0.556 [0.462, 0.650] | 0.001 | 0.017 | 0.019 | 0.014 | 0.867 |
| MapAnything | gray | 0.609 [0.554, 0.665] | 0.786 [0.721, 0.849] | 0.004 | 0.041 | 0.029 | 0.034 | 0.632 |
| MapAnything | blur | 0.575 [0.514, 0.637] | 0.801 [0.735, 0.863] | 0.008 | 0.042 | 0.028 | 0.034 | 0.674 |
| MapAnything | noise | 0.578 [0.524, 0.631] | 0.819 [0.765, 0.870] | 0.007 | 0.041 | 0.033 | 0.034 | 0.690 |
| Depth Anything 3 | gray | 0.619 [0.560, 0.674] | 0.816 [0.751, 0.877] | 0.001 | 0.019 | 0.015 | 0.014 | 0.874 |
| Depth Anything 3 | blur | 0.622 [0.574, 0.668] | 0.813 [0.748, 0.872] | 0.001 | 0.018 | 0.015 | 0.014 | 0.853 |
| Depth Anything 3 | noise | 0.618 [0.553, 0.679] | 0.780 [0.702, 0.854] | 0.000 | 0.019 | 0.016 | 0.014 | 0.901 |

Headline numbers, pooled over blur and noise, DTU (Table 1) beside CO3Dv2:

| model | AUROC DTU | AUROC CO3D | CUR DTU | CUR CO3D |
|---|---|---|---|---|
| VGGT-1B | 0.903 [0.873, 0.929] | 0.681 [0.632, 0.729] | 0.242 [0.181, 0.304] | 0.651 [0.566, 0.734] |
| Pi3 | 0.912 [0.881, 0.940] | 0.891 [0.845, 0.930] | 0.281 [0.189, 0.379] | 0.272 [0.190, 0.360] |
| Pi3X | 0.813 [0.762, 0.860] | 0.732 [0.680, 0.779] | 0.532 [0.403, 0.663] | 0.570 [0.500, 0.641] |
| MapAnything | 0.743 [0.687, 0.795] | 0.576 [0.521, 0.632] | 0.622 [0.534, 0.707] | 0.810 [0.756, 0.863] |
| Depth Anything 3 | 0.711 [0.653, 0.763] | 0.620 [0.567, 0.670] | 0.701 [0.614, 0.784] | 0.797 [0.728, 0.861] |

Reading of Table 3. The ranking changes. Pi3 is the only model whose behaviour transfers from the lab rig to phone captures (CUR 0.27 on both). VGGT, the best model on DTU together with Pi3, leaks 65% of unsupported pixels on CO3D, 2.7x its DTU value, and under a noise fill its confidence is almost blind to non-observation (AUROC 0.59). MapAnything and DA3 let about four in five unsupported pixels through. Pairwise CUR on blur and noise, paired sign-flip over 18 scenes with Holm correction over 10 pairs: Pi3 leaks less than every other model (adjusted p 0.0005 each); Pi3X leaks less than MapAnything and DA3 (0.0005); VGGT leaks less than MapAnything (0.010) and DA3 (0.0005); VGGT vs Pi3X (0.23) and MapAnything vs DA3 (0.73) are not separated. Compared with DTU, the per-model CIs of VGGT, MapAnything and DA3 do not overlap across datasets, those of Pi3 and Pi3X do.

Imagined geometry is accurate here too. AbsRel_U is within 0.005 of AbsRel_S for every model (0.015 to 0.033), so CUR-err stays below 0.04 and MapAnything's confident-and-wrong behaviour from DTU does not appear (AbsRel_S 0.041, CUR-err 0.007). The objects are everyday things with strong shape priors (apples, balls, mugs), and the models complete them well; the failure is again that the completion is reported with the confidence of an observation. The expected separation between CUR and CUR-err on less predictable scenes has not materialised on CO3D; it remains a prediction for the heritage data.

Fill effects differ from DTU. On DTU every fill raised CUR over gray for VGGT, Pi3 and Pi3X. On CO3D the noise fill raises VGGT's CUR by 0.20 (p 0.00005) but blur does not (p 0.30); Pi3X reacts to both (+0.26, +0.23, p 0.00005); Pi3 reacts weakly (+0.06, +0.10, p 0.007 and 0.017); MapAnything and DA3 do not react to the fill at all (p 0.06 to 0.91), consistent with their confidence not encoding observation.

Notes on Table 3. (1) One scene, vase_374_41862_83720, has U_transfer 0.80 at seed 0 against the 0.85 gate (small sphere near the silhouette). It is kept; removing it moves every pooled number by at most 0.013 and changes no conclusion. (2) The hidden region covers 2 to 27% of the object's valid pixels in the reference view, versus a tighter range on DTU, because CO3D objects occupy a small part of the frame. (3) CO3D depth is COLMAP MVS at 1 to 3% noise, so AbsRel values below about 0.03 should be read as equal.

## 5. Method

**Architecture.** Frozen 3D backbone (VGGT or Pi3X) provides per-pixel geometry and tokens. Frozen SigLIP2 ViT provides language-aligned features at the same grid. A finite scalar quantizer (FSQ) turns backbone depth and local point outputs into discrete tokens. A discrete diffusion head of about 100M parameters, with uniform-state corruption, denoises token grids conditioned on the backbone and SigLIP2 features.

**Training.** Self-supervised: mask random 3D-consistent regions of the clean backbone output, corrupt the tokens, and train the head to recover them. No hallucination labels are used in training. Hyperparameters are tuned on the DTU validation split only and frozen before CO3Dv2 or heritage data are touched (freeze date: TBD, logged in `LOG.md`).

**Inference.** Draw K samples. Per pixel: geometry = per-sample median; disagreement = token entropy across samples (or depth spread); label = observed if the backbone and all samples agree within a tolerance, imagined if samples agree with each other but not with any observation, abstain otherwise.

**Compute.** K samples cost K head passes on top of one backbone pass. We report wall-clock and FLOPs and compare baselines at equal compute (for example, perturbation variance with K backbone passes). TBD(P3-compute).

## 6. Experiments

**Training-free baselines.** Native confidence; MapAnything multi-view consistency confidence; reprojection consistency across views; aligned perturbation variance (ReVar3R); MC over input augmentations; Trust3R evidential head where its code permits. All at equal compute.

**Main method table (Table 4).** On HalluBench-3D DTU, per detector: AUROC, CUR, CUR-err, wall-clock, FLOPs. At least 3 training seeds, preferably 5. Paired sign-flip permutation test on per-scan means against native confidence, Holm-corrected over pairs. Gate 2 reads this table.

Rows available so far (real, run P3-pv-smoke, 2026-10-28; 18 scans, seed 0, blur fill, radius 0.18; native rows from P1-full with identical keys; `results/P3-pv-smoke/`):

| backbone | detector | forwards | AUROC | CUR | CUR-err |
|---|---|---|---|---|---|
| VGGT-1B | native confidence | 1 | 0.919 [0.890, 0.946] | 0.241 [0.148, 0.343] | 0.000 |
| VGGT-1B | perturbation variance, K=4 (ReVar3R style, photometric) | 4 | 0.655 [0.586, 0.724] | 0.647 [0.536, 0.752] | 0.001 |
| VGGT-1B | reprojection consistency | 1 | TBD(P3-reproj) | | |
| VGGT-1B | K-sample disagreement, discrete diffusion head | 1 + K head passes | TBD(P3-main) | | |
| Pi3X | native confidence | 1 | 0.832 [0.782, 0.879] | 0.502 [0.362, 0.642] | 0.003 |
| Pi3X | perturbation variance, K=4 | 4 | 0.551 [0.492, 0.609] | 0.824 [0.753, 0.886] | 0.003 |
| Pi3X | K-sample disagreement, discrete diffusion head | 1 + K head passes | TBD(P3-main) | | |

Perturbation variance is worse than native confidence on both backbones (paired sign-flip over scans, AUROC and CUR, p <= 0.00065 in all four tests) while costing four forwards. The imagined geometry is stable under input perturbation, so sample variance is nearly the same on supported and unsupported pixels. This is the failure mode ReVar3R's authors describe as stable systematic bias, and it is why we draw disagreement from a generative head rather than from the input. A fuller perturbation baseline (geometric augmentations, K=8, similarity alignment) is an ablation, TBD(P3-pv-full).

**Ablations (Table 5).** K in {1, 2, 4, 8, 16}; fill type; radius; backbone (VGGT vs Pi3X); with vs without SigLIP2 features. TBD(P3-abl).

**Continual adaptation (Table 6).** Stream of sites; new-site AUROC/CUR and old-site forgetting for LoRA + replay vs naive fine-tuning vs frozen. TBD(P4-cont).

**Heritage stress test (Figure 5).** Public photo collections of monuments; qualitative observed / imagined / abstain maps and a language-query figure ("show only observed parts of the dome"). Not a main claim. TBD(P4-heritage).

## 7. Limitations

Static scenes only. DTU is small-object, lab-captured; the benchmark labels are a proxy for real occlusion, since real unobserved regions are not surrounded by a filled patch. The method costs K extra head passes at test time. Continual adaptation is evaluated on a short stream. All results depend on the frozen backbones' behaviour and may change with newer checkpoints.

## Disclosure

Before submission: check the CVPR 2027 author guidelines on AI-assisted research and writing and disclose as required. Keep the submission double-blind.
