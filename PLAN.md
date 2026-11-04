# Project brief: Observe, Imagine, Abstain

CVPR 2027 submission on geometric hallucination in feed-forward 3D reconstruction. This file is the brief the work follows: the idea, the rules, the deadlines, the phases and their gates, and the statistics protocol. The run registry is `EXPERIMENTS.md`, the dated log is `LOG.md`.

## 1. Rules

1. **No fabricated numbers.** Every number in a table, figure, abstract or claim comes from a logged run whose CSV and config exist under `results/`. A result that does not exist yet is written as `pending` with the run id that produces it.
2. **Report every run**, including failures. Nothing is dropped silently; an exclusion is stated in the paper.
3. **Hyperparameters are tuned on the development split only** (DTU validation scans), then frozen before CO3Dv2 or any other dataset is touched. The freeze date is logged.
4. **Fair baselines**: same views, resolution, data and compute budget. Report forwards per detector; compare at equal compute where a method uses extra samples.
5. **Gates are real.** The outcome of each gate is reported as measured. A failed gate leads to a pivot, not to a reframing.
6. **No overclaiming.** "Built on 3D foundation models", never "our foundation model". Static scenes, not "world models".
7. **Review integrity.** Double-blind: no author identity in the submission, the supplementary or the videos. No contact with organisers, area chairs or reviewers about this paper.
8. **Plain writing.** Concrete, concise, no hype words.
9. **LLM use is declared** in the LLM Usage Statement that CVPR 2027 requires, as a section after the main text; the authors own every decision, number and claim.

## 2. Deadlines (verify on cvpr.thecvf.com)

- Paper registration: Nov 10, 2026 AoE (title, abstract, authors locked)
- Paper submission: Nov 16, 2026 AoE
- Supplementary: Nov 23, 2026 AoE

## 3. The idea

**Problem.** Feed-forward 3D models (VGGT, Pi3/Pi3X, MapAnything, Depth Anything 3, VGGT-Omega) reconstruct a scene from unposed photos in one pass. Where no view shows a surface they fill it from learned priors and often assign the invented geometry a confidence like that of observed geometry. A user cannot tell observed from imagined.

**Definition.** A predicted point is hallucinated when it lies on a surface that no input view shows. Labels are free by construction: hide a 3D region of the ground-truth surface in every view (3D-consistent masking), fill the hole, run the model, score its confidence on hidden (unsupported) against visible (supported) pixels.

**Metrics.** AUROC (does confidence separate supported from unsupported), CUR (share of unsupported pixels that pass the threshold keeping 90% of supported pixels), CUR-err (passing and wrong by more than 10% depth), depth error on both sets, and clean-input controls on the same pixels.

**Detector.** Do not ban imagination; make it explicit. A small support head on the frozen multi-view features of the backbone predicts, per pixel, whether the output is observed or imagined, with an abstain band. It is trained without labels by applying the benchmark's masking to the backbone's own clean prediction on training scenes disjoint from the benchmark. Controls: a head on per-image features only (can only see the patch), a linear probe, a cross-view transformer head, the inpainted fill the head never saw, CO3Dv2 which the head never saw, ground-truth-placed masks, extra initialisation seeds, fusion with native confidence, the reprojection-consistency detector and the perturbation-variance family at equal compute.

**Where Gaussian splatting and transformers enter.** The benchmark is representation-agnostic: a feed-forward Gaussian-splatting backbone (pixel-aligned Gaussians with opacities, AnySplat) is scored like a pointmap transformer, with opacity as its confidence; a rendered-alpha variant tests whether the renderer's own visibility helps. The transformer head tests whether cross-view attention inside the detector adds anything beyond features that already passed through the backbone's global attention. Dynamic (4D) scenes are out of scope for this submission and are stated as future work: a temporally unobserved surface is the same question with a time axis.

## 4. Compute

- Benchmark runs: Kaggle T4 (fp16, 10 views, 420x518), about 8 GPU hours in total.
- Method runs: one NVIDIA L4 (24 GB) on AWS, bf16, from Nov 4, 2026. Every stage of `scripts/run_h100.py` fits in 10 GB; the plan was written for an H100 and runs unchanged.
- Data: DTU training set (MVSNet layout, 119 scans: 79 training, 18 validation, 22 test), CO3Dv2 single-sequence subset preprocessed by `scripts/co3d.py`.

## 5. Phases and gates

| Phase | Dates (2026) | Work | Gate |
|---|---|---|---|
| P0 Novelty | Oct 28 | Literature search; compare against ReVar3R, GeoCond, Trust3R, Free Geometry, VGGT-knows. | Done: nobody labels unobserved geometry per point or detects it from features. |
| P1 Benchmark | Oct 28 | DTU, VGGT + Pi3X, 18 scans x 5 seeds x 2 radii x 3 fills. | **Gate 1 passed**: CUR well above 0 with intervals excluding 0, AUROC well below 1. |
| P2 Scale | Oct 28 | Pi3, MapAnything, DA3; stress tests; CO3Dv2. | Done. |
| P3 Method | Nov 1 to 8 | Support head and all controls; reprojection and fuller perturbation baselines; 22 extension scans; Gaussian-splatting backbone. | **Gate 2**: the head beats native confidence on AUROC and CUR, paired sign-flip p < 0.05 over 18 scans, at one backbone forward, and the per-image control does not explain the gain on the inpainted fill or on CO3Dv2. Fail: benchmark paper plus the best training-free detector. |
| Register | Nov 10 | Title and abstract matching the gate outcome. | |
| P5 Write | Nov 8 to 16 | Paper, figures, statistics, limitations. | |
| P6 Supp | Nov 17 to 23 | Anonymised code and results archive, extra tables, videos. | |

Cut in this order if time runs out: the heritage study, continual adaptation, the generative head.

## 6. Statistics protocol

- Unit of analysis: scene. Seeds are mask placements (benchmark) or initialisation seeds (head).
- Means with 95% cluster-bootstrap intervals over scenes (10,000 resamples).
- Paired comparisons: sign-flip permutation test on per-scene means (20,000 permutations), Holm correction within a family of tests.
- Always report the clean-input control on the same pixels. Never report a single best seed.

## 7. Paper structure (8 pages, CVPR template, then the LLM Usage Statement and references)

1. Introduction with one figure: a confident hallucination against the observed / imagined split.
2. Related work: feed-forward 3D, hallucination in feed-forward 3D, uncertainty for 3D, generative and multi-sample 3D, hallucination in VLMs.
3. Definition and metrics. 4. Benchmark protocol, guards, models, statistics. 5. Results: DTU, stress, CO3Dv2, cheap detectors.
6. Method and its controls. 7. Experiments on the method. 8. Limitations. 9. Conclusion.
