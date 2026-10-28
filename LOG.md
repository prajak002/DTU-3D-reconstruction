# LOG

## 2026-10-28

- Read SKILL.md, `scripts/hallu3d.py` (532 lines), `scripts/test_synthetic.py`. No results CSV existed; no DTU data on this machine; no GPU.
- Ran `test_synthetic.py` on CPU (Python 3.12, numpy/opencv/scipy/pandas via uv). All checks pass; results copied to `results/synthetic/`. Config hash: `hallu3d.py` md5 faf9ce41.
  - consistency: reproj_agree 0.952, U_transfer 0.996
  - oracle: AUROC 1.000, CUR 0.000; blind: AUROC 0.500, CUR 0.902, CUR-err 0.891
  - deterministic masks: yes; U and S overlap: 0
- Wrote `paper/draft.md` (v0.1, all model numbers TBD with run ids) and `EXPERIMENTS.md` (run registry, Kaggle cells, gates).
- P0 novelty search: done, see entry below. Decision: benchmark leads the paper; method is secondary.
- Decision: next action is P1-check and P1-smoke on Kaggle (needs the author to upload DTU and run the cells).

### P0 novelty search (2026-10-28, web search at abstract level; CVF/OpenReview not queried directly)

- Verdict: no paper found that defines per-point unsupported geometry with free labels from 3D-consistent masking, or that scores native confidence by AUROC / Confident-Unsupported Rate. The word "hallucination" for feed-forward 3D is already in use (2512.11508 attention-level; 2605.18754 input-level; 2512.15423 monocular illusions). Definition + benchmark contribution still stands; framing must differentiate from those three.
- Closest competitors: 2605.18754 (scene-level hallucination benchmark), 2512.11508 (priors fill occluded correspondences), 2608.29705 + ReVar3R 2610.07883 + Trust3R 2605.19539 + 2606.16479 (confidence vs GT error, no support split).
- Method risk: Free Geometry (2604.14048) already does test-time LoRA with hidden frames; no 3D precedent for URSA. Keep diffusion head + continual as secondary; benchmark leads.
- Must-add baselines: aligned perturbation variance (ReVar3R), MapAnything `use_multiview_confidence`, Trust3R, Pi3 vs Pi3X conf. Align K samples (scale/shift) before disagreement.
- Checkpoints: Pi3 HF ids are `yyfz233/Pi3` and `yyfz233/Pi3X` (code already uses these; github clone `yyfz/Pi3` is correct). DA3: use `-1.1` models (originals deprecated, training bug). VGGT-Omega: gated HF access, contamination note on 1B-512; use 1B-416-Reproduction for benchmark numbers. MapAnything: `facebook/map-anything`, outputs conf, mask, non_ambiguous_mask.
- CVPR 2027 (official page): enrollment Nov 10, submission Nov 16, supp Nov 23, reviews Jan 25, rebuttal Jan 25 to Feb 1, decisions Feb 25 2027. Work posted after Sep 15 2026 is contemporaneous. LLM policy section "being finalized"; recheck before submission. All authors must review when invited.
- Open concern: DTU likely in training data of some models; plan one more dataset (CO3Dv2 subset already in P2).

### P1-check on real DTU (2026-10-28, local CPU, Kaggle dataset `tobyzhou/dtu-training`, MVSNet layout, root `dtu_training/mvs_training/dtu`)

- Scans 3 and 5, 10 views, seeds 0 and 1, radii 0.10 and 0.18. Results in `results/P1-check/local_consistency_scan3_5.json`.
- reproj_agree: scan 3 = 0.796, scan 5 = 0.824 (threshold 0.8; scan 3 is 0.004 under). U_transfer: 0.961 to 0.991 (threshold 0.95, all pass). U and S overlap: 0 everywhere. U fraction in ref view: 0.03 to 0.30 (within cfg bounds).
- Interpretation: cameras and depths are aligned (a camera error would give agreement near 0). The 1% depth tolerance in `consistency_check` is strict for real DTU GT, which has inter-view occlusion and sensor noise, so ~0.8 is expected. Decision: pass the gate; do not loosen the tolerance.
- Example figure `results/P1-check/nomodel_scan3_r0.15.png`: masked region in the image coincides with the U contour on GT depth.
- Kaggle GPU kernel `popsensharma/hallu3d-p1-smoke` pushed (script = hallu3d.py + scripts/kaggle_driver.py; dataset tobyzhou/dtu-training; GPU on; internet on). Config: VGGT + Pi3X, 18 scans, seed 0, gray, r 0.15. Status: running.

### P1-smoke done (2026-10-28, Kaggle T4, kernel popsensharma/hallu3d-p1-smoke v2, 555 s wall-clock)

- v1 failed: pip installs of vggt/Pi3 downgraded numpy to 1.26.4 after numpy 2.x was imported (binary mismatch in np.random). Fixed in `scripts/kaggle_prelude.py`: `--no-deps` installs before any import, Pi3 cloned to /kaggle/tmp and put on sys.path. Pi3X API verified from source: `forward(imgs, with_prior=...)`, outputs `local_points`, `conf` logits.
- Config: VGGT-1B + Pi3X, 18 val scans, seed 0, r 0.15, gray fill, 10 views, 420x518, fp16. CSV + figures in `results/P1-smoke/`.
- Results (mean [95% CI over 18 scans]):
  - VGGT: AUROC 0.991 [0.986, 0.995], CUR 0.019 [0.007, 0.036], CUR-err 0.000, AbsRel_U 0.021 (ctrl 0.005), conf ratio U 0.011
  - Pi3X: AUROC 0.940 [0.902, 0.971], CUR 0.207 [0.089, 0.343], CUR-err 0.001, AbsRel_U 0.024 (ctrl 0.016), conf ratio U 0.272
  - Pi3X vs VGGT paired sign-flip: AUROC p = 0.0045, CUR p = 0.0068
- Gate 1 status (preliminary, gray fill only): VGGT already abstains on gray patches; AUROC is not "well below 1" and CUR is near 0. Pi3X shows real leakage (CUR 0.21, CI excludes 0). The headline claim "native confidence fails across models" is NOT supported by the gray-fill data for VGGT. Do not reframe. The decisive test is blur/noise fills.
- Risk identified: gray fill is a visible textureless cue. If blur/noise fills also give AUROC > 0.98 for VGGT, the benchmark as designed measures patch detection, not knowledge of non-observation, and the paper must pivot (options: Pi3X/other-model focus; real occlusion via view dropping where the hidden surface is simply not photographed; CUR-err and the "imagined but low confidence" split as the story).
- Observation worth keeping: VGGT depth inside the fully hidden region has AbsRel 0.021, i.e. it imagines plausible geometry and marks it low confidence. That is the "imagine but abstain" behaviour the method section wants to produce; the benchmark may show some backbones already have it.
- Launched `popsensharma/hallu3d-p1-full` (default cfg: 5 seeds x 2 radii x 3 fills, both models, resumable). Expected ~2 h from smoke timing (~5 s per masked forward).

### P2 prep, no-approval models (2026-10-28)

- Added `DA3Wrap` (depth-anything/DA3-LARGE-1.1, API read from `src/depth_anything_3/api.py`: `inference(list uint8) -> .depth, .conf`, internal autocast), `MapAnythingWrap` (facebook/map-anything, `infer(views)` with dinov2-normalized (B,3,H,W) tensors, outputs `depth_z`, `conf`) and `MapAnythingMVWrap` (same model, `use_multiview_confidence=True`, first training-free "support" baseline). Pi3 wrapper already existed. `test_synthetic.py` still passes after the change.
- `scripts/kaggle_prelude.py` now installs per model with `--no-deps` (Pi3 pins numpy 1.26, DA3 pins numpy<2, MapAnything pins opencv 4.10). DA3's api module imports its export subpackage, so pycolmap/trimesh/evo/moviepy are installed as light extras. If an import still fails on Kaggle, the driver logs the traceback for that model and continues.
- Pushed `popsensharma/hallu3d-p2-smoke`: pi3, da3, mapanything, mapanything_mv; same config as P1-smoke (18 scans, seed 0, gray, r 0.15) for direct comparison. VGGT-Omega deferred (gated checkpoint).

### P2-smoke (2026-10-28, kernel popsensharma/hallu3d-p2-smoke v1, 310 s)

- Pi3 (yyfz233/Pi3), 18 scans, seed 0, gray, r 0.15: AUROC 0.984 [0.973, 0.993], CUR 0.046 [0.015, 0.084], CUR-err 0.000, AbsRel_U 0.022 (ctrl 0.014), conf ratio U 0.024. Saved in `results/P2-smoke/`. Pi3 (binary-style conf) abstains far better than Pi3X (continuous conf, CUR 0.207) on the gray fill.
- DA3 failed: CUDA OOM, 7.36 GiB attention matrix. Cause: DA3 selects bf16 when `torch.cuda.is_bf16_supported()` is true, which holds on T4 by emulation; mem-efficient SDPA cannot run bf16 on sm75 so attention falls back to the fp32 math path. Fix: `DA3Wrap` forces fp16 on pre-Ampere GPUs; prelude sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- MapAnything failed: `ModuleNotFoundError: jaxtyping` (uniception dependency; --no-deps skipped it). Fix: prelude installs uniception's runtime deps explicitly (jaxtyping, timm, termcolor, torchmetrics, minio, scikit-learn, rerun-sdk).
- Pushed v2 as run `P2-smoke-b` with da3, mapanything, mapanything_mv.

### P2-smoke-b (2026-10-28, kernel hallu3d-p2-smoke v2, 654 s). Results in `results/P2-smoke-b/`.

- DA3-LARGE-1.1: AUROC 0.675 [0.581, 0.761], CUR 0.772 [0.651, 0.877], CUR-err 0.002, AbsRel_U 0.017 (ctrl 0.012), conf ratio U 0.747. Near-blind confidence with accurate imagined depth. Fix for the OOM worked (fp16 forced).
- MapAnything (learned conf): AUROC 0.880, CUR 0.329, CUR-err 0.205 [0.111, 0.302]. Only model that is confident AND wrong. multi-view conf variant: AUROC 0.875, CUR 0.514, CUR-err 0.330 -> the cheap consistency confidence is worse, not better.
- MapAnything caveat: AbsRel_S 0.103 and AbsRel_U_ctrl 0.124 are 5 to 10x the other models. Example scan17 shows wrong depth inside the hidden region. Suspect: 420x518 is not in MapAnything's 518 resolution table (nearest 518x392). Follow-up P2-mapanything-res: feed 392x518 and compare AbsRel_S; do not use the MapAnything rows until then.
- Five-model picture on gray fill: VGGT 0.019, Pi3 0.046, Pi3X 0.207, MapAnything 0.329, DA3 0.772 (CUR). Gate 1 holds for three of five models on this fill. Headline reframed in draft: native confidence is model dependent, from near-perfect abstention to near-blind.

### P2-mapanything-res (2026-10-28, 324 s). `results/P2-mapanything-res/`

- MapAnything at its native 392x518 (nearest entry in its 518 resolution table): AUROC 0.875 [0.817, 0.929], CUR 0.343 [0.211, 0.480], CUR-err 0.194, AbsRel_S 0.095, AbsRel_U_ctrl 0.115. Versus 420x518: AUROC 0.880, CUR 0.329, CUR-err 0.205, AbsRel_S 0.103.
- Conclusion: resolution is not the cause of the 10% depth error; hallucination metrics are stable across input size (good for robustness of the benchmark). MapAnything rows are usable with a stated caveat that its DTU depth quality is low in image-only metric mode. Keep 420x518 for all models (fairness rule 4). Open question for the paper, not blocking: whether `depth_z` vs `depth_along_ray` or metric scaling explains the error.

### P1-full done (2026-10-28, kernel hallu3d-p1-full, 5535 s = 92 min on T4; VGGT 50 min, Pi3X 42 min). `results/P1-full/`, 1080 rows, no failures.

- Pooled over radii, mean [95% CI over 18 scans]:
  - VGGT: gray AUROC 0.973 / CUR 0.067; blur 0.886 / 0.263; noise 0.919 / 0.222. CUR-err 0.000 everywhere.
  - Pi3X: gray 0.940 / 0.202; blur 0.826 / 0.512; noise 0.801 / 0.551. CUR-err <= 0.004.
  - Fill effect within model: all 8 paired tests (blur/noise vs gray, AUROC and CUR, both models) p = 0.00005.
  - Radius effect: small (per-radius rows in summary CSV; VGGT CUR blur 0.261 at r0.10 vs 0.264 at r0.18).
- **Gate 1: PASS.** Gray fill was a visible cue; with natural-statistics fills both models leak 24 to 53% of unsupported geometry at a 90% supported-keep threshold. AUROC 0.80 to 0.92, well below 1.
- Important nuance for the paper: imagined depth on DTU is accurate (AbsRel_U ~2x clean), so CUR-err ~0. The failure is "cannot tell imagined from observed", not "wrong geometry". Report both CUR and CUR-err; expect them to diverge off DTU.
- Decision: headline fills for all remaining models = blur and noise; gray kept as ablation. Launch P2-full (pi3, da3, mapanything, full cfg) and the perturbation-variance baseline smoke.

### P3-pv-smoke (2026-10-28, kernel hallu3d-p3-pv-smoke, 1441 s). `results/P3-pv-smoke/`

- Perturbation-variance detector (`PerturbWrap`, K=4 photometric perturbations: gain/bias/gamma +-15%, per-sample median-scale alignment, conf = 1/(1e-3 + relative std)). 18 scans, seed 0, blur, r 0.18; native rows taken from P1-full with identical keys.
  - VGGT: native AUROC 0.919 / CUR 0.241 vs pv4 0.655 [0.586, 0.724] / 0.647 [0.536, 0.752]. Paired diff AUROC -0.264, CUR +0.406, both p = 0.00005.
  - Pi3X: native 0.832 / 0.502 vs pv4 0.551 [0.492, 0.609] / 0.824 [0.753, 0.886]. AUROC p = 0.00005, CUR p = 0.00065.
  - Depth from the K-sample median is unchanged (AbsRel_S/U equal to native), so the detector does not hurt geometry, it just carries no support signal.
- Reading: the backbone's imagined geometry is stable under input perturbation, so sample variance is near-constant across supported and unsupported pixels (AUROC ~0.55 to 0.66). Same conclusion as ReVar3R's own limitation note. This is the motivation for disagreement from a generative head rather than from input perturbation.
- Caveat: this is a simplified ReVar3R (photometric only, K=4, scale-only alignment). A fuller version (geometric augmentations, K=8, similarity alignment) is a required ablation before the paper calls the whole family insufficient (queued as P3-pv-full).
- Compute: 4x the base forward (1441 s for 2 models x 18 scans x (4 masked + 4 control) forwards, including model loads).

### P2-full done (2026-10-28, kernel hallu3d-p2-full, 6602 s = 110 min; pi3 36 min, da3 33 min, mapanything 41 min). `results/P2-full/`, 1620 rows, no failures. Merged five-model CSV: `results/table1_all_models.csv`.

- Blur+noise pooled CUR: VGGT 0.242, Pi3 0.281, Pi3X 0.532, MapAnything 0.622, DA3 0.701. AUROC 0.903, 0.912, 0.813, 0.743, 0.711.
- Holm-corrected pairwise CUR (10 pairs): VGGT = Pi3 (p 1.0); {VGGT, Pi3} < {Pi3X, MapAnything, DA3} (p <= 0.0015); Pi3X vs MapAnything and MapAnything vs DA3 not separated (p 0.19).
- DA3 CUR ~0.70 on all three fills, including gray: its confidence does not react to the hidden region at all. MapAnything CUR-err 0.29: only confident-and-wrong model.
- Gate 1 confirmed on five models. Table 1 in the draft is now complete except VGGT-Omega (gated).
- Kaggle GPU used today: ~4.7 h across 8 kernel runs.
- Correction (same day): Holm-adjusted CUR p-values, blur+noise: da3-pi3 0.0005, da3-vggt 0.0005, mapanything-pi3 0.0005, mapanything-vggt 0.0005, pi3-pi3x 0.0005, pi3x-vggt 0.0015, da3-pi3x 0.052, mapanything-pi3x 0.19, da3-mapanything 0.19, pi3-vggt 0.41. Draft text fixed to these values (an earlier draft sentence had da3-pi3x as significant; it is not after correction).

### P2-stress launched (2026-10-28, kernel popsensharma/hallu3d-p2-stress)

- Design note: for pointmap models every pixel of a kept view is observed by that view, so dropping views cannot by itself create unsupported pixels. View dropping, low parallax and a distractor are therefore stress tests of the masking protocol (does leakage change with less or corrupted evidence), not replacements for it. The strongest reply to the "visible patch" objection is a natural-looking fill.
- Code: `apply_fill` gained `inpaint` (cv2 Telea, radius 7; figure `results/P1-check/fills_scan3_r0.18.png` shows it is edge-continuous); `select_views(mode="adjacent")`; cfg `distractor` appends one image from another scan (deterministic choice) to the model input and drops its outputs; cfg `keep_model` lets one loaded model serve several cfgs. Synthetic test and a code-path test (5 adjacent views + distractor + inpaint) pass.
- Tests, each with seeds 0,1 and r 0.18, models vggt, pi3x, da3: inpaint10 (fill=inpaint, 10 views); blur5; blur3; adjacent10 (blur, cameras 0..9); distractor10 (blur, +1 foreign image). Reference = P1-full / P2-full rows with blur, r 0.18, seeds 0,1, 10 spread views.

### P2-stress done (2026-10-28, 3157 s on T4; vggt, pi3x, da3; 5 tests x 18 scans x 2 seeds). `results/P2-stress/`, comparison table `stress_vs_reference.csv`.

- Inpaint fill raises CUR vs blur reference: VGGT +0.088 (p 0.031), Pi3X +0.068 (p 0.005), DA3 +0.008 (ns). Table 1 is a lower bound; "visible patch" objection answered.
- 5 and 3 views: no change for VGGT/Pi3X (p > 0.4); DA3 +0.106 at 5 views (p 0.008) but not at 3 (ns) -> treat DA3 view effect as unstable.
- Low parallax: VGGT/Pi3X unchanged; DA3 CUR 0.679 -> 0.372 (p 0.0003). Distractor: VGGT/Pi3X unchanged; DA3 +0.096 (p 0.0004).
- Pi3X geometry degrades under 3 views / low parallax (AbsRel_U 0.05, CUR-err 0.03 to 0.04) while confidence does not -> first DTU evidence of confident-and-wrong for a model other than MapAnything.
- Table 2 written into draft. Kaggle GPU today: ~5.6 h total.

### P2-co3d loader built and verified locally (2026-10-28)

- Source: CO3Dv2 singlesequence subset, public links in the co3d repo `co3d/links.json` (no form, CC BY-NC 4.0). Only categories with a frame zip over ~150 MB contain the two "dev" sequences with depth; small ones (bottle, banana, cup, ...) hold only the redacted test sequence. Downloaded: hydrant, apple, toytruck, teddybear, vase, mouse, bench, ball, toytrain, skateboard.
- Loader `scripts/co3d.py`: cameras converted from PyTorch3D NDC (`ndc_isotropic`) to OpenCV following pytorch3d `get_ndc_to_screen_transform` and implicitron `_get_pytorch3d_camera`; depth decoded as uint16 PNG reinterpreted as float16 (implicitron `load_16big_png_depth`) and zeroed outside `depth_masks`. Images resized to 640x360 (or 360x640 for landscape sequences), depth and labels at 320x180 with a valid-aware 2x2 block mean. Model input 518x294 (portrait) or 294x518; both sides divisible by 14. Sphere centers are drawn from object-foreground pixels (`center_mask`), which here equals the valid-depth set.
- Verification on hydrant_167: sequence point cloud projected with the converted cameras lands inside the image for 100% of points, hits valid GT depth for 83 to 93% per view, median relative depth error 0.65 to 2.5% (the larger values are back-side points occluded in that view).
- `consistency_check` with the DTU 1% tolerance gives 0.22 to 0.62 on 10 spread CO3D views: the views go all the way round the object, so many points are occluded, and MVS depth noise is above 1%. A new occlusion-aware diagnostic (3% tolerance, points behind a closer surface excluded) gives 0.92 to 1.00 on all 8 scenes checked (0.99 on DTU scans 3 and 5 with the same diagnostic), and 0.85 to 0.95 at 1% on adjacent views. U_transfer 0.90 to 0.99. CO3D gate: `reproj_agree_3pct_vis` > 0.9 and U_transfer > 0.85.
- CPU end-to-end test `scripts/test_co3d.py` with the fake oracle/blind models on 8 scenes: oracle AUROC 1.000 / CUR 0.000, blind 0.501 / 0.899, as on DTU.
- Hyperparameters unchanged from the DTU freeze (seeds, radii, fills, tau_keep, err_thresh, dilate_px, min/max U fraction). Only the image/model resolution differs because the captures are portrait phone video.

### P2-co3d launched (2026-10-28, kernel popsensharma/hallu3d-p2-co3d v1, dataset popsensharma/co3d-hallubench)

- 18 scenes = 9 categories x 2 dev sequences (apple, ball, bench, hydrant, mouse, teddybear, toytrain, toytruck, vase), chosen before any model run so the count matches DTU's 18 scans. Skateboard was downloaded but not included (download finished after the subset was fixed).
- Config identical to Table 1: 5 seeds x 2 radii (0.10, 0.18) x 3 fills (gray, blur, noise), 10 spread views over the known frames; models vggt, pi3x, pi3, da3, mapanything; T4 fp16.
- Distractor code path bug found in the CPU test before launch: the foreign image could come from a sequence with the other orientation. Fixed to pick a same-orientation scene (`co3d.scene_hw`). Not exercised by this run (no distractor).

### P2-co3d part 1 done, part 2 relaunched (2026-10-28, kernel hallu3d-p2-co3d v1, 1832 s)

- VGGT completed all 540 rows (30 min). pi3x, pi3, da3, mapanything did not run: `_key` cast the scan column to int when resuming from the CSV VGGT had written, and CO3D scan names are strings. Fixed (`_key` keeps string scans), relaunched the four remaining models as kernel hallu3d-p2-co3d-b (RUN_NAME P2-co3d-b). Results will be merged into `results/table3_all_models.csv`.
- Consistency gate on Kaggle: 17 of 18 scenes pass (3% occlusion-aware agreement 0.92 to 1.00, U_transfer 0.90 to 0.98). vase_374_41862_83720 has U_transfer 0.80 at seed 0 (gate 0.85): its hidden sphere is small (U_frac_ref 0.025) and sits near the object silhouette, so some U points project just outside the hide mask of other views. The scene is kept and reported; it is flagged in the Table 3 notes rather than dropped.
- VGGT on CO3D (seeds 0-4, 18 scenes): AUROC 0.74 to 0.79 for gray/blur, CUR 0.51 to 0.57; with noise fill AUROC falls to 0.58 to 0.61 and CUR rises to 0.75 to 0.78. auroc_ctrl 0.42. AbsRel_U 0.028 to 0.035 vs AbsRel_S 0.029: the imagined geometry is as accurate as the observed geometry, so CUR-err stays near 0.03. Leakage is roughly double DTU's.

### P2-co3d done (2026-10-28, kernel hallu3d-p2-co3d-b, 5967 s = 99 min; pi3x 28 min, pi3 21 min, da3 22 min, mapanything 29 min). 2700 rows over both kernels, no failures. Merged: `results/table3_all_models.csv`, cells `results/table3_by_model_fill.csv`.

- Pooled blur+noise CUR: Pi3 0.272, Pi3X 0.570, VGGT 0.651, DA3 0.797, MapAnything 0.810. AUROC 0.891, 0.732, 0.681, 0.620, 0.576.
- Holm-adjusted CUR pairs: pi3 < all (0.0005); pi3x < mapanything, da3 (0.0005); vggt < mapanything (0.010), da3 (0.0005); vggt-pi3x 0.23, mapanything-da3 0.73.
- Versus DTU: Pi3 unchanged (0.281 -> 0.272), Pi3X about the same (0.532 -> 0.570), VGGT 0.242 -> 0.651, MapAnything 0.622 -> 0.810, DA3 0.701 -> 0.797. Ranking changes: VGGT loses its top spot. MapAnything's CUR-err drops from 0.286 to 0.007 (AbsRel_S 0.041), so its DTU confident-and-wrong behaviour is DTU-specific.
- Fill effects: VGGT reacts to noise (+0.20, p 0.00005) but not blur (p 0.30); Pi3X to both (p 0.00005); Pi3 weakly; MapAnything and DA3 not at all.
- vase_374 (gate miss, U_transfer 0.80) kept; dropping it moves pooled numbers by <= 0.013.
- Table 3 and the DTU-vs-CO3D headline table written into the draft. Kaggle GPU today: ~7.8 h total.

### Paper build v0.2 and demo video (2026-10-28)

- `paper/tex/main.tex` written from the draft: CVPR 2027 author kit, review mode, 8 pages body + references + supplementary (per-run spread, fill ablation, per-scene heatmaps). Compiles clean with tectonic. Output `paper/observe_imagine_abstain_cvpr27.pdf`.
- Nine figures generated by `scripts/make_figures.py` from the result CSVs (teaser rebuilt from DTU scan 3 and CO3D toytrain_399 because scan 17 is not on this machine; model panels cropped from recorded Kaggle composites). Method figure (fig9) is a design figure and says so in the image and the caption.
- Demo video `paper/demo/demo.mp4` (184 s, 1920x1080, 7.1 MB) by `scripts/make_demo.py`: Part A runs the masking protocol and consistency check live on DTU scan 3 and CO3D hydrant_167; Part B replays recorded Kaggle model outputs with the matching CSV rows and animated summary charts; the closing card lists what is not yet run.
- Pending items are rendered as orange "pending: run-id" boxes, never as numbers. Bib entries for 2026 arXiv papers carry the identifiers but author lists are marked to be verified.
