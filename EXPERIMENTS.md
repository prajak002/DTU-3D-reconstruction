# Experiment plan and run registry

Each run has an id. A number enters the paper only when the run's CSV and `_cfg.json` exist under `results/<run-id>/`. Keep this file and `LOG.md` in sync.

## Done

| run-id | what | where | outcome |
|---|---|---|---|
| synthetic | `test_synthetic.py` sanity test on CPU (Mac, 2026-10-28) | `results/synthetic/` | PASS: oracle AUROC 1.000 / CUR 0.000; blind AUROC 0.500 / CUR 0.902; reproj_agree 0.952, U_transfer 0.996; masks deterministic; U and S disjoint. `hallu3d.py` md5 faf9ce41. |

| P1-check | `consistency_check` on real DTU scans 3, 5 (local CPU and Kaggle) | seeds 0,1; r 0.10, 0.18 | `results/P1-check/` | PASS with note: reproj_agree 0.796 / 0.824, U_transfer 0.96 to 0.99. |
| P1-smoke | VGGT + Pi3X, 18 scans, seed 0, gray, r 0.15, T4 | `results/P1-smoke/` | 555 s | VGGT AUROC 0.991 / CUR 0.019; Pi3X AUROC 0.940 / CUR 0.207. See LOG. |

| P1-full | VGGT + Pi3X, DEFAULT_CFG (18 x 5 x 2 x 3), T4 | `results/P1-full/` | 92 min | **Gate 1 PASS.** blur/noise: VGGT AUROC 0.89/0.92, CUR 0.26/0.22; Pi3X 0.83/0.80, CUR 0.51/0.55. |
| P2-smoke, P2-smoke-b | pi3, da3, mapanything(+mv), smoke cfg | `results/P2-smoke*/` | 5 + 11 min | gray CUR: pi3 0.046, mapanything 0.329, da3 0.772. |
| P2-mapanything-res | mapanything at native 392x518 | `results/P2-mapanything-res/` | 5 min | unchanged vs 420x518; depth error is the model's. |

| P2-full | pi3 + da3 + mapanything, DEFAULT_CFG | `results/P2-full/` | 110 min | blur+noise CUR: pi3 0.28, mapanything 0.62, da3 0.70. Table 1 complete for 5 models. |
| P3-pv-smoke | vggt_pv4 + pi3x_pv4, blur, r 0.18, seed 0 | `results/P3-pv-smoke/` | 24 min | perturbation variance worse than native: AUROC 0.66 / 0.55 vs 0.92 / 0.83. |

| P2-stress | inpaint fill, 5/3 views, adjacent cams, distractor; vggt, pi3x, da3; seeds 0,1; r 0.18 | `results/P2-stress/` | 53 min | Table 2. Inpaint raises CUR (VGGT +0.09, Pi3X +0.07); view count no effect; DA3 reacts to parallax and distractor. |

## Running

(none)

## Queued (Kaggle T4/P100, fp16, 10 views, 420x518)

| run-id | what | config | expected time | fills |
|---|---|---|---|---|
| P1-full | VGGT + Pi3X full | 18 scans x 5 seeds x 2 radii x 3 fills = 540 masked forwards per model + 18 control | ~1 session per model (estimate, measure on smoke) | Table 1 rows 1 to 2; **Gate 1** (Nov 1). |
| P2-omega | VGGT-Omega (gated; needs HF access) | same as P1-full | ~1 h | Table 1 last row. |
| P2-co3d | CO3Dv2 singlesequence subset, 2 dev sequences per category (hydrant, apple, toytruck, teddybear, vase, mouse, bench, ball, toytrain, skateboard as available), 5 models | scenes x 5 seeds x 2 radii x 3 fills, 10 spread views over the ~100 known frames, model input 518x294 (portrait) or 294x518 | ~35 min per model | Table 3. **Done** 2026-10-28: kernels hallu3d-p2-co3d (vggt, 1832 s) and hallu3d-p2-co3d-b (pi3x, pi3, da3, mapanything, 5967 s). `results/P2-co3d/`, `results/P2-co3d-b/`, merged `results/table3_all_models.csv`. |
| P3-pv-full | fuller perturbation baseline: geometric aug, K=8, similarity alignment; all 5 models, blur+noise | Kaggle, ~8x base cost | ~2 h per model | Table 4 ablation. |
| P3-reproj | reprojection-consistency detector (no learning) | Kaggle | cheap | Table 4 row. |
| P3-main | diffusion head, K-sample disagreement vs training-free baselines | H200; 3 to 5 training seeds | | Table 4; **Gate 2** (Nov 7). |
| P3-abl | K, fill, radius, backbone, SigLIP2 on/off | | | Table 5. |
| P3-compute | wall-clock and FLOPs per detector | | | Section 5 compute. |
| P4-cont | LoRA + replay on a site stream | | | Table 6. |
| P4-heritage | heritage photos, language-query figure | | | Figure 5. |

## How to run on Kaggle from this machine

```
cat scripts/kaggle_prelude.py scripts/hallu3d.py scripts/kaggle_driver.py > <dir>/hallu3d_kaggle.py
# edit RUN_NAME / MODELS / CFG defaults at the top of the driver section, write kernel-metadata.json
# (dataset_sources: tobyzhou/dtu-training, enable_gpu, enable_internet), then:
kaggle kernels push -p <dir>
kaggle kernels status <user>/<kernel>
kaggle kernels output <user>/<kernel> -p <outdir>     # out/<RUN_NAME>.csv, _summary.csv, _consistency.json, ex_<RUN_NAME>/
```

## Kaggle cells for P1 (notebook alternative)

```
!pip -q install git+https://github.com/facebookresearch/vggt.git
!git clone -q https://github.com/yyfz/Pi3.git && pip -q install -e Pi3
import sys; sys.path.append('/kaggle/working/Pi3'); sys.path.append('/kaggle/working/scripts')
import hallu3d as H
root = H.find_dtu_root()
views = H.select_views(10)
for sc in [3, 5]:
    d = H.load_scan(root, sc, views)
    m = H.build_masks(d, 0, 0.15, H.DEFAULT_CFG)
    print(sc, H.consistency_check(d, m))          # P1-check
H.run_benchmark(root, ["vggt", "pi3x"], cfg={"seeds": [0], "fills": ["gray"], "radius_fracs": [0.15]},
                out_csv="m1_smoke.csv", save_examples_dir="ex")   # P1-smoke
H.run_benchmark(root, ["vggt", "pi3x"], out_csv="m1_results.csv")  # P1-full, resumable
print(H.summarize("m1_results.csv").to_string())
print(H.paired_permutation(__import__("pandas").read_csv("m1_results.csv"), "auroc", "vggt", "pi3x"))
```

Download `m1_results.csv`, `m1_results_cfg.json`, and `ex/` into `results/P1-full/` and update `LOG.md`.

## Gates

- **Gate 1 (Nov 1):** on P1-full, CUR mean with 95% CI excluding 0 and AUROC clearly below 1 for both models. Fail: models already abstain; report that, pivot to method-only or stop.
- **Gate 2 (Nov 7):** on P3-main, disagreement beats native confidence on AUROC and CUR, paired sign-flip p < 0.05 over scans, at equal compute. Fail: paper = benchmark + best training-free detector.

## Known code notes

- `paired_permutation` returns NaN with fewer than 2 scans (by design); the synthetic test has 1 scan.
- `consistency_check` now takes the image size from the data; it also returns `reproj_agree_3pct_vis` (3% tolerance, occluded points excluded), which is the gate used for CO3D's MVS ground truth. The DTU gate stays at `reproj_agree` (1%).
- CO3D: `scripts/co3d.py` (preprocess raw singlesequence zips to `<scene>/images/*.jpg` + `meta.npz`, loader, camera conversion). `cfg["dataset"]="co3d"` switches `run_benchmark`; scans are scene names. Driver: `kaggle_driver_co3d.py`; the Kaggle build is `prelude + hallu3d.py + co3d.py + driver`. Kaggle dataset: `popsensharma/co3d-hallubench` (private, preprocessed only).
- CO3D facts that shape the subset: per category the singlesequence zip has 3 sequences, only the two "dev" ones carry depth (the test one is redacted), and only `*_known` frames (about 102 of 202) carry depth. Depth is COLMAP MVS, valid on the object only (6 to 15% of pixels). Categories whose zip is under ~150 MB contain only the test sequence and are useless. Stress tests are in `kaggle_driver_stress.py`.
