# Experiment plan and run registry

Each run has an id. A number enters the paper only when the run's CSV and `_cfg.json` exist under `results/<run-id>/`. Keep this file and `LOG.md` in sync.

## Done

| run-id | what | config | where | time | outcome |
|---|---|---|---|---|---|
| synthetic | `test_synthetic.py` sanity test on CPU (2026-10-28; re-run 2026-10-31 with the reprojection check) | 1 synthetic scan, seeds 0,1, r 0.15, gray + noise | `results/synthetic/` | CPU, ~1 min | PASS: oracle AUROC 1.000 / CUR 0.000; blind AUROC 0.500 / CUR 0.902; reproj_agree 0.952, U_transfer 0.996; masks deterministic; U and S disjoint. |
| P1-check | `consistency_check` on real DTU scans 3, 5 (local CPU and Kaggle) | seeds 0,1; r 0.10, 0.18 | `results/P1-check/` | CPU | PASS with note: reproj_agree 0.796 / 0.824, U_transfer 0.96 to 0.99. |
| P1-smoke | VGGT + Pi3X | 18 scans, seed 0, gray, r 0.15, T4 | `results/P1-smoke/` | 555 s | VGGT AUROC 0.991 / CUR 0.019; Pi3X AUROC 0.940 / CUR 0.207. See LOG. |
| P1-full | VGGT + Pi3X | DEFAULT_CFG (18 x 5 x 2 x 3), T4 | `results/P1-full/` | 92 min | **Gate 1 PASS.** blur/noise: VGGT AUROC 0.89/0.92, CUR 0.26/0.22; Pi3X 0.83/0.80, CUR 0.51/0.55. |
| P2-smoke, P2-smoke-b | pi3, da3, mapanything (+mv) | smoke cfg | `results/P2-smoke*/` | 5 + 11 min | gray CUR: pi3 0.046, mapanything 0.329, da3 0.772. |
| P2-mapanything-res | mapanything at native resolution | 392x518 | `results/P2-mapanything-res/` | 5 min | unchanged vs 420x518; depth error is the model's. |
| P2-full | pi3 + da3 + mapanything | DEFAULT_CFG | `results/P2-full/` | 110 min | blur+noise CUR: pi3 0.28, mapanything 0.62, da3 0.70. Table 1 complete for 5 models; merged `results/table1_all_models.csv`. |
| P3-pv-smoke | vggt_pv4 + pi3x_pv4 | blur, r 0.18, seed 0 | `results/P3-pv-smoke/` | 24 min | perturbation variance worse than native: AUROC 0.66 / 0.55 vs 0.92 / 0.83. |
| P2-stress | inpaint fill, 5/3 views, adjacent cams, distractor; vggt, pi3x, da3 | seeds 0,1; r 0.18 | `results/P2-stress/` | 53 min | Table 2. Inpaint raises CUR (VGGT +0.09, Pi3X +0.07); view count no effect; DA3 reacts to parallax and distractor. |
| P2-co3d, P2-co3d-b | five models on the CO3Dv2 subset (18 scenes = 9 categories x 2 dev sequences) | DEFAULT_CFG, 10 spread views, 518x294 | `results/P2-co3d*/`, merged `results/table3_all_models.csv` | 31 + 99 min | Table 3. Pooled blur+noise CUR: Pi3 0.272, Pi3X 0.570, VGGT 0.651, DA3 0.797, MapAnything 0.810. vase_374 misses the U_transfer gate (0.80), kept and flagged. |
| P3-check | feature hooks on the real backbones (`run_h100.py check`), 200-step `vggt_oia` smoke on training scans 7, 8, 14, 16, evaluation on scans 3, 5, 17 | Kaggle T4, kernel `hallu3d-p3-check` v1 (commit ca2977b) | `results/P3-check/` | 43 min (training 36 min = 10.7 s/step on T4) | **Smoke only, nothing enters the paper.** Hooks give the documented shapes (VGGT 4 x 2048-d maps at 30 x 37, Pi3X 1 x 2048-d; peak 7.7 / 9.2 GB). Self-consistency of predicted depth + cameras: U_transfer 0.986 (VGGT), 0.981 (Pi3X). 200-step head, Table-4 keys: vggt AUROC 0.954 / 0.930 / 0.982, CUR 0.112 / 0.282 / 0.001 on scans 3 / 5 / 17; vggt_oia 0.9999 / 1.0000 / 0.9999, CUR 0.000 on all three; vggt_oiafuse 0.996 / 1.000 / 1.000. The kernel's own per-scan print showed NaN because it filtered the gray fill, which this run did not use; the CSV is complete. The per-image control and the inpainted fill decide whether this is support detection or patch detection. |
| P3-native, P3-reproj, P3-pv | native confidence, reprojection consistency (`<base>_reproj`) and perturbation variance (`<base>_pv4`) on the Table-4 keys, paired with the heads on identical rows | blur, r 0.18, seed 0, 18 scans; vggt on the L4 (bf16), pi3x on a Kaggle T4 (fp16, kernel `hallu3d-p3-pi3x`) | `results/P3-native/`, `P3-reproj/`, `P3-pv/` | minutes | Table 4. Native 0.915 / 0.255 (VGGT), 0.832 / 0.502 (Pi3X); reproj at chance 0.488 / 0.931 and 0.478 / 0.929; pv4 0.666 / 0.627 and 0.550 / 0.824. |
| P3-oia-dtu | support head `<base>_oia`, `vggt_oiafuse`, per-image control `<base>_oiapatch`, full configuration | 18 scans x 5 seeds x 2 radii x 3 fills; VGGT heads trained on the L4 (1500 steps, 7651 s), Pi3X heads on the Kaggle T4 (1500 steps, 18360 s) | `results/P3-oia-dtu/` (2700 rows) | L4 about 1.5 h, T4 about 1.5 h | **Gate 2 part 1 PASS** for both backbones (Table-4 keys: VGGT dAUROC +0.085, dCUR -0.255; Pi3X +0.167, -0.501; p = 0.00005). **Part 2 FAIL**: the per-image control matches the head on every cell (AUROC within 0.002, all 1.000 at r 0.18 except blur). |
| P3-oia-stress_* | head and control on the five stress tests (inpaint10 = held-out fill, blur5, blur3, adjacent10, distractor10) | seeds 0, 1; r 0.18; both backbones | `results/P3-oia-stress_*/` (144 rows each) | about 1 h per backbone | head 0.994 to 1.000 everywhere; control the same (inpaint: VGGT 1.000 vs 0.999, Pi3X 0.994 vs 0.994). The held-out fill does not separate head from control. |
| P3-oia-abl, P3-oia-abl-seed1/2 | linear probe `vggt_oialin`, control `vggt_oiapatch`, transformer head `vggt_oiaattn`, `pi3x_oiapatch`, extra seeds `vggt_oia_s1`, `_s2` | seed 0, r 0.18, gray / blur / noise | `results/P3-oia-abl*/` | 30 min | every learned head at AUROC 1.000 / CUR 0.000 on the Table-4 keys; the probe and the transformer head add nothing because there is nothing left to add. |
| P3-oia-co3d | support head, fusion and per-image control on CO3Dv2 (never seen by the heads), full configuration | 18 scenes x 5 seeds x 2 radii x 3 fills; VGGT heads (L4), Pi3X head (Kaggle rows) | `results/P3-oia-co3d/` (2070 rows; the 540 `pi3x_oiapatch` rows still running) | L4 75 min | blur+noise AUROC: vggt_oia 0.992 [0.987, 0.996], vggt_oiapatch 0.989 [0.985, 0.993], vggt_oiafuse 0.961, pi3x_oia 0.975, native vggt 0.681. Head minus control +0.003 at p 0.031: the pre-registered clause "head beats control at p<0.05" is formally met while the control carries 99% of the gain; **not counted as a pass** (paper says so). |
| P3-kaggle-vggt | independent replicate of the VGGT heads (oia, oialin, oiapatch, seeds 1, 2; no oiaattn) trained on a T4 from commit 74a13c7, then every VGGT stage incl. co3d (1620 rows) and dtuext (396 rows) | kernel `popsensharma/hallu3d-p3-vggt` v1, 11.3 h | `results/P3-kaggle-vggt/` (CSVs and logs only; different weights from the reported heads, never merged) | 11.3 h | same picture as the L4 heads: DTU 1.000 / 0.000; **CO3Dv2 head 0.992 vs control 0.990 (p 0.10), both +0.31 over native**; dtuext head and control 1.000. Its own `P3-stats.md` is in the folder. |
| P3-kaggle-pi3x | Pi3X heads (`pi3x_oia`, `pi3x_oiapatch`, 1500 steps) and their baselines, dtu, stress, ablate stages, 450 of 1080 co3d rows | kernel `popsensharma/hallu3d-p3-pi3x` v1, 8.2 h | `results/P3-kaggle-pi3x/` (raw copy); rows merged into the `P3-*` folders above and the heads adopted on the L4 as the Pi3X heads of record (`scripts/box/adopt_kaggle_pi3x.sh`) | 8.2 h | finished before the L4 could train the Pi3X heads (box stopped at step 800 on 2026-11-05); adopted as planned ("whichever finishes first"). |

## Running

| run-id | what | config | where | started | notes |
|---|---|---|---|---|---|
| P4-vis on Kaggle | `scripts/kaggle_p3_stages.py` with COMMIT 4132d11, STAGES `trainvis,vis,vis_co3d`: trains `<base>_oiavis`, `_oiavispatch`, `_oiavislin` on the 79 DTU training scans (T4, about 4.5 h), then the vis protocol on the 18 DTU scans (the vggt kernel also scores the native confidence of pi3x, pi3, da3 and mapanything) and on CO3Dv2 | kernels `popsensharma/hallu3d-p4-vis-vggt`, `hallu3d-p4-vis-pi3x` v1 | Kaggle T4, fp16 | 2026-11-06 11:55 UTC | whichever of Kaggle or the L4 finishes first supplies the vis heads of record (the other becomes a replicate, as with P3). |
| P3 (all) | `scripts/box/p3_all.sh`, relaunched 2026-11-06 08:35 UTC (`tmux` session `p3`, `~/logs/p3_all.out`): pipeline A (VGGT: ablate done, **co3d running**, then dtuext, eth3d) and pipeline B (Pi3X stages are no-ops after the Kaggle adoption except the remaining 630 co3d rows; then `gs`, `pvfull`), a final sweep, `compute_cost`, `stats`, the three-way figure | every stage waits for free GPU memory (`wait_free`, no practical cap); 12 GB swap enabled by the script; the Kaggle Pi3X heads are in `checkpoints/` (the box's step-750 ones kept as `*_box750.pt`) | AWS g6.xlarge, one L4, shared with another user's jobs (12 GB that morning) | VGGT co3d at about 30 rows a minute; results are copied back into `results/P3-*/` as stages finish | the box's result folders are its own copies; code is updated with `git fetch && git checkout origin/main -- scripts` (a plain pull refuses because the result files differ). |

## Queued (Kaggle T4/P100, fp16, 10 views, 420x518)

| run-id | what | config | expected time | fills |
|---|---|---|---|---|
| P2-omega | VGGT-Omega (gated; needs HF access) | same as P1-full | ~1 h | Table 1 last row. |
| P3-pv-full | fuller perturbation baseline: geometric aug, K=8, similarity alignment; all 5 models, blur+noise | Kaggle, ~8x base cost | ~2 h per model | Table 4 ablation. |
| P3-reproj | reprojection-consistency detector (no learning): `vggt_reproj`, `pi3x_reproj` (`hallu3d.ReprojWrap` + `reproj_support_score`, cameras from the model). | L4, stage `baselines` (pipeline A for vggt, B for pi3x), Table-4 keys | ~10 min | Table 4 row (same keys as the native rows). |
| P3-oia (main) | **support head** (`scripts/oia_head.py`): conv head on frozen VGGT / Pi3X features, label-free from the backbone's own clean prediction, 79 DTU training scans; the head never sees CO3Dv2 (the backbones did, see paper). Model names `vggt_oia`, `vggt_oiafuse`, `pi3x_oia`, control `<base>_oiapatch` on every row (issue #16). | L4, stages `train`, `dtu`, `co3d`, `stress` | train 2.3 h per backbone (all heads jointly), evals about 4 h | Table 4 (+ Table 1/3 rows), Table 5; **Gate 2**: pass condition in the paper, decided by `results/P3-stats.md`. |
| P3-oia-abl | linear probe (`vggt_oialin`), per-image control (`vggt_oiapatch`), cross-view transformer head (`vggt_oiaattn`), extra initialisation seeds (`vggt_oia_s1`, `_s2`; same data stream, different init), GT-placed masks (`vggt_oiagt`, separate run, if time), `pi3x_oiapatch` | stage `ablate`, seed 0, radius 0.18, gray / blur / noise | 30 min | Table 5. |
| P3 pipeline B (pi3x) | `train --bases pi3x` (pi3x_oia + pi3x_oiapatch jointly), `baselines`, `dtu`, `stress`, `co3d` for pi3x; then `pvfull` (all five backbones, `<base>_pvf8`, issue #3) and `dtuext` (22 MVSNet test scans, issue #9) | `~/p3b.sh`, started when pipeline A's training has released its memory | about 6 h | Table 4, Table 5, extension rows. |
| P3-gs | feed-forward Gaussian-splatting backbone (AnySplat, `scripts/anysplat_wrap.py`): `anysplat` (opacity as confidence; verified on DTU scan 3, depth AbsRel 0.016, 6.6 GB, 5 s per scene), `anysplat_render` (rendered alpha; unverified), `anysplat_reproj` | L4, stage `gs`: full DTU configuration, Table-4 keys, CO3Dv2 | ~2 h | Table 1 and Table 3 rows (sixth model) and a paragraph on representation. |
| P3-eth3d | third dataset: 9 ETH3D scenes passing the gate (see LOG 2026-11-04), native five models + vggt_oia + vggt_oiapatch, full configuration, co-visible views | L4, stage `eth3d` | ~2.5 h | new table: building-scale scenes. |
| **P4-vis** | **true-visibility protocol** (decided 2026-11-06 after Gate 2 part 2 failed in substance). Label: `hallu3d.support_count` projects every pixel's ground-truth point into the other input views; U = pixels no other view sees (strict: no unknown either), S = seen by two or more other views, outside a dilation of U. No patch, no fill. Heads trained on these labels on the 79 DTU training scans with 3 / 5 / 10 random views per step: `<base>_oiavis` (cross-view features), `<base>_oiavispatch` (per-image control), `<base>_oiavislin` (probe). Protocol `hallu3d.run_vis`: clean inputs, 3 / 5 / 10 views, three view-set rotations, every detector (native, reproj, fill-trained head and control, vis heads), metrics as in the benchmark. CPU-tested with the fake backbone (`test_oia.py`). | stages `trainvis`, `vis`, `vis_co3d`, `vis_eth3d` (`run_h100.py`); 18 DTU scans, 18 CO3Dv2 scenes, 9 ETH3D scenes | training 2.5 h per backbone, evaluation about 3 h | **Gate 3** (see Gates). Decides whether a head can detect missing support rather than the fill. |
| P3-main (future work) | discrete diffusion head, K-sample disagreement | not before the deadline | | moved to future work (issue #1). |
| P3-compute | wall-clock and FLOPs per detector | | | Section 5 compute. |
| P4-cont | LoRA + replay on a site stream | | | Table 6. |
| P4-heritage | heritage photos, language-query figure | | | Figure 5. |

## How to run on the AWS L4 box

The box (`g6.xlarge`, eu-central-1, one L4) has the repository at `~/hallu`, the DTU training set at `~/data/dtu` (Cameras, Depths and lights 1, 3, 5 of every scan; the zip was deleted to make room) and the preprocessed CO3Dv2 subset at `~/data/co3d`. `source ~/env.sh` sets PATH (`/opt/pytorch` venv, torch 2.12 cu130), PYTHONPATH (Pi3 clone + `~/hallu/scripts`) and the HF token. Stages are launched from a driver script in tmux (`~/p3a.sh`, `~/p3b.sh`); each writes `results/<RUN>/` on the box, which is copied back into this repository with scp. The box costs about $1/h while running; stop it when idle.

```
ssh ocr-rnd   # see the instance notes; then:
source ~/env.sh && python scripts/run_h100.py <stage> --dtu $DTU --co3d $CO3D --bases vggt,pi3x
```

## How to run on Kaggle from this machine

Authentication: a Kaggle access token in `~/.kaggle/access_token` (the CLI reads `KAGGLE_API_TOKEN`). On the Windows machine the antivirus re-signs TLS and the CLI's `requests` does not trust the Windows store, so run the CLI through a wrapper that calls `truststore.inject_into_ssl()` first (`pip install kaggle truststore`). The P3 kernels clone the public repository at a pinned commit instead of concatenating files: see `scripts/kaggle_p3_check.py` and its `kernel-metadata.json` next to it.


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

- **Gate 3 (P4-vis, pre-registered 2026-11-06 before any vis head was trained).** The visibility-trained head `<base>_oiavis` counts as detecting missing support when, on the DTU vis protocol pooled over 3 / 5 / 10 views, (a) it beats native confidence on AUROC and CUR (paired sign-flip p < 0.05) and (b) it beats its per-image control `<base>_oiavispatch` on AUROC by **at least 0.02** with p < 0.05 (the Gate 2 clause "any positive difference at p < 0.05" was too weak and is not reused), and (c) the same two conditions hold on at least one dataset the head never saw (CO3Dv2 or ETH3D). The fill-trained head `<base>_oia` is reported on the same rows as the transfer test of patch detection. If (b) fails, the paper reports that the single-view-support signal is also recoverable from per-image features (image borders, depth discontinuities) and the method claim is dropped for good.

- **Gate 1 (Nov 1):** on P1-full, CUR mean with 95% CI excluding 0 and AUROC clearly below 1 for both models. Fail: models already abstain; report that, pivot to method-only or stop.
- **Gate 2 (Nov 7):** on P3-main, disagreement beats native confidence on AUROC and CUR, paired sign-flip p < 0.05 over scans, at equal compute. Fail: paper = benchmark + best training-free detector.

## Known code notes

- 2026-10-31 housekeeping (no numbers changed): `requirements.txt` added; `test_synthetic.py` uses a temp dir (or `$HALLU3D_TMP`), checks the reprojection detector, and exits non-zero when a check fails; `make_figures.py --dtu/--co3d` (or `HALLU3D_DTU`/`HALLU3D_CO3D`) replaces a hard-coded scratch path and skips figs 1 and 2 when the raw data is absent; `make_demo.py` finds ffmpeg on PATH (`$FFMPEG` overrides); the Kaggle drivers import `hallu3d` when run as separate files; `make_site.py` writes `docs/explorer.html` + the CSV bundle and checks the hand-written numbers in `docs/index.html` against the CSVs; `make_supp_archive.py` builds the anonymised supplementary archive; `make_walkthrough.py` builds the end-to-end video from the synthetic test and the logged results, no raw data needed.
- 2026-11-01: `oia_head.py` (support head, label-free training, `<model>_oia*` wrappers), `run_h100.py` (the P3 plan, resumable stages, stats with Holm correction), `test_oia.py` (CPU plumbing test, passes). Real backbone feature hooks untested until a GPU: run `python scripts/run_h100.py check` first and compare the printed shapes with the docstrings in `oia_head.py`.

- `paired_permutation` returns NaN with fewer than 2 scans (by design); the synthetic test has 1 scan.
- `consistency_check` now takes the image size from the data; it also returns `reproj_agree_3pct_vis` (3% tolerance, occluded points excluded), which is the gate used for CO3D's MVS ground truth. The DTU gate stays at `reproj_agree` (1%).
- CO3D: `scripts/co3d.py` (preprocess raw singlesequence zips to `<scene>/images/*.jpg` + `meta.npz`, loader, camera conversion). `cfg["dataset"]="co3d"` switches `run_benchmark`; scans are scene names. Driver: `kaggle_driver_co3d.py`; the Kaggle build is `prelude + hallu3d.py + co3d.py + driver`. Kaggle dataset: `popsensharma/co3d-hallubench` (private, preprocessed only).
- CO3D facts that shape the subset: per category the singlesequence zip has 3 sequences, only the two "dev" ones carry depth (the test one is redacted), and only `*_known` frames (about 102 of 202) carry depth. Depth is COLMAP MVS, valid on the object only (6 to 15% of pixels). Categories whose zip is under ~150 MB contain only the test sequence and are useless. Stress tests are in `kaggle_driver_stress.py`.
