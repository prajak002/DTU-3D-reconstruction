# Observe, Imagine, Abstain

Hallucination-aware 3D reconstruction. This repository holds **HalluBench-3D**, a benchmark that measures how confidently feed-forward 3D models report geometry that no camera saw, the logged results for five models on two datasets, the CVPR 2027 paper build, a demo video and the project website.

Project page: https://prajak002.github.io/DTU-3D-reconstruction/
Paper (PDF): `paper/observe_imagine_abstain_cvpr27.pdf` (supplementary as a separate PDF: `paper/observe_imagine_abstain_cvpr27_supp.pdf`)
Demo video: `paper/demo/demo.mp4` (protocol on real data + replay, 3 min). End-to-end walkthrough: `paper/demo/walkthrough.mp4` (protocol, models, metrics, runs, paper, artefacts, 5 min, chaptered)

---

## 1. The problem in one paragraph

Models such as VGGT, Pi3, MapAnything and Depth Anything 3 take a handful of unposed photos and return dense 3D geometry in one pass. They output geometry at *every* pixel, including pixels whose true surface appears in no photo. The training objective rewards a plausible guess, so the models learn to guess, and the guess is usually good. The problem is the confidence map: it does not tell you which parts were observed and which were imagined. A conservator looking at a reconstructed dome cannot tell whether a crack is in a photograph or came from what domes usually look like.

## 2. What we measure

A predicted point is **hallucinated** when it lies on a surface that no input view shows. We get these labels for free:

1. Place a sphere on the ground-truth surface of a scene.
2. Hide every pixel, in every view, whose true 3D point lies inside the sphere. The hidden region is consistent in 3D, so no camera sees it from any angle.
3. Fill the hole with gray, a blur, noise, or an inpainting, so the result is not an artefact of one kind of patch.
4. Run the model on the filled views. Pixels inside the sphere are **unsupported (U)**, pixels well outside it are **supported (S)**, a thin ring between is ignored.
5. Score the model's own confidence.

Metrics, per (model, scene, seed, radius, fill):

| metric | meaning |
|---|---|
| AUROC | probability that a supported pixel gets higher confidence than an unsupported one. 1.0 = perfect, 0.5 = confidence carries no information about support. |
| CUR | Confident-Unsupported Rate. Pick the threshold that keeps 90% of supported pixels; CUR is the share of unsupported pixels that also pass. This is the hallucination rate a user experiences after filtering by confidence. 0 is ideal, 0.9 is a blind guesser. |
| CUR-err | share of unsupported pixels that pass the threshold *and* are wrong by more than 10% depth. Confident and wrong. |
| AbsRel_U, AbsRel_S | depth error on unsupported and supported pixels, after median scaling fitted on supported pixels only. |
| controls | the same pixels scored under clean input, and the ratio of confidence inside the hidden region to its clean value. |

Statistics: 95% cluster-bootstrap intervals over scenes (10,000 resamples), paired sign-flip permutation tests on per-scene means (20,000 permutations), Holm correction within each family of comparisons.

## 3. Headline result

CUR pooled over blur and noise fills, 18 scenes per dataset, 5 mask seeds, 2 radii. Lower is better.

| model | DTU (lab rig) | CO3Dv2 (phone captures) |
|---|---|---|
| VGGT-1B | 0.242 | 0.651 |
| Pi3 | 0.281 | 0.272 |
| Pi3X | 0.532 | 0.570 |
| MapAnything | 0.622 | 0.810 |
| Depth Anything 3 | 0.701 | 0.797 |

What this says:

- Every model lets a large share of unseen geometry through a confidence filter: between a quarter and four fifths.
- Only Pi3 behaves the same on both datasets. VGGT, joint best on DTU, leaks 2.7x more on phone captures.
- The imagined geometry is usually *accurate* (depth error inside the hidden region is about the same as outside). The models imagine well; they do not say that they did.
- An inpainted fill, which leaves no seam, makes leakage worse, not better. Dropping from 10 views to 3 changes nothing. So the leak is not an artefact of a visible patch.
- The cheapest fix, variance over perturbed inputs, is worse than native confidence on both backbones we tried.

Full tables with intervals and the per-fill cells are in the paper (Tables 1 to 4) and in `results/table1_all_models.csv`, `results/table3_all_models.csv`.

## 4. What is done and what is not

**Done, every number from a logged run** (about 8 GPU hours on one Kaggle T4):
benchmark on DTU and CO3Dv2, five models, three fills plus an inpainted fill, stress tests (view dropping, low parallax, distractor image), two failing training-free baselines, intervals and paired tests for all of it.

**Support head trained on true visibility (P4-vis, November 6, 2026 evening; `results/P4-vis/`, `results/P3-stats.md`):** a head on the frozen cross-view features, trained on ground-truth multi-view support of the 79 DTU training scans (which pixels no other input view sees; no patch, no fill), separates single-view from supported pixels on the 18 benchmark scans with 3, 5 and 10 clean views at AUROC 0.993 (VGGT) and 0.990 (Pi3X) against 0.772 / 0.820 for native confidence and 0.878 / 0.863 for the per-image control (paired p = 0.00005; CUR 0.015 / 0.021 against 0.53 / 0.49 native). Gate 3, pre-registered with a 0.02 margin, is met on DTU and on every dataset the heads never saw: CO3Dv2 phone orbits (head 0.93, native 0.70 to 0.75, control below chance at 0.43 to 0.47), ETH3D building scans (0.95 to 0.98 against 0.72 to 0.89 and 0.69 to 0.72) and nine phototourism landmarks (0.93, against 0.80 to 0.85 and 0.72 to 0.73). Reprojection consistency is the best training-free detector on this protocol (0.954 / 0.914 on DTU), ties the head on ETH3D and loses to it on CO3Dv2 and the landmarks.

**Label-free head, the method of record (P5 and P6, November 7, 2026; H200; `results/P4-vis*`, `results/P7-vis-imc/`, `results/P6-adapt-*/`, `results/P6-dump/`, `results/P3-stats.md`):** the same head trained on the support count of the backbone's own predicted depth and cameras, no ground truth anywhere (`<model>_oiavisself`), matches the ground-truth-trained head within 0.02 on all four datasets and three backbones (VGGT, Pi3X and Pi3: 0.992 / 0.988 / 0.989 on DTU, 0.93 / 0.94 / 0.93 on CO3Dv2, 0.98 / 0.96 / 0.95 on ETH3D, 0.93 / 0.94 / 0.93 on the landmarks), beats its own per-image control everywhere and its teacher, the reprojection test, on CO3Dv2 and the landmarks (Gate 4, pre-registered on November 6, passes). Controls added on the same rows: the distance to the image border and depth edges (at or below chance off DTU; the per-image control is indistinguishable from the border score there, so it had learned the frame edge), a probe on each VGGT layer (support becomes linearly decodable from layer 17 of 24; layer 4 carries only the per-image cue), two extra seeds (within 0.005), 20 views (heads 0.99). Trained on the target capture itself, still without labels, the head gains where the backbone's own geometry there is reliable (Pi3X on CO3Dv2 and ETH3D, +0.025 and +0.027) and loses where it is not (VGGT on CO3Dv2, -0.09). From the per-pixel scores (`scripts/analyze_dumps.py`): a threshold set on the other scenes of a dataset lets 1 to 20% of single-view pixels through against 57 to 83% for native confidence, and the fixed bands 0.3 / 0.7 mark 57 to 87% of pixels observed with 0.5 to 5% single-view inside. What the head cannot do: put back to the benchmark's filled patches it scores 0.62 to 0.75, below native confidence (`results/P6-vis-on-fill*/`). It certifies agreement between views, and a consistent invention has it; the paper says so.

**Fill-trained head, result (November 6, 2026; `results/P3-stats.md`):** the label-free support head (`<model>_oia`, `scripts/oia_head.py`) beats native confidence on every DTU cell (AUROC 0.99 to 1.00, CUR near 0 for VGGT and Pi3X; the pre-registered part 1 passes), but its per-image control, a head of the same size that sees only single-image features and can therefore only detect the patch, scores the same on every cell, including the held-out inpainted fill (part 2 fails). The linear probe and the cross-view transformer head also reach 1.00. The paper reports this as a negative result: a head trained on the benchmark's fills learns to see the fill, and the benchmark cannot certify a detector trained on its own labels without the per-image control. The reprojection-consistency detector is at chance (AUROC 0.49, 0.48); the perturbation-variance baseline is worse than native confidence. On CO3Dv2, which the heads never saw, the VGGT head scores 0.992 and its control 0.989 (both 0.31 above native; the 0.003 difference reaches p = 0.03, which meets the letter of the pre-registered criterion and not its purpose, and is not counted as a pass). The Pi3X CO3Dv2 control, completed on November 7, scores 0.988 against the head's 0.969: the control is the better fill detector.

**Not done** (tracked as GitHub issues, see below): the generative successor of the head (a discrete diffusion head whose K-sample disagreement marks imagined geometry, issue #1, now future work), continual adaptation with LoRA and replay (#5), and the qualitative heritage walkthrough with language queries (#6; the quantitative phototourism result above covers the heritage captures). The natural-occlusion validation of the masking proxy (#7) is the true-visibility protocol. The paper marks every one of these as `pending` or future work; no number is estimated.

## 5. Repository layout

```
scripts/
  hallu3d.py             the benchmark: DTU loader, 3D-consistent masking, fills, metrics,
                         model wrappers (VGGT, Pi3, Pi3X, DA3, MapAnything), statistics
  co3d.py                CO3Dv2 loader and preprocessing (camera conversion from PyTorch3D)
  kaggle_prelude.py      dependency setup that runs before numpy is imported (Kaggle)
  kaggle_driver.py       DTU runs            kaggle_driver_stress.py   stress tests
  kaggle_driver_co3d.py  CO3Dv2 runs
  test_synthetic.py      CPU test with oracle / blind fake models on a synthetic scene
  oia_head.py            the support head (P3-oia): label-free training on frozen backbone features, <model>_oia wrappers
  run_h100.py            the GPU plan for P3: check, train, dtu, co3d, stress, baselines, ablate, stats
  test_oia.py            CPU test of the support head with a fake backbone (plumbing, not the real hooks)
  test_co3d.py           CPU test of the CO3D path and consistency checks
  make_figures.py        every paper figure from the CSVs in results/
  make_demo.py           the demo video (needs the raw DTU / CO3D data)
  make_walkthrough.py    the end-to-end walkthrough video (needs no data: runs test_synthetic.py live, replays results/)
  make_site.py           docs/explorer.html (per-scene explorer) + docs/assets/hallubench3d_results.zip,
                         and a check that the numbers on the site equal the CSVs
  make_supp_archive.py   anonymised code + results archive for the supplementary
requirements.txt         CPU dependencies (numpy, opencv, scipy, pandas, matplotlib, pillow)
results/
  P1-full/ P2-full/      DTU, five models, full configuration
  P2-stress/             inpaint fill, 5 and 3 views, low parallax, distractor
  P2-co3d/ P2-co3d-b/    CO3Dv2, five models
  P3-pv-smoke/           perturbation-variance baseline
  table1_all_models.csv  merged DTU results      table3_all_models.csv  merged CO3Dv2 results
  (each run folder: CSV, config JSON, run log, Kaggle kernel log, example figures)
paper/
  tex/                   LaTeX source (CVPR 2027 author kit)   figures/   fig1 to fig9
  observe_imagine_abstain_cvpr27.pdf                            demo/      demo.mp4, walkthrough.mp4 + README
  draft.md               working draft and number ledger
docs/                    project website (GitHub Pages): index.html, explorer.html (generated), assets/
EXPERIMENTS.md           run registry, gates, how to push a Kaggle kernel
LOG.md                   dated log of every run, failure, fix and correction
PLAN.md                  the project brief: idea, rules, deadlines, phases, statistics protocol
```

## 6. Reproduce

```
pip install -r requirements.txt            # CPU side; model runs additionally need torch and the model repos
```

**Benchmark on real data (GPU).** Runs were done on Kaggle. Build a single-file kernel and push it:

```
cat scripts/kaggle_prelude.py scripts/hallu3d.py scripts/kaggle_driver.py > kernel/hallu3d_kaggle.py
# edit RUN_NAME / MODELS / CFG defaults at the top of the driver section, then
kaggle kernels push -p kernel        # needs kernel-metadata.json, see EXPERIMENTS.md
kaggle kernels output <user>/<kernel> -p results/<run>
```

DTU comes from the public Kaggle dataset `tobyzhou/dtu-training` (MVSNet layout). CO3Dv2 is preprocessed with `co3d.preprocess_co3d()` from the official single-sequence zips (public links in the CO3D repository, no form) into a small folder of JPEGs and `meta.npz` files.

**Run a model locally** (any CUDA GPU with 16 GB):

```python
import hallu3d as H
root = H.find_dtu_root(["/path/to/dtu"])
H.run_benchmark(root, ["vggt", "pi3x"], cfg={"seeds": [0], "fills": ["blur"]}, out_csv="out.csv")
print(H.summarize("out.csv"))
```

Model names: `vggt`, `pi3`, `pi3x`, `da3`, `mapanything`, `mapanything_mv` (multi-view consistency confidence), `<model>_pv<K>` (perturbation variance over K forwards), `<model>_reproj` (reprojection-consistency confidence from the model's own cameras; VGGT, Pi3, Pi3X; code written, not yet run on a GPU).

**CPU-only checks** (no model needed):

```
python scripts/test_synthetic.py            # exit 0 iff oracle scores AUROC 1 / CUR 0, blind 0.5 / 0.9, masks are
                                            # deterministic and disjoint, and the reprojection score behaves
python scripts/test_oia.py                  # exit 0 iff the support-head plumbing works end to end (fake backbone, CPU)
python scripts/run_h100.py all --dtu <dir with DTU incl. training scans> --co3d <co3d_prep>
                                            # GPU plan (H100 or L4): trains the heads, evaluates them and every baseline, writes results/P3-*/
python scripts/test_co3d.py <co3d_prep> <out>
```

**Paper, figures, video, site:**

```
python scripts/make_figures.py --dtu <dir with DTU> --co3d <co3d_prep>   # writes paper/figures/; without the data
                                                                          # flags, figs 3 to 9 (CSV-only) are still built
python scripts/make_demo.py --dtu <dir with DTU> --co3d <co3d_prep>      # writes paper/demo/demo.mp4 (needs ffmpeg)
python scripts/make_walkthrough.py          # writes paper/demo/walkthrough.mp4 (needs ffmpeg, no data; about 5 min to render)
python scripts/make_site.py                 # docs/explorer.html + docs/assets/hallubench3d_results.zip, checks index.html
python scripts/make_supp_archive.py         # supp/hallubench3d_supp_<date>.zip, anonymised (fails if a name survives)
cd paper/tex && tectonic main.tex           # paper + supplementary in one PDF (or pdflatex + bibtex)
cd paper/tex && tectonic supp.tex           # supplementary alone; both share preamble.tex and supp_body.tex
```

## 7. Rules the work followed

- No fabricated numbers. Every number in the paper and on the website maps to a CSV and a config file in `results/`. Anything not yet run is written as `pending`.
- Every run is reported, including the ones that failed and the one CO3D scene that narrowly missed a consistency guard. `LOG.md` records each correction made to a reported number.
- Hyperparameters were fixed on the DTU validation split and frozen before CO3Dv2 was touched.
- Baselines are compared at equal compute.
- Writing is plain: no hype words, no overclaiming about "world models" or "foundation models" of our own.

## 8. Open work

See the issue tracker. The items that matter most, in order: finish the support-head runs and decide Gate 2 (#15, #16); validate the masked-patch protocol against real occlusion (#7); a feed-forward Gaussian-splatting backbone on the benchmark and a third, building-scale dataset (#9); the heritage study (#6); the generative head (#1) and continual adaptation (#5) as future work.

## 9. Citation

Anonymous CVPR 2027 submission. If you use the benchmark before publication, please cite the repository.

```
@misc{hallubench3d2026,
  title  = {Observe, Imagine, Abstain: Detecting Hallucinated Geometry
            in Feed-Forward 3D Reconstruction},
  author = {Anonymous},
  year   = {2026},
  note   = {CVPR 2027 submission, code and results at this repository}
}
```

## 10. LLM usage

CVPR 2027 requires an LLM Usage Statement in every submission. Ours is the last section of `paper/tex/main.tex`: a general-purpose LLM assistant was used for parts of the code, literature lookup and text editing; the study design, hyperparameters, experiments, number verification and conclusions are the authors' own. Nothing in `results/` was produced by a language model.
