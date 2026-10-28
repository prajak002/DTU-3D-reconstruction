# Observe, Imagine, Abstain

Hallucination-aware 3D reconstruction. This repository holds **HalluBench-3D**, a benchmark that measures how confidently feed-forward 3D models report geometry that no camera saw, the logged results for five models on two datasets, the CVPR 2027 paper build, a demo video and the project website.

Project page: https://prajak002.github.io/DTU-3D-reconstruction/
Paper (PDF): `paper/observe_imagine_abstain_cvpr27.pdf`
Demo video: `paper/demo/demo.mp4`

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

**Not done** (tracked as GitHub issues, see below): the proposed detector (a discrete diffusion head over quantized geometry tokens whose K-sample disagreement labels pixels observed / imagined / abstain), its ablations, continual adaptation with LoRA and replay, the heritage photo study, and VGGT-Omega (gated checkpoint). The paper marks every one of these as `pending`; no number is estimated.

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
  test_co3d.py           CPU test of the CO3D path and consistency checks
  make_figures.py        every paper figure from the CSVs in results/
  make_demo.py           the demo video
results/
  P1-full/ P2-full/      DTU, five models, full configuration
  P2-stress/             inpaint fill, 5 and 3 views, low parallax, distractor
  P2-co3d/ P2-co3d-b/    CO3Dv2, five models
  P3-pv-smoke/           perturbation-variance baseline
  table1_all_models.csv  merged DTU results      table3_all_models.csv  merged CO3Dv2 results
  (each run folder: CSV, config JSON, run log, Kaggle kernel log, example figures)
paper/
  tex/                   LaTeX source (CVPR 2027 author kit)   figures/   fig1 to fig9
  observe_imagine_abstain_cvpr27.pdf                            demo/      demo.mp4 + README
  draft.md               working draft and number ledger
docs/                    project website (GitHub Pages)
EXPERIMENTS.md           run registry, gates, how to push a Kaggle kernel
LOG.md                   dated log of every run, failure, fix and correction
SKILL.md                 the project brief and the rules the work followed
```

## 6. Reproduce

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

**CPU-only checks** (no model needed):

```
python scripts/test_synthetic.py            # oracle must score AUROC 1 / CUR 0, blind 0.5 / 0.9
python scripts/test_co3d.py <co3d_prep> <out>
```

**Paper, figures, video:**

```
python scripts/make_figures.py              # writes paper/figures/
python scripts/make_demo.py                 # writes paper/demo/demo.mp4
cd paper/tex && tectonic main.tex           # or pdflatex
```

## 7. Rules the work followed

- No fabricated numbers. Every number in the paper and on the website maps to a CSV and a config file in `results/`. Anything not yet run is written as `pending`.
- Every run is reported, including the ones that failed and the one CO3D scene that narrowly missed a consistency guard. `LOG.md` records each correction made to a reported number.
- Hyperparameters were fixed on the DTU validation split and frozen before CO3Dv2 was touched.
- Baselines are compared at equal compute.
- Writing is plain: no hype words, no overclaiming about "world models" or "foundation models" of our own.

## 8. Open work

See the issue tracker. The items that matter most, in order: train the diffusion head and run the pre-registered Gate 2 (#1); validate the masked-patch protocol against real held-out-view occlusion (#7); the heritage study (#6); the faithful perturbation baseline (#3); scale to more scenes and a scene-level dataset (#9).

## 9. Citation

Anonymous CVPR 2027 submission. If you use the benchmark before publication, please cite the repository.

```
@misc{hallubench3d2026,
  title  = {Observe, Imagine, Abstain: Hallucination-Aware 3D Reconstruction
            with Discrete Diffusion and Continual Adaptation},
  author = {Anonymous},
  year   = {2026},
  note   = {CVPR 2027 submission, code and results at this repository}
}
```

## 10. Disclosure

Experiments, code and drafting were done with assistance from Claude (Anthropic). Claude is not an author. The disclosure required by the CVPR 2027 guidelines will be added to the submission.
