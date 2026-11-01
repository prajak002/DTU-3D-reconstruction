# HalluBench-3D demo video

`demo.mp4`: 1920x1080, 30 fps, H.264, about 3 minutes, no audio. Built by `scripts/make_demo.py`.

## What is live and what is replayed

**Part A (0:06 to 1:04) is live.** The script loads a real DTU scan (scan 3, MVSNet layout) and a
real CO3Dv2 scene (hydrant_167_18184_34441, preprocessed with `scripts/co3d.py`) and runs the
benchmark code on the CPU: `hallu3d.build_masks` places the sphere, `apply_fill` makes the four fills,
`consistency_check` produces the geometry numbers printed on screen. Those numbers are whatever the
code returns at build time; they are not typed in.

**Part B (1:04 to end) is a replay.** No GPU is available on the build machine, so model outputs are
the composites saved by the Kaggle runs on 2026-10-28 (`results/*/ex_*/*.png`) and the metrics are read
from the logged CSVs (`results/table1_all_models.csv`, `results/table3_all_models.csv`,
`results/P2-stress/P2-stress_inpaint10.csv`, `results/P2-stress/stress_vs_reference.csv`). Every
replayed frame carries the tag "model outputs recorded on Kaggle T4, 2026-10-28". The bar charts
recompute the cluster bootstrap CIs from the CSVs at build time.

The closing card lists what has not been run (diffusion head, continual adaptation, heritage study,
VGGT-Omega).

## Segments

| time | segment |
|---|---|
| 0:00 | title |
| 0:06 | Part A, DTU scan 3: views, sphere, hidden region, fills, labels, geometry check |
| 0:44 | Part A, CO3Dv2 hydrant: views, sphere, hidden region, geometry check |
| 1:04 | Part B, DTU: two scenes per model, five models |
| 1:44 | Part B, DTU inpaint stress test |
| 1:54 | Part B, CO3Dv2: two scenes per model |
| 2:26 | charts: CUR and AUROC on DTU vs CO3Dv2, stress tests |
| 2:45 | closing: measured vs not yet run |

## Regenerate

From the project root, with the benchmark virtualenv (numpy, opencv, pandas, matplotlib, pillow)
and ffmpeg at `/opt/homebrew/bin/ffmpeg`:

```
python scripts/make_demo.py --dtu <dir containing a DTU root> --co3d <preprocessed CO3D root> \
    --frames <scratch dir for PNG frames> --out paper/demo/demo.mp4
```

The DTU root is found by `hallu3d.find_dtu_root` (needs `Cameras/train`, `Rectified`, `Depths`).
The CO3D root is the output of `co3d.preprocess_co3d` (folders with `meta.npz`). Rendering takes
about two minutes on a laptop.

---

# End-to-end walkthrough (`walkthrough.mp4`)

`walkthrough.mp4`: 1920x1080, 30 fps, H.264, 5:10, no audio, MP4 chapter markers. Built by
`scripts/make_walkthrough.py`, which needs **no DTU or CO3D data**: it runs `scripts/test_synthetic.py` as a
subprocess and loads the synthetic DTU-format scan that the test writes, so Stages 1 and 2 are the real
benchmark code running at build time, on a synthetic scene (every such frame is tagged
"computed now, on this machine" or "live, synthetic scene"). Stages 3 and 4 replay the Kaggle runs under
`results/` (tagged "model outputs recorded on Kaggle T4, 2026-10-28"); the charts recompute the cluster
bootstrap intervals from the CSVs. Stage 6 lists the artefacts actually present on disk when the video was
built, with the git commit.

| time | chapter | what is on screen |
|---|---|---|
| 0:00 | title | plan of the video |
| 0:09 | the problem | paper figure 1, the definition, AUROC and CUR |
| 0:28 | protocol, live | `test_synthetic.py` output; views, sphere, hidden region, fills, labels, `consistency_check` |
| 1:15 | models and metrics, live | oracle and blind reference models through `run_benchmark`, `summarize` table, `reproj_support_score` |
| 1:57 | recorded runs, DTU | paper figure 2; five models, scans 17 and 106, with CSV rows |
| 2:45 | recorded runs, inpainted fill | VGGT, Pi3X, DA3 on scan 17 |
| 2:55 | recorded runs, CO3Dv2 | five models, toytrain and apple |
| 3:27 | the numbers | CUR and AUROC DTU vs CO3Dv2, stress tests, training-free detectors vs native confidence |
| 3:58 | paper and method | figures 4, 8, 9; measured vs pending |
| 4:32 | artefacts | command chain, the nine figures, files present on disk, where to look |

## Regenerate

From the project root, with the requirements installed and ffmpeg on PATH (or `$FFMPEG`):

```
python scripts/make_walkthrough.py [--out paper/demo/walkthrough.mp4] [--frames <scratch dir>]
```

Rendering takes a few minutes on a laptop (the bootstrap intervals dominate). Frames go to a fresh temp
directory unless `--frames` is given.
