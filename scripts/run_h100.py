"""GPU plan for the P3 runs: the support head (scripts/oia_head.py) against every baseline, on the benchmark.
Written for one H100; run on one L4 (24 GB) in November 2026, where every stage fits in memory (peak 9 GB).

Every stage is resumable (run_benchmark skips rows already in the CSV) and writes the same layout as the Kaggle
runs: results/<RUN>/<RUN>.csv, _cfg.json, _summary.csv, ex_<RUN>/ composites, <RUN>.log. Stages:

  check      import the backbones, one forward per backbone with the feature hooks, shapes and timing
  train      support heads. One joint run per backbone trains, on the same backbone forwards, the main conv head
             (vggt_oia), the linear probe (oialin), the per-image control (oiapatch), the cross-view transformer
             head (oiaattn) and two extra initialisation seeds of the main head (oia_s1, oia_s2); oiagt (masks
             placed with ground-truth depth) runs alone because its masked input differs; pi3x: oia + oiapatch
  dtu        P3-oia-dtu     full DTU configuration (18 scans x 5 seeds x 2 radii x 3 fills) for vggt_oia, vggt_oiafuse,
             vggt_oiapatch (the per-image control, paired on every row) and pi3x_oia
  co3d       P3-oia-co3d    full CO3Dv2 configuration; the heads never saw CO3D (the backbones did, see paper)
  stress     P3-oia-stress  inpaint10, blur5, blur3, adjacent10, distractor10 (same configs as P2-stress), with the
             per-image control on every row: inpaint is the fill the heads never saw and the decisive control
  baselines  P3-reproj (vggt_reproj, pi3x_reproj), P3-pv (vggt_pv4, pi3x_pv4), P3-native (vggt, pi3x rerun on this
             GPU) on the Table-4 keys (seed 0, blur, radius 0.18) so every detector is paired on identical rows
  pvfull     P3-pv-full     the fuller perturbation baseline (<base>_pvf8: photometric + blur + flip + view
             permutation, least-squares scale alignment, MAD spread) on the Table-4 keys plus noise, all bases
  ablate     P3-oia-abl     oialin, oiapatch, oiaattn, oiagt and the two extra head seeds on seed 0, radius 0.18, 3 fills
  dtuext     P3-dtuext      the 22 MVSNet test scans (never used anywhere else): native models, vggt_oia and the
             per-image control, seed 0, radius 0.18, 3 fills (the benchmark grows from 18 to 40 DTU scenes)
  stats      paired sign-flip tests of every detector against native confidence and of the head against its
             per-image control, Holm correction, pass condition, stress and CO3D tables (results/P3-stats.md)

Usage:
  python scripts/run_h100.py all   --dtu <dir with a DTU root incl. the 79 training scans> --co3d <preprocessed CO3D root>
  python scripts/run_h100.py train --dtu ...                 # or any single stage
Options: --out results  --ckpt checkpoints  --steps 1500  --seeds 0,1,2  --bases vggt,pi3x  --skip-gt
Estimated H100 time: check 2 min, train 25 min per head (7 heads, 3 h), dtu 30 min, co3d 30 min, stress 20 min,
baselines 25 min, ablate 20 min. The whole plan fits in one working day; 'dtu' alone answers the pass condition.

Pass condition (pre-registered in the paper): vggt_oia beats native confidence on AUROC and CUR with paired
sign-flip p < 0.05 over the 18 scans, at one backbone forward. The script reports the outcome either way."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
import co3d
import oia_head as O
try:                                   # feed-forward Gaussian splatting backbone (scripts/anysplat_wrap.py), optional
    import anysplat_wrap
    anysplat_wrap.register(H.MODEL_REGISTRY)
except Exception as _e:                # noqa: BLE001
    anysplat_wrap = None
    print(f"[run] anysplat backbone unavailable: {type(_e).__name__}: {_e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T4_KEYS = {"seeds": [0], "fills": ["blur"], "radius_fracs": [0.18]}
FULL = {"seeds": [0, 1, 2, 3, 4], "fills": ["gray", "blur", "noise"], "radius_fracs": [0.10, 0.18]}
STRESS = {
    "inpaint10": {"seeds": [0, 1], "fills": ["inpaint"], "radius_fracs": [0.18]},
    "blur5": {"seeds": [0, 1], "fills": ["blur"], "radius_fracs": [0.18], "n_views": 5},
    "blur3": {"seeds": [0, 1], "fills": ["blur"], "radius_fracs": [0.18], "n_views": 3},
    "adjacent10": {"seeds": [0, 1], "fills": ["blur"], "radius_fracs": [0.18], "view_mode": "adjacent"},
    "distractor10": {"seeds": [0, 1], "fills": ["blur"], "radius_fracs": [0.18], "distractor": True},
}


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(time.strftime("%H:%M:%S ") + s + "\n"); self.f.flush()


def run(args, run_name, models, cfg, root):
    d = os.path.join(args.out, run_name)
    os.makedirs(d, exist_ok=True)
    log = Log(os.path.join(d, f"{run_name}.log"))
    csv = os.path.join(d, f"{run_name}.csv")
    log(f"== {run_name}: models {models}, cfg {json.dumps({k: v for k, v in cfg.items() if k != 'scans'})}")
    for m in models:
        try:
            t0 = time.time()
            H.run_benchmark(root, [m], cfg=cfg, out_csv=csv, save_examples_dir=os.path.join(d, f"ex_{run_name}"))
            log(f"   {m}: {time.time() - t0:.0f}s")
        except Exception:
            log(f"!! {m} failed:\n{traceback.format_exc()}")
    if os.path.exists(csv):
        s = H.summarize(csv)
        s.to_csv(os.path.join(d, f"{run_name}_summary.csv"), index=False)
        log(s.to_string())
    return csv


def co3d_cfg(args, extra):
    scenes = co3d.list_co3d_scenes(args.co3d_root)
    return {**extra, "dataset": "co3d", "scans": scenes}


# ----------------------------------------------------------------------------- stages
def stage_check(args):
    import torch
    log = Log(os.path.join(args.out, "P3-check", "P3-check.log"))
    log("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    data = H.load_scan(args.dtu_root, 3, H.select_views(10), 3)
    for base in args.bases:
        try:
            w = H.MODEL_REGISTRY[base]()
            x = w._prep(data["imgs"], H.DEFAULT_CFG["model_hw"])
            for feats in ("agg", "patch"):
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    d, c, F, K, E = O.FORWARD_FEATS[base](w, x, w.dtype, feats)
                torch.cuda.synchronize()
                log(f"{base} feats={feats}: depth {tuple(d.shape)} conf {tuple(c.shape)} feature maps "
                    f"{[tuple(f.shape) for f in F]} K[0] fx={K[0, 0, 0]:.1f} {time.time() - t0:.2f}s "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            # the model's cameras vs ground truth: reprojection agreement of the model's own depth
            md = O.model_res_data(data, H.DEFAULT_CFG["model_hw"], (d.cpu().numpy(), c.cpu().numpy(), K.cpu().numpy(), E.cpu().numpy()))
            m = H.build_masks(md, 0, 0.18, {**H.DEFAULT_CFG, "dilate_px": 3})
            chk = H.consistency_check(md, m)
            log(f"{base} self-consistency of predicted depth + cameras: {chk}  (U_transfer must be high, > 0.9)")
            w.free()
        except Exception:
            log(f"!! {base} check failed:\n{traceback.format_exc()}")
    train_scans = [s for s in O.DTU_TRAIN_SCANS if os.path.isdir(os.path.join(args.dtu_root, "Rectified", f"scan{s}_train"))]
    log(f"DTU training scans present: {len(train_scans)} of {len(O.DTU_TRAIN_SCANS)}")
    if args.co3d_root:
        log(f"CO3D scenes: {len(co3d.list_co3d_scenes(args.co3d_root))}")


def stage_train(args):
    extra = [s for s in args.seeds if s != 0]
    jobs = []            # (base, variants, extra seeds, first checkpoint that marks the job done)
    if "vggt" in args.bases:
        jobs.append(("vggt", args.variants, extra, "vggt_oia.pt"))
        if not args.skip_gt:
            jobs.append(("vggt", "oiagt", [], "vggt_oiagt.pt"))
    for b in args.bases:
        if b != "vggt":
            jobs.append((b, "oia,oiapatch", [], f"{b}_oia.pt"))
    for base, variants, seeds, marker in jobs:
        mp = os.path.join(args.ckpt, marker)
        if os.path.exists(mp) and not args.retrain:
            import torch
            st = int(torch.load(mp, map_location="cpu", weights_only=False).get("step", 0))
            if st >= args.steps:
                print(f"[train] {marker} exists at step {st}, skipping {base} {variants}"); continue
            print(f"[train] {marker} exists at step {st} < {args.steps}: resuming {base} {variants}")
        # one subprocess per job so that GPU memory is released between backbones
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "oia_head.py"), "train", "--base", base,
               "--variant", variants, "--dtu", args.dtu_root, "--out", args.ckpt, "--steps", str(args.steps), "--seed", "0"]
        if "," not in variants:
            cmd[cmd.index("--out") + 1] = os.path.join(args.ckpt, marker)
        if seeds:
            cmd += ["--extra-seeds", ",".join(str(x) for x in seeds)]
        if os.path.exists(mp) and not args.retrain:
            cmd.append("--resume")
        print("[train]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


def _ckpt_env(args):
    os.environ["OIA_CKPT_DIR"] = args.ckpt


def _have(args, name):
    """True when the head checkpoint exists and its training is complete (step >= --steps). A head that is still
    training (the Pi3X control was at step 750 when the 2026-11-06 ablation stage picked it up) is not evaluated."""
    p = os.path.join(args.ckpt, f"{name}.pt")
    if not os.path.exists(p):
        return False
    import torch
    st = int(torch.load(p, map_location="cpu", weights_only=False).get("step", 0))
    if st < args.steps:
        print(f"[skip] {name}.pt is at step {st} < {args.steps}; not evaluated until its training is complete", flush=True)
        return False
    return True


def _head_models(args, with_fuse=True):
    """The head models that have a checkpoint: <base>_oia, vggt_oiafuse, and the per-image control <base>_oiapatch."""
    ms = []
    for b in args.bases:
        if _have(args, f"{b}_oia"):
            ms.append(f"{b}_oia")
            if b == "vggt" and with_fuse:
                ms.append("vggt_oiafuse")
        if _have(args, f"{b}_oiapatch"):
            ms.append(f"{b}_oiapatch")
    return ms


def stage_dtu(args):
    _ckpt_env(args)
    run(args, "P3-oia-dtu", _head_models(args), FULL, args.dtu_root)


def stage_co3d(args):
    _ckpt_env(args)
    run(args, "P3-oia-co3d", _head_models(args), co3d_cfg(args, FULL), args.co3d_root)


def stage_stress(args):
    _ckpt_env(args)
    models = _head_models(args, with_fuse=False)
    for test, extra in STRESS.items():
        run(args, f"P3-oia-stress_{test}", models, {**extra, "keep_model": True}, args.dtu_root)


def stage_baselines(args):
    _ckpt_env(args)
    run(args, "P3-reproj", [f"{b}_reproj" for b in args.bases if b in ("vggt", "pi3x", "pi3")], T4_KEYS, args.dtu_root)
    run(args, "P3-pv", [f"{b}_pv4" for b in args.bases], T4_KEYS, args.dtu_root)
    run(args, "P3-native", list(args.bases), T4_KEYS, args.dtu_root)


def stage_pvfull(args):
    cfg = {"seeds": [0], "fills": ["blur", "noise"], "radius_fracs": [0.18]}
    run(args, "P3-pv-full", [f"{b}_pvf8" for b in args.pv_bases], cfg, args.dtu_root)


def stage_dtuext(args):
    _ckpt_env(args)
    cfg = {"seeds": [0], "fills": ["gray", "blur", "noise"], "radius_fracs": [0.18], "scans": H.DTU_TEST_SCANS}
    models = list(args.ext_bases) + [m for m in ("vggt_oia", "vggt_oiapatch") if _have(args, m)]
    run(args, "P3-dtuext", models, cfg, args.dtu_root)


# ETH3D scenes that pass the CO3D consistency gate (reproj_agree_3pct_vis > 0.9 and U_transfer > 0.85 on the ten
# co-visible views, results/P3-eth3d-check/consistency_co3d.json, 2026-11-04). Excluded: pipes (U_transfer 0.58),
# playground (0.75), delivery_area (too few hidden pixels, U_transfer undefined), office (a depth file was truncated
# during download and the raw archive was gone before it was noticed).
ETH3D_SCENES = ["courtyard", "electro", "facade", "kicker", "meadow", "relief", "relief_2", "terrace", "terrains"]


def stage_eth3d(args):
    """Third dataset: ETH3D DSLR training scenes (scripts/eth3d.py), building scale, laser ground truth.
    Native models plus the head and its per-image control, full configuration, cfg dataset = co3d (same loader)."""
    _ckpt_env(args)
    if not args.eth3d_root:
        print("[skip] eth3d: no --eth3d root"); return
    scenes = [s for s in ETH3D_SCENES if s in co3d.list_co3d_scenes(args.eth3d_root)]
    models = list(args.ext_bases) + [m for m in ("vggt_oia", "vggt_oiapatch") if _have(args, m)]
    # frames are stored in co-visibility order (eth3d.py --reorder); "adjacent" takes the reference + 9 most overlapping
    run(args, "P3-eth3d", models, {**FULL, "dataset": "co3d", "scans": scenes, "view_mode": "adjacent"}, args.eth3d_root)


# ----------------------------------------------------------------------------- P4-vis: true-visibility protocol
VIS_CFG = {"vis_views": [3, 5, 10], "seeds": [0, 1, 2]}


def _vis_models(args, extra_native=()):
    """Every detector on the vis protocol: native confidence, reprojection, the two hand-crafted controls (image
    border, depth edges), the fill-trained head and its control (do they transfer?), the visibility-trained head,
    its per-image control and the linear probe, the label-free head and the adapted heads."""
    ms = []
    for b in args.bases:
        ms += [b] + ([f"{b}_reproj"] if b in ("vggt", "pi3x", "pi3", "da3") else []) + [f"{b}_{k}" for k in HEURISTICS]
        ms += [f"{b}_{v}" for v in VIS_DETECTORS if _have(args, f"{b}_{v}")]
    return ms + [m for m in extra_native if m not in ms]


HEURISTICS = ("border", "dedge")      # hallu3d.HeuristicWrap: training-free controls on the base model's own output


# every head variant scored on the vis protocol when its checkpoint is complete: the fill-trained head and control
# (transfer), the visibility-trained head, control and probe with two extra seeds of the head, the label-free
# visibility head with its control and probe, and the per-layer probes (mechanism)
VIS_DETECTORS = ("oia", "oiapatch", "oiavis", "oiavispatch", "oiavislin", "oiavis_s1", "oiavis_s2",
                 "oiavisself", "oiavisselfpatch", "oiavisselflin",
                 "oiavislin4", "oiavislin11", "oiavislin17", "oiavislin23", "oiavispatchlin")
# the provenance heads are scored by stage prov (their own run family, with the attribution metrics)


def _train_heads(args, base, variants, marker, extra_seeds=(), extra_of=None, root=None, scans=None, tag=None):
    """One joint training run (one subprocess) of several heads with the same label source; `marker` is the variant
    whose checkpoint decides skip / resume. root / scans / tag: train on another root (a CO3D-format folder) on the
    given scenes, saving <base>_<variant>_x<tag>.pt (label-free adaptation, stages adapt_co3d / adapt_eth3d)."""
    suffix = f"_x{tag}" if tag else ""
    mp = os.path.join(args.ckpt, f"{base}_{marker}{suffix}.pt")
    if os.path.exists(mp) and not args.retrain:
        import torch
        st = int(torch.load(mp, map_location="cpu", weights_only=False).get("step", 0))
        if st >= args.steps:
            print(f"[train] {base}_{marker}{suffix}.pt exists at step {st}, skipping {variants}"); return
        print(f"[train] {base}_{marker}{suffix}.pt exists at step {st} < {args.steps}: resuming {variants}")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "oia_head.py"), "train", "--base", base,
           "--variant", ",".join(variants), "--dtu", root or args.dtu_root, "--out", args.ckpt,
           "--steps", str(args.steps), "--seed", "0"]
    if scans:
        cmd += ["--scans", ",".join(str(s) for s in scans)]
    if tag:
        cmd += ["--tag", tag]
    if extra_seeds:
        cmd += ["--extra-seeds", ",".join(str(x) for x in extra_seeds)]
        if extra_of:
            cmd += ["--extra-seeds-of", extra_of]
    if os.path.exists(mp) and not args.retrain:
        cmd.append("--resume")
    print("[train]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def stage_trainvisself(args):
    """Label-free visibility heads (P5-visself): oiavisself, oiavisselfpatch, oiavisselflin jointly, labels from the
    support count of the backbone's own prediction. Same cost as trainvis (one forward per step)."""
    for base in args.bases:
        _train_heads(args, base, ["oiavisself", "oiavisselfpatch", "oiavisselflin"], "oiavisself")


def stage_trainprov(args):
    """Provenance heads (P10-prov): <base>_oiaprov on the ground-truth per-view visibility bits and <base>_oiaprovself
    on the per-view support of the backbone's own prediction (label-free). Different label sources, so one run each
    per backbone; same cost per step as the other visibility heads plus the pair attention."""
    for base in args.bases:
        _train_heads(args, base, ["oiaprov"], "oiaprov")
        _train_heads(args, base, ["oiaprovself"], "oiaprovself")


def stage_prov(args):
    """P10-prov: the provenance heads and the reprojection attribution baseline on the vis protocol of every dataset
    given (DTU, CO3Dv2, ETH3D, phototourism), with the per-view attribution metrics (hallu3d.provenance_metrics):
    results/P10-prov-<dataset>/P10-prov-<dataset>.csv. The AUROC / CUR columns of these rows are the support score of
    the heads on the same cells as P4-vis*, so the heads join the vis table; prov_auroc, prov_top1 and prov_count_mae
    are the attribution columns. One scene per dataset (first scan, seed 0, 5 views) is also dumped with its per-view
    maps for the qualitative figure (results/P10-prov-dump/)."""
    _ckpt_env(args)
    heads = [f"{b}_{v}" for b in args.bases for v in ("oiaprov", "oiaprovself") if _have(args, f"{b}_{v}")]
    if not heads:
        print("[skip] prov: no provenance head checkpoint is complete"); return
    models = heads + [f"{b}_reproj" for b in args.bases if b in ("vggt", "pi3x", "pi3", "da3")]
    jobs = [("dtu", args.dtu_root, dict(VIS_CFG))]
    if args.co3d_root:
        jobs.append(("co3d", args.co3d_root, co3d_cfg(args, VIS_CFG)))
    if args.eth3d_root:
        jobs.append(("eth3d", args.eth3d_root, _eth3d_cfg(args, VIS_CFG)))
    if args.imc_root:
        jobs.append(("imc", args.imc_root, _imc_cfg(args, VIS_CFG)))
    dump = os.path.join(args.out, "P10-prov-dump")
    for tag, root, cfg in jobs:
        _run_vis(args, f"P10-prov-{tag}", models, {**cfg, "prov": True}, root)
        scans = cfg.get("scans", H.DEFAULT_CFG["scans"])
        small = {**cfg, "prov": True, "scans": list(scans)[:2], "seeds": [0], "vis_views": [5], "dump_pairs": os.path.join(dump, tag)}
        _run_vis(args, f"P10-prov-dump-{tag}", [m for m in models if m.endswith("provself") or m.endswith("reproj")], small, root)


def stage_trainvisprobe(args):
    """Mechanism and seeds: per-layer linear probes with true-visibility labels (VGGT: aggregator layers 4/11/17/23;
    every backbone: the per-image encoder probe) plus two extra initialisation seeds of the visibility head, all in
    one run per backbone."""
    for base in args.bases:
        probes = (["oiavislin4", "oiavislin11", "oiavislin17", "oiavislin23"] if base == "vggt" else []) + ["oiavispatchlin"]
        _train_heads(args, base, probes, probes[0], extra_seeds=(1, 2), extra_of="oiavis")


def stage_trainvis(args):
    """Heads trained on true-visibility labels (oiavis, oiavispatch, oiavislin jointly), one subprocess per backbone."""
    for base in args.bases:
        mp = os.path.join(args.ckpt, f"{base}_oiavis.pt")
        if os.path.exists(mp) and not args.retrain:
            import torch
            st = int(torch.load(mp, map_location="cpu", weights_only=False).get("step", 0))
            if st >= args.steps:
                print(f"[trainvis] {base}_oiavis.pt exists at step {st}, skipping"); continue
            print(f"[trainvis] {base}_oiavis.pt exists at step {st} < {args.steps}: resuming")
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "oia_head.py"), "train", "--base", base,
               "--variant", "oiavis,oiavispatch,oiavislin", "--dtu", args.dtu_root, "--out", args.ckpt,
               "--steps", str(args.steps), "--seed", "0"]
        if os.path.exists(mp) and not args.retrain:
            cmd.append("--resume")
        print("[trainvis]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


def _run_vis(args, run_name, models, cfg, root):
    d = os.path.join(args.out, run_name)
    os.makedirs(d, exist_ok=True)
    log = Log(os.path.join(d, f"{run_name}.log"))
    csv = os.path.join(d, f"{run_name}.csv")
    log(f"== {run_name}: models {models}, cfg {json.dumps({k: v for k, v in cfg.items() if k != 'scans'})}")
    for m in models:
        try:
            t0 = time.time()
            H.run_vis(root, [m], cfg=cfg, out_csv=csv)
            log(f"   {m}: {time.time() - t0:.0f}s")
        except Exception:
            log(f"!! {m} failed:\n{traceback.format_exc()}")
    if os.path.exists(csv):
        import pandas as pd
        x = pd.read_csv(csv)
        x = x[~x.get("skipped", False).astype(bool)] if "skipped" in x else x
        tmp = csv.replace(".csv", "_rows.csv"); x.to_csv(tmp, index=False)
        H.summarize(tmp).to_csv(os.path.join(d, f"{run_name}_summary.csv"), index=False)
        os.remove(tmp)
    return csv


def stage_vis(args):
    """P4-vis on the 18 DTU benchmark scans: every detector, 3 / 5 / 10 clean views, three view-set rotations."""
    _ckpt_env(args)
    _run_vis(args, "P4-vis", _vis_models(args, args.ext_bases), VIS_CFG, args.dtu_root)


def stage_vis_co3d(args):
    _ckpt_env(args)
    _run_vis(args, "P4-vis-co3d", _vis_models(args, [b for b in args.ext_bases if b not in args.bases]),
             co3d_cfg(args, VIS_CFG), args.co3d_root)


def _eth3d_cfg(args, extra):
    scenes = [s for s in ETH3D_SCENES if s in co3d.list_co3d_scenes(args.eth3d_root)]
    return {**extra, "dataset": "co3d", "scans": scenes, "view_mode": "adjacent"}


def stage_vis_eth3d(args):
    _ckpt_env(args)
    if not args.eth3d_root:
        print("[skip] vis_eth3d: no --eth3d root"); return
    _run_vis(args, "P4-vis-eth3d", _vis_models(args, [b for b in args.ext_bases if b not in args.bases]),
             _eth3d_cfg(args, VIS_CFG), args.eth3d_root)


# ----------------------------------------------------------------------------- P6: what a reviewer would ask next
def _co3d_folds(scenes):
    """Two folds of the 18 CO3Dv2 scenes by sequence: A = the first sequence of every category, B = the second, so a
    head trained on A has never seen a frame of any B scene (same categories, different objects and captures)."""
    by_cat = {}
    for s in sorted(scenes):
        by_cat.setdefault(s.split("_")[0], []).append(s)
    folds = {"co3dA": [], "co3dB": []}
    for cat, seqs in by_cat.items():
        folds["co3dA"].append(seqs[0])
        folds["co3dB"] += seqs[1:2]
    return folds


ADAPT_VARIANTS = ["oiavisself", "oiavisselfpatch"]   # the label-free head and its per-image control


def stage_adapt_co3d(args):
    """Label-free adaptation to phone captures (P6-adapt-co3d). Per backbone and fold, the label-free visibility head
    and its per-image control are trained on the 9 scenes of one CO3Dv2 fold with labels from the backbone's OWN
    prediction (no ground truth, no DTU), then scored on the 9 scenes of the other fold on the vis protocol. The
    rows carry the fold in the model name (<base>_oiavisself_xco3dA was trained on fold A and is scored on fold B);
    stats pools the two as <base>_oiavisself_xco3d: every scene scored by a head that never saw it."""
    _ckpt_env(args)
    folds = _co3d_folds(co3d.list_co3d_scenes(args.co3d_root))
    for base in args.bases:
        for tag, train_scenes in folds.items():
            _train_heads(args, base, ADAPT_VARIANTS, "oiavisself", root=args.co3d_root, scans=train_scenes, tag=tag)
    for tag, test_tag in (("co3dA", "co3dB"), ("co3dB", "co3dA")):
        models = [f"{b}_{v}_x{tag}" for b in args.bases for v in ADAPT_VARIANTS if _have(args, f"{b}_{v}_x{tag}")]
        if models:
            _run_vis(args, "P6-adapt-co3d", models, {**co3d_cfg(args, VIS_CFG), "scans": folds[test_tag]}, args.co3d_root)


ETH3D_ADAPT_SCENES = ["delivery_area", "office", "pipes", "playground"]   # the four scenes outside the benchmark set


def stage_adapt_eth3d(args):
    """Label-free adaptation to building-scale scenes (P6-adapt-eth3d): heads trained, without ground truth, on the
    four ETH3D scenes that are not in the benchmark set (they failed the consistency gate as evaluation scenes but
    serve as unlabelled captures), scored on the nine benchmark scenes."""
    _ckpt_env(args)
    if not args.eth3d_root:
        print("[skip] adapt_eth3d: no --eth3d root"); return
    train_scenes = [s for s in ETH3D_ADAPT_SCENES if s in co3d.list_co3d_scenes(args.eth3d_root)]
    if not train_scenes:
        print("[skip] adapt_eth3d: none of the adaptation scenes is preprocessed"); return
    for base in args.bases:
        _train_heads(args, base, ADAPT_VARIANTS, "oiavisself", root=args.eth3d_root, scans=train_scenes, tag="eth3d")
    models = [f"{b}_{v}_xeth3d" for b in args.bases for v in ADAPT_VARIANTS if _have(args, f"{b}_{v}_xeth3d")]
    if models:
        _run_vis(args, "P6-adapt-eth3d", models, _eth3d_cfg(args, VIS_CFG), args.eth3d_root)


def stage_vis_on_fill(args):
    """The benchmark's own question put to the visibility detectors (P6-vis-on-fill): the heads trained on single-view
    support (ground-truth labels and label-free), their per-image controls, the probe and the two hand-crafted
    controls, scored zero-shot on the filled patches of the masking benchmark (Table-4 keys on DTU; blur and noise
    at radius 0.18, seed 0, on CO3Dv2), paired with the native rows of P3-native / table3. The benchmark defines
    unsupported as "visible in no view"; the heads were trained on "visible in one view only". Whether a detector of
    the second fires on the first is a result either way. Also the full-protocol rows of MapAnything's multi-view
    confidence (mapanything_mv), which the paper quoted from a smoke run."""
    _ckpt_env(args)
    models = []
    for b in args.bases:
        models += [f"{b}_{k}" for k in HEURISTICS]
        models += [f"{b}_{v}" for v in ("oiavis", "oiavisself", "oiavispatch", "oiavisselfpatch", "oiavislin")
                   if _have(args, f"{b}_{v}")]
    run(args, "P6-vis-on-fill", models, T4_KEYS, args.dtu_root)
    if args.co3d_root:
        run(args, "P6-vis-on-fill-co3d", models,
            co3d_cfg(args, {"seeds": [0], "fills": ["blur", "noise"], "radius_fracs": [0.18]}), args.co3d_root)
    if "mapanything_mv" in H.MODEL_REGISTRY:
        run(args, "P6-mapanything-mv", ["mapanything_mv"], T4_KEYS, args.dtu_root)


# ----------------------------------------------------------------------------- P7: in-the-wild heritage photo collections
IMC_SCENES = ["british_museum", "florence_cathedral_side", "lincoln_memorial_statue", "london_bridge", "milan_cathedral",
              "mount_rushmore", "piazza_san_marco", "sagrada_familia", "st_pauls_cathedral"]   # all nine pass the gate (LOG 2026-11-07)


def _imc_cfg(args, extra):
    scenes = [s for s in IMC_SCENES if s in co3d.list_co3d_scenes(args.imc_root)]
    return {**extra, "dataset": "co3d", "scans": scenes, "view_mode": "adjacent"}


def stage_vis_imc(args):
    """P7-vis-imc: the vis protocol on nine phototourism landmarks (tourist photos, COLMAP depth), every detector."""
    _ckpt_env(args)
    if not args.imc_root:
        print("[skip] vis_imc: no --imc root"); return
    _run_vis(args, "P7-vis-imc", _vis_models(args, [b for b in args.ext_bases if b not in args.bases]),
             _imc_cfg(args, VIS_CFG), args.imc_root)


def stage_adapt_imc(args):
    """P7-adapt-imc: label-free adaptation to tourist photos. Fold A = the first five landmarks (sorted), fold B = the
    other four; the label-free head and its control are trained on one fold without ground truth and scored on the
    other; stats pools the folds as <base>_oiavisself_ximc."""
    _ckpt_env(args)
    if not args.imc_root:
        print("[skip] adapt_imc: no --imc root"); return
    scenes = [s for s in IMC_SCENES if s in co3d.list_co3d_scenes(args.imc_root)]
    folds = {"imcA": scenes[:5], "imcB": scenes[5:]}
    for base in args.bases:
        for tag, train_scenes in folds.items():
            _train_heads(args, base, ADAPT_VARIANTS, "oiavisself", root=args.imc_root, scans=train_scenes, tag=tag)
    for tag, test_tag in (("imcA", "imcB"), ("imcB", "imcA")):
        models = [f"{b}_{v}_x{tag}" for b in args.bases for v in ADAPT_VARIANTS if _have(args, f"{b}_{v}_x{tag}")]
        if models:
            _run_vis(args, "P7-adapt-imc", models, {**_imc_cfg(args, VIS_CFG), "scans": folds[test_tag]}, args.imc_root)


# ----------------------------------------------------------------------------- P8: cross-model disagreement
XMODEL_PAIRS = ["vggt_xpi3x", "vggt_xpi3", "pi3x_xpi3", "vggt_xda3", "pi3x_xda3", "vggt_xmapanything"]


def stage_xmodel(args):
    """P8-xmodel: do two backbones invent the same surface? Cross-model disagreement (hallu3d.CrossModelWrap) scored
    on the benchmark's filled patches (DTU Table-4 keys; CO3Dv2 blur and noise, seed 0, radius 0.18) and, as a
    control, on real single-view geometry (the DTU vis protocol). Pairs that need a backbone that is not installed
    fail and are logged."""
    pairs = [p for p in XMODEL_PAIRS if all(b in H.MODEL_REGISTRY for b in p.split("_x"))]
    run(args, "P8-xmodel", pairs, T4_KEYS, args.dtu_root)
    if args.co3d_root:
        run(args, "P8-xmodel-co3d", pairs, co3d_cfg(args, {"seeds": [0], "fills": ["blur", "noise"], "radius_fracs": [0.18]}), args.co3d_root)
    _run_vis(args, "P8-xmodel-vis", pairs[:3], VIS_CFG, args.dtu_root)


MAIN_DETECTORS = ("reproj", "border", "dedge", "oiavis", "oiavispatch", "oiavisself", "oiavisselfpatch")


def _main_models(args):
    ms = []
    for b in args.bases:
        ms += [b] + [f"{b}_{v}" for v in MAIN_DETECTORS if v in ("reproj", "border", "dedge") or _have(args, f"{b}_{v}")]
    return ms


def stage_vis_ext(args):
    """View counts outside the training range (P4-vis-ext): 2 and 20 clean DTU views (the heads were trained on 3,
    5 and 10). Native confidence, reprojection, the hand-crafted controls, the visibility heads and the controls."""
    _ckpt_env(args)
    _run_vis(args, "P4-vis-ext", _main_models(args), {"vis_views": [2, 20], "seeds": [0, 1, 2]}, args.dtu_root)


def stage_dump(args):
    """Per-pixel scores of the main detectors on the vis protocol (P6-dump): DTU, CO3Dv2 and ETH3D, written under
    results/P6-dump/<dataset>/ for scripts/analyze_dumps.py (risk-coverage curves, calibration, precision at the
    operating point). The rows duplicate P4-vis* and are kept apart."""
    _ckpt_env(args)
    models = _main_models(args)
    jobs = [("dtu", VIS_CFG, args.dtu_root)]
    if args.co3d_root:
        jobs.append(("co3d", co3d_cfg(args, VIS_CFG), args.co3d_root))
    if args.eth3d_root:
        jobs.append(("eth3d", _eth3d_cfg(args, VIS_CFG), args.eth3d_root))
    if args.imc_root:
        jobs.append(("imc", _imc_cfg(args, VIS_CFG), args.imc_root))
    jobs = [j for j in jobs if j[0] in args.dump_datasets.split(",")]
    for name, cfg, root in jobs:
        dd = os.path.join(args.out, "P6-dump", name)
        _run_vis(args, f"P6-dump-{name}", models, {**cfg, "dump_dir": dd}, root)


def stage_gs(args):
    """Sixth backbone: AnySplat, feed-forward 3D Gaussian splatting from unposed views (P3-gs). 'anysplat' scores the
    opacity of the pixel-aligned Gaussian as confidence; 'anysplat_render' (needs gsplat) scores the rendered
    accumulated alpha. Full DTU configuration, then CO3Dv2 and the Table-4 keys with the reprojection detector."""
    if anysplat_wrap is None:
        print("[skip] gs: anysplat_wrap could not be imported"); return
    models = ["anysplat"] + (["anysplat_render"] if args.gs_render else [])
    run(args, "P3-gs", models, FULL, args.dtu_root)
    run(args, "P3-gs-t4", ["anysplat_reproj"], T4_KEYS, args.dtu_root)
    if args.co3d_root:
        run(args, "P3-gs-co3d", models, co3d_cfg(args, FULL), args.co3d_root)


def stage_ablate(args):
    _ckpt_env(args)
    models = [m for m in ("vggt_oialin", "vggt_oiapatch", "vggt_oiaattn", "vggt_oiagt", "pi3x_oiapatch") if _have(args, m)]
    cfg = {"seeds": [0], "fills": ["gray", "blur", "noise"], "radius_fracs": [0.18]}
    run(args, "P3-oia-abl", models, cfg, args.dtu_root)
    # extra initialisation seeds of the main head: evaluated through the same wrapper, rows renamed vggt_oia_s<seed>
    import pandas as pd
    key = ["model", "scan", "seed", "radius_frac", "fill"]
    for s in args.seeds[1:]:
        ck = os.path.join(args.ckpt, f"vggt_oia_s{s}.pt")
        if not os.path.exists(ck):
            continue
        os.environ["OIA_CKPT_VGGT_OIA"] = ck
        csv = os.path.join(args.out, f"P3-oia-abl-seed{s}", f"P3-oia-abl-seed{s}.csv")
        if os.path.exists(csv):   # rows were renamed after the previous call; rename back so that the run resumes
            df = pd.read_csv(csv); df["model"] = "vggt_oia"
            df.drop_duplicates(subset=key).to_csv(csv, index=False)
        csv = run(args, f"P3-oia-abl-seed{s}", ["vggt_oia"], cfg, args.dtu_root)
        if os.path.exists(csv):
            df = pd.read_csv(csv); df["model"] = f"vggt_oia_s{s}"
            df.drop_duplicates(subset=key).to_csv(csv, index=False)
            H.summarize(csv).to_csv(csv.replace(".csv", "_summary.csv"), index=False)
    os.environ.pop("OIA_CKPT_VGGT_OIA", None)


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m, running = {}, len(items), 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def stage_stats(args):
    import pandas as pd
    frames = []
    for run_name in ["P3-oia-dtu", "P3-reproj", "P3-pv", "P3-pv-full", "P3-native", "P3-oia-abl", "P3-dtuext", "P3-gs", "P3-gs-t4",
                     "P6-vis-on-fill", "P6-mapanything-mv", "P8-xmodel"] + [f"P3-oia-abl-seed{s}" for s in args.seeds[1:]]:
        p = os.path.join(args.out, run_name, f"{run_name}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    # native rows from the recorded full runs, for the full-configuration comparison
    for p in [os.path.join(ROOT, "results", "table1_all_models.csv")]:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print("no P3 results yet"); return
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model", "scan", "seed", "radius_frac", "fill"])
    bench = [str(x) for x in H.DEFAULT_CFG["scans"]]      # the 18 benchmark scans; the dtuext rows share the keys
    t4 = df[(df.seed == 0) & (df.fill == "blur") & (df.radius_frac == 0.18) & df.scan.astype(str).isin(bench)]
    lines = ["# P3 statistics (generated by scripts/run_h100.py stats)", "",
             "## Table 4 keys: 18 DTU scans, seed 0, blur, radius 0.18", "",
             "| detector | n scans | AUROC | CUR | CUR-err | p AUROC vs native | p CUR vs native |", "|---|---|---|---|---|---|---|"]
    pv = {}
    for base in args.bases:
        dets = [m for m in sorted(t4.model.unique()) if m == base or m.startswith(base + "_")]
        for m in dets:
            d = t4[t4.model == m]
            if d.empty:
                continue
            ci = {c: H.cluster_bootstrap_ci(d, c, n_boot=10000) for c in ("auroc", "cur", "cur_err")}
            pa = pc = float("nan")
            if m != base and base in t4.model.values:
                pa = H.paired_permutation(t4, "auroc", m, base)[1]
                pc = H.paired_permutation(t4, "cur", m, base)[1]
                pv[f"{m}:auroc"], pv[f"{m}:cur"] = pa, pc
            f = lambda c: f"{ci[c][0]:.3f} [{ci[c][1]:.3f}, {ci[c][2]:.3f}]"
            lines.append(f"| {m} | {d.scan.nunique()} | {f('auroc')} | {f('cur')} | {f('cur_err')} | {pa:.4g} | {pc:.4g} |")
    if pv:
        adj = holm(pv)
        lines += ["", "Holm-adjusted p-values over all detector-vs-native tests: " +
                  ", ".join(f"{k} {v:.4g}" for k, v in sorted(adj.items()))]
    # the head against its per-image control, per fill and on the stress tests (inpaint is the unseen fill)
    frames_s = []
    for test in STRESS:
        p = os.path.join(args.out, f"P3-oia-stress_{test}", f"P3-oia-stress_{test}.csv")
        if os.path.exists(p):
            x = pd.read_csv(p); x["test"] = test; frames_s.append(x)
    for test in STRESS:                                      # native rows of the same tests (P2-stress, one CSV per test)
        ref = os.path.join(ROOT, "results", "P2-stress", f"P2-stress_{test}.csv")
        if os.path.exists(ref) and frames_s:
            x = pd.read_csv(ref); x["test"] = test; frames_s.append(x)
    for base in args.bases:
        head, ctrl = f"{base}_oia", f"{base}_oiapatch"
        for fill in ("gray", "blur", "noise"):
            sub = df[(df.seed == 0) & (df.fill == fill) & (df.radius_frac == 0.18)]
            if head in sub.model.values and ctrl in sub.model.values:
                da = H.paired_permutation(sub, "auroc", head, ctrl)
                dc = H.paired_permutation(sub, "cur", head, ctrl)
                lines += ["", f"{head} vs per-image control {ctrl}, {fill}: dAUROC {da[0]:+.3f} (p {da[1]:.4g}), "
                          f"dCUR {dc[0]:+.3f} (p {dc[1]:.4g})"]
    if frames_s:
        ds = pd.concat(frames_s, ignore_index=True)
        lines += ["", "## Stress tests (seeds 0, 1; radius 0.18): head, per-image control and native", "",
                  "| test | model | n scans | AUROC | CUR | CUR-err |", "|---|---|---|---|---|---|"]
        for test in STRESS:
            dt = ds[ds.test == test]
            for m in sorted(dt.model.unique()):
                d = dt[dt.model == m]
                ci = {c: H.cluster_bootstrap_ci(d, c, n_boot=10000) for c in ("auroc", "cur", "cur_err")}
                f = lambda c: f"{ci[c][0]:.3f} [{ci[c][1]:.3f}, {ci[c][2]:.3f}]"
                lines.append(f"| {test} | {m} | {d.scan.nunique()} | {f('auroc')} | {f('cur')} | {f('cur_err')} |")
            for base in args.bases:
                head, ctrl = f"{base}_oia", f"{base}_oiapatch"
                if head in dt.model.values and ctrl in dt.model.values:
                    da = H.paired_permutation(dt, "auroc", head, ctrl); dc = H.paired_permutation(dt, "cur", head, ctrl)
                    lines.append(f"| {test} | {head} vs {ctrl} | | dAUROC {da[0]:+.3f} (p {da[1]:.4g}) | dCUR {dc[0]:+.3f} (p {dc[1]:.4g}) | |")
                if head in dt.model.values and base in dt.model.values:
                    da = H.paired_permutation(dt, "auroc", head, base); dc = H.paired_permutation(dt, "cur", head, base)
                    lines.append(f"| {test} | {head} vs {base} | | dAUROC {da[0]:+.3f} (p {da[1]:.4g}) | dCUR {dc[0]:+.3f} (p {dc[1]:.4g}) | |")
    # pass condition, pre-registered in the paper
    # part 1: the head beats native confidence on AUROC and CUR (paired sign-flip p < 0.05, Table 4 keys)
    # part 2: the per-image control does not explain the gain on the inpainted fill (stress inpaint10) or on
    #         CO3Dv2: the head beats the control on AUROC at p < 0.05, or the control's AUROC gain over native
    #         confidence is under half of the head's
    def _ctrl_check(dt, head, ctrl, base):
        if not all(m in dt.model.values for m in (head, ctrl, base)):
            return None
        hb = H.paired_permutation(dt, "auroc", head, base)
        cb = H.paired_permutation(dt, "auroc", ctrl, base)
        hc = H.paired_permutation(dt, "auroc", head, ctrl)
        ok = (hc[0] > 0 and hc[1] < 0.05) or (cb[0] < 0.5 * hb[0])
        return ok, hb, cb, hc

    co3d_df = None
    p = os.path.join(args.out, "P3-oia-co3d", "P3-oia-co3d.csv")
    ref3 = os.path.join(ROOT, "results", "table3_all_models.csv")
    if os.path.exists(p):
        co3d_df = pd.concat([pd.read_csv(p)] + ([pd.read_csv(ref3)] if os.path.exists(ref3) else []), ignore_index=True)
        co3d_df = co3d_df[co3d_df.fill.isin(["blur", "noise"])]
    for base in args.bases:
        m, ctrl = f"{base}_oia", f"{base}_oiapatch"
        if m in t4.model.values and base in t4.model.values:
            da = H.paired_permutation(t4, "auroc", m, base)
            dc = H.paired_permutation(t4, "cur", m, base)
            ok = da[0] > 0 and da[1] < 0.05 and dc[0] < 0 and dc[1] < 0.05
            lines += ["", f"**Pass condition for {m}, part 1 (head vs native, Table 4 keys)**: dAUROC {da[0]:+.3f} "
                      f"(p {da[1]:.4g}), dCUR {dc[0]:+.3f} (p {dc[1]:.4g}) -> " + ("PASS" if ok else "FAIL")]
            checks = {}
            if frames_s:
                checks["inpainted fill (stress inpaint10)"] = _ctrl_check(ds[ds.test == "inpaint10"], m, ctrl, base)
            if co3d_df is not None:
                checks["CO3Dv2 (blur and noise fills)"] = _ctrl_check(co3d_df, m, ctrl, base)
            for name, r in checks.items():
                if r is None:
                    lines.append(f"part 2, {name}: head, control or native rows missing")
                    continue
                ok2, hb, cb, hc = r
                lines.append(f"part 2, {name}: head-native dAUROC {hb[0]:+.3f}, control-native {cb[0]:+.3f}, "
                             f"head-control {hc[0]:+.3f} (p {hc[1]:.4g}) -> "
                             + ("control does not explain the gain" if ok2 else "control explains the gain"))
            have2 = [r for r in checks.values() if r is not None]
            if have2:
                ok_all = ok and all(r[0] for r in have2)
                lines.append(f"**Gate 2 for {m}**: " + ("PASS" if ok_all else "FAIL")
                             + f" (part 1 {'ok' if ok else 'fail'}; part 2 {sum(int(r[0]) for r in have2)}/{len(have2)}"
                             " tests; criterion: head beats control at p<0.05 or the control's AUROC gain over native"
                             " is under half the head's, on inpaint and CO3Dv2)")
            else:
                lines.append(f"**Gate 2 for {m}**: part 2 not yet run (needs the stress inpaint10 and co3d stages)")
    # full configuration, pooled blur + noise, DTU and CO3D
    for run_name, title in [("P3-oia-dtu", "DTU"), ("P3-oia-co3d", "CO3Dv2 (head never saw it)"), ("P3-eth3d", "ETH3D"),
                            ("P3-dtuext", "DTU extension scans (seed 0, radius 0.18)"), ("P3-gs", "DTU, Gaussian-splatting backbone"),
                            ("P3-gs-co3d", "CO3Dv2, Gaussian-splatting backbone"),
                            ("P6-vis-on-fill-co3d", "CO3Dv2, visibility detectors on the filled patches (seed 0, radius 0.18)"),
                            ("P8-xmodel-co3d", "CO3Dv2, cross-model disagreement on the filled patches (seed 0, radius 0.18)")]:
        p = os.path.join(args.out, run_name, f"{run_name}.csv")
        ref = os.path.join(ROOT, "results", {"DTU": "table1_all_models.csv", "CO3Dv2 (head never saw it)": "table3_all_models.csv"}.get(title, "none"))
        if not os.path.exists(p):
            continue
        dd = pd.concat([pd.read_csv(p)] + ([pd.read_csv(ref)] if os.path.exists(ref) else []), ignore_index=True)
        dd = dd[dd.fill.isin(["blur", "noise"])]
        lines += ["", f"## Full configuration, {title}, pooled over blur and noise fills", "", "| model | AUROC | CUR | CUR-err |", "|---|---|---|---|"]
        for m in sorted(dd.model.unique()):
            d = dd[dd.model == m]
            ci = {c: H.cluster_bootstrap_ci(d, c, n_boot=10000) for c in ("auroc", "cur", "cur_err")}
            f = lambda c: f"{ci[c][0]:.3f} [{ci[c][1]:.3f}, {ci[c][2]:.3f}]"
            lines.append(f"| {m} | {f('auroc')} | {f('cur')} | {f('cur_err')} |")
    # P4-vis: natural occlusion, no patch. The visibility-trained head against its per-image control and native
    # confidence, per number of input views; the fill-trained head shows whether patch detection transfers.
    for run_name, title in [("P4-vis", "DTU"), ("P4-vis-co3d", "CO3Dv2"), ("P4-vis-eth3d", "ETH3D"),
                            ("P4-vis-ext", "DTU, 2 and 20 views (outside the training range)"),
                            ("P7-vis-imc", "Phototourism heritage landmarks (tourist photos, COLMAP depth)"),
                            ("P6-adapt-co3d", "CO3Dv2, heads adapted label-free on the other fold (pooled with P4-vis-co3d)"),
                            ("P6-adapt-eth3d", "ETH3D, heads adapted label-free on the four non-benchmark scenes (pooled with P4-vis-eth3d)"),
                            ("P7-adapt-imc", "Phototourism, heads adapted label-free on the other fold (pooled with P7-vis-imc)"),
                            ("P8-xmodel-vis", "DTU, cross-model disagreement on real single-view geometry")]:
        p = os.path.join(args.out, run_name, f"{run_name}.csv")
        if not os.path.exists(p):
            continue
        dv = pd.read_csv(p)
        if "adapt" in run_name:                    # pool the folds under one name and add the reference detectors
            dv["model"] = dv.model.str.replace(r"_x(co3d|eth3d|imc)[AB]$", r"_x\1", regex=True)
            ref_run = {"co3d": "P4-vis-co3d", "eth3d": "P4-vis-eth3d", "imc": "P7-vis-imc"}[run_name.rsplit("-", 1)[1]]
            ref = os.path.join(args.out, ref_run, f"{ref_run}.csv")
            if os.path.exists(ref):
                dv = pd.concat([dv, pd.read_csv(ref)], ignore_index=True)
        if "skipped" in dv:
            dv = dv[~dv.skipped.astype(bool)]
        lines += ["", f"## True-visibility protocol, {title} (clean views; U = pixels no other view sees, S = seen by two or more)", "",
                  "| views | detector | n scenes | AUROC | CUR | CUR-err | AbsRel S | AbsRel U |", "|---|---|---|---|---|---|---|---|"]
        for fill in sorted(dv.fill.unique(), key=lambda f: int(f[3:])):
            dt = dv[dv.fill == fill]
            for m in sorted(dt.model.unique()):
                d = dt[dt.model == m]
                ci = {c: H.cluster_bootstrap_ci(d, c, n_boot=10000) for c in ("auroc", "cur", "cur_err")}
                f = lambda c: f"{ci[c][0]:.3f} [{ci[c][1]:.3f}, {ci[c][2]:.3f}]"
                lines.append(f"| {fill[3:]} | {m} | {d.scan.nunique()} | {f('auroc')} | {f('cur')} | {f('cur_err')} | "
                             f"{d.absrel_S.mean():.3f} | {d.absrel_U.mean():.3f} |")
            for base in args.bases:
                head, ctrl, fill_head = f"{base}_oiavis", f"{base}_oiavispatch", f"{base}_oia"
                sh, sctrl, rp = f"{base}_oiavisself", f"{base}_oiavisselfpatch", f"{base}_reproj"
                bd, de = f"{base}_border", f"{base}_dedge"
                ad = [(f"{sh}_x{t}", f"{sctrl}_x{t}") for t in ("co3d", "eth3d", "imc")]            # adapted head vs its control
                ad += [(f"{sh}_x{t}", m) for t in ("co3d", "eth3d", "imc") for m in (sh, head, base, rp)]  # vs DTU-trained, native, reproj
                for a, b in ((head, ctrl), (head, base), (head, fill_head), (fill_head, base), (head, rp),
                             (head, bd), (head, de), (rp, bd), (rp, de),
                             (sh, sctrl), (sh, base), (sh, rp), (sh, head), (sh, bd), (sh, de),
                             (head, f"{head}_s1"), (head, f"{head}_s2"), *ad):
                    if a in dt.model.values and b in dt.model.values:
                        da = H.paired_permutation(dt, "auroc", a, b); dc = H.paired_permutation(dt, "cur", a, b)
                        lines.append(f"| {fill[3:]} | {a} vs {b} | | dAUROC {da[0]:+.3f} (p {da[1]:.4g}) | dCUR {dc[0]:+.3f} (p {dc[1]:.4g}) | | | |")
                for hh, cc in ((head, ctrl), (sh, sctrl)):
                    if hh in dt.model.values and cc in dt.model.values and base in dt.model.values:
                        ok2, hb, cb, hc = _ctrl_check(dt, hh, cc, base)
                        lines.append(f"| {fill[3:]} | **{hh} gate** | | head-native {hb[0]:+.3f}, control-native {cb[0]:+.3f}, "
                                     f"head-control {hc[0]:+.3f} (p {hc[1]:.4g}) | "
                                     + ("control does not explain the gain" if ok2 else "control explains the gain") + " | | | |")
    out = os.path.join(args.out, "P3-stats.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"-> {out}")


STAGES = {"check": stage_check, "train": stage_train, "dtu": stage_dtu, "co3d": stage_co3d, "stress": stage_stress,
          "baselines": stage_baselines, "pvfull": stage_pvfull, "ablate": stage_ablate, "dtuext": stage_dtuext,
          "eth3d": stage_eth3d, "gs": stage_gs, "stats": stage_stats,
          "trainvis": stage_trainvis, "vis": stage_vis, "vis_co3d": stage_vis_co3d, "vis_eth3d": stage_vis_eth3d,
          "trainvisself": stage_trainvisself, "trainvisprobe": stage_trainvisprobe,
          "adapt_co3d": stage_adapt_co3d, "adapt_eth3d": stage_adapt_eth3d, "vis_ext": stage_vis_ext, "dump": stage_dump,
          "vis_on_fill": stage_vis_on_fill, "vis_imc": stage_vis_imc, "adapt_imc": stage_adapt_imc, "xmodel": stage_xmodel,
          "trainprov": stage_trainprov, "prov": stage_prov}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--dtu", default=None, help="directory searched for a DTU root")
    ap.add_argument("--co3d", default=None, help="preprocessed CO3D root (folders with meta.npz)")
    ap.add_argument("--eth3d", default=None, help="preprocessed ETH3D root (scripts/eth3d.py; same folder format)")
    ap.add_argument("--imc", default=None, help="preprocessed phototourism root (scripts/imc.py; same folder format)")
    ap.add_argument("--dump-datasets", default="dtu,co3d,eth3d,imc", help="dump stage: which datasets to dump")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2", help="initialisation seeds of the main head (0 = the reported one)")
    ap.add_argument("--bases", default="vggt,pi3x")
    ap.add_argument("--pv-bases", default="vggt,pi3x,pi3,da3,mapanything", help="backbones for the pvfull stage")
    ap.add_argument("--ext-bases", default="vggt,pi3x,pi3,da3,mapanything", help="native models for the dtuext stage")
    ap.add_argument("--skip-gt", action="store_true", help="do not train the GT-placed variant (oiagt)")
    ap.add_argument("--variants", default="oia,oialin,oiapatch,oiaattn",
                    help="self-labelled VGGT heads trained jointly (drop oiaattn on a 16 GB GPU)")
    ap.add_argument("--gs-render", action="store_true", help="gs stage: also run anysplat_render (needs gsplat)")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    a.bases = a.bases.split(",")
    a.pv_bases = a.pv_bases.split(",")
    a.ext_bases = a.ext_bases.split(",")
    a.dtu_root = None
    if a.stage != "stats":
        a.dtu_root = H.find_dtu_root((a.dtu,)) if a.dtu else H.find_dtu_root()
    a.co3d_root = (co3d.find_co3d_root((a.co3d,)) if a.co3d else None) if a.stage in ("all", "co3d", "check", "gs", "vis_co3d", "adapt_co3d", "dump", "vis_on_fill", "xmodel", "prov") else None
    a.eth3d_root = co3d.find_co3d_root((a.eth3d,)) if (a.eth3d and a.stage in ("all", "eth3d", "vis_eth3d", "adapt_eth3d", "dump", "prov")) else None
    a.imc_root = co3d.find_co3d_root((a.imc,)) if (a.imc and a.stage in ("all", "vis_imc", "adapt_imc", "dump", "prov")) else None
    stages = list(STAGES) if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "co3d" and not a.co3d_root:
            print("[skip] co3d: no --co3d root"); continue
        if s == "eth3d" and not a.eth3d_root:
            print("[skip] eth3d: no --eth3d root"); continue
        print(f"\n##### stage {s} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        STAGES[s](a)


if __name__ == "__main__":
    main()
