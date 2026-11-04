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
        jobs.append(("vggt", "oia,oialin,oiapatch,oiaattn", extra, "vggt_oia.pt"))
        if not args.skip_gt:
            jobs.append(("vggt", "oiagt", [], "vggt_oiagt.pt"))
    for b in args.bases:
        if b != "vggt":
            jobs.append((b, "oia,oiapatch", [], f"{b}_oia.pt"))
    for base, variants, seeds, marker in jobs:
        if os.path.exists(os.path.join(args.ckpt, marker)) and not args.retrain:
            print(f"[train] {marker} exists, skipping {base} {variants}"); continue
        # one subprocess per job so that GPU memory is released between backbones
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "oia_head.py"), "train", "--base", base,
               "--variant", variants, "--dtu", args.dtu_root, "--out", args.ckpt, "--steps", str(args.steps), "--seed", "0"]
        if "," not in variants:
            cmd[cmd.index("--out") + 1] = os.path.join(args.ckpt, marker)
        if seeds:
            cmd += ["--extra-seeds", ",".join(str(x) for x in seeds)]
        print("[train]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


def _ckpt_env(args):
    os.environ["OIA_CKPT_DIR"] = args.ckpt


def _have(args, name):
    return os.path.exists(os.path.join(args.ckpt, f"{name}.pt"))


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


def stage_ablate(args):
    _ckpt_env(args)
    models = [m for m in ("vggt_oialin", "vggt_oiapatch", "vggt_oiaattn", "vggt_oiagt", "pi3x_oiapatch") if _have(args, m)]
    cfg = {"seeds": [0], "fills": ["gray", "blur", "noise"], "radius_fracs": [0.18]}
    run(args, "P3-oia-abl", models, cfg, args.dtu_root)
    # extra initialisation seeds of the main head: evaluated through the same wrapper, rows renamed vggt_oia_s<seed>
    for s in args.seeds[1:]:
        ck = os.path.join(args.ckpt, f"vggt_oia_s{s}.pt")
        if not os.path.exists(ck):
            continue
        os.environ["OIA_CKPT_VGGT_OIA"] = ck
        csv = run(args, f"P3-oia-abl-seed{s}", ["vggt_oia"], cfg, args.dtu_root)
        if os.path.exists(csv):
            import pandas as pd
            df = pd.read_csv(csv); df["model"] = f"vggt_oia_s{s}"; df.to_csv(csv, index=False)
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
    for run_name in ["P3-oia-dtu", "P3-reproj", "P3-pv", "P3-pv-full", "P3-native", "P3-oia-abl", "P3-dtuext"] + [f"P3-oia-abl-seed{s}" for s in args.seeds[1:]]:
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
    t4 = df[(df.seed == 0) & (df.fill == "blur") & (df.radius_frac == 0.18)]
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
    ref = os.path.join(ROOT, "results", "P2-stress", "P2-stress.csv")
    if os.path.exists(ref) and frames_s:
        x = pd.read_csv(ref)
        if "test" not in x.columns:
            x["test"] = "?"
        frames_s.append(x)
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
    # pass condition
    for base in args.bases:
        m = f"{base}_oia"
        if m in t4.model.values and base in t4.model.values:
            da = H.paired_permutation(t4, "auroc", m, base)
            dc = H.paired_permutation(t4, "cur", m, base)
            ok = da[0] > 0 and da[1] < 0.05 and dc[0] < 0 and dc[1] < 0.05
            lines += ["", f"**Pass condition for {m}**: dAUROC {da[0]:+.3f} (p {da[1]:.4g}), dCUR {dc[0]:+.3f} (p {dc[1]:.4g}) -> "
                      + ("PASS" if ok else "FAIL")]
    # full configuration, pooled blur + noise, DTU and CO3D
    for run_name, title in [("P3-oia-dtu", "DTU"), ("P3-oia-co3d", "CO3Dv2 (zero-shot)")]:
        p = os.path.join(args.out, run_name, f"{run_name}.csv")
        ref = os.path.join(ROOT, "results", "table1_all_models.csv" if title == "DTU" else "table3_all_models.csv")
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
    out = os.path.join(args.out, "P3-stats.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"-> {out}")


STAGES = {"check": stage_check, "train": stage_train, "dtu": stage_dtu, "co3d": stage_co3d, "stress": stage_stress,
          "baselines": stage_baselines, "pvfull": stage_pvfull, "ablate": stage_ablate, "dtuext": stage_dtuext,
          "stats": stage_stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--dtu", default=None, help="directory searched for a DTU root")
    ap.add_argument("--co3d", default=None, help="preprocessed CO3D root (folders with meta.npz)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2", help="initialisation seeds of the main head (0 = the reported one)")
    ap.add_argument("--bases", default="vggt,pi3x")
    ap.add_argument("--pv-bases", default="vggt,pi3x,pi3,da3,mapanything", help="backbones for the pvfull stage")
    ap.add_argument("--ext-bases", default="vggt,pi3x,pi3,da3,mapanything", help="native models for the dtuext stage")
    ap.add_argument("--skip-gt", action="store_true", help="do not train the GT-placed variant (oiagt)")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    a.bases = a.bases.split(",")
    a.pv_bases = a.pv_bases.split(",")
    a.ext_bases = a.ext_bases.split(",")
    a.dtu_root = None
    if a.stage != "stats":
        a.dtu_root = H.find_dtu_root((a.dtu,)) if a.dtu else H.find_dtu_root()
    a.co3d_root = (co3d.find_co3d_root((a.co3d,)) if a.co3d else None) if a.stage in ("all", "co3d", "check") else None
    stages = list(STAGES) if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "co3d" and not a.co3d_root:
            print("[skip] co3d: no --co3d root"); continue
        print(f"\n##### stage {s} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        STAGES[s](a)


if __name__ == "__main__":
    main()
