"""Project-site pages generated from results/ (no hand-typed numbers). Run from anywhere:
  python scripts/make_site.py [--raw-base URL] [--check-only]

Writes
  docs/explorer.html                  per-scene explorer: the recorded composite for every (model, scene, radius)
                                      with the CSV row it came from, plus a leaderboard with bootstrap intervals
  docs/assets/hallubench3d_results.zip  every result CSV, config and consistency JSON under results/
and checks that the headline numbers written in docs/index.html equal the ones recomputed from the CSVs.

Composite images are not copied into docs/ (they are 200 MB); the explorer loads them from the repository
(--raw-base, default: raw.githubusercontent.com of this repo, branch main)."""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")
RAW_BASE = "https://raw.githubusercontent.com/prajak002/DTU-3D-reconstruction/main/"

NAME = {"vggt": "VGGT-1B", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAnything", "da3": "Depth Anything 3"}
ORDER = ["vggt", "pi3", "pi3x", "mapanything", "da3"]
DATASETS = {  # dataset -> (merged table, runs whose ex_ folders hold the composites)
    "DTU": ("table1_all_models.csv", ["P1-full", "P2-full"]),
    "CO3Dv2": ("table3_all_models.csv", ["P2-co3d", "P2-co3d-b"]),
}
PNG_RE = re.compile(r"^(?P<model>" + "|".join(sorted(NAME, key=len, reverse=True)) + r")_scan(?P<scan>.+)_r(?P<rf>0\.\d+)\.png$")


def r3(x):
    return None if pd.isna(x) else round(float(x), 3)


def scene_label(ds, scan):
    if ds == "DTU":
        return f"scan {scan}"
    parts = str(scan).split("_")
    return f"{parts[0]} {parts[1]}" if len(parts) > 1 else str(scan)


def build_data(raw_base):
    out = {"generated_from": [], "datasets": {}}
    for ds, (table, runs) in DATASETS.items():
        df = pd.read_csv(os.path.join(RES, table))
        df["scan"] = df["scan"].astype(str)
        out["generated_from"].append(f"results/{table}")
        pooled = df[df.fill.isin(["blur", "noise"])]
        board = []
        for m in ORDER:
            g = pooled[pooled.model == m]
            cur, clo, chi, n = H.cluster_bootstrap_ci(g, "cur")
            au, alo, ahi, _ = H.cluster_bootstrap_ci(g, "auroc")
            ce, _, _, _ = H.cluster_bootstrap_ci(g, "cur_err")
            board.append(dict(model=m, name=NAME[m], cur=r3(cur), cur_lo=r3(clo), cur_hi=r3(chi),
                              auroc=r3(au), auroc_lo=r3(alo), auroc_hi=r3(ahi), cur_err=r3(ce), n_scenes=int(n), n_runs=int(len(g))))
        per_scene = (pooled.groupby(["scan", "model"])[["cur", "auroc"]].mean().round(3).reset_index())
        scenes = {}
        for sc, g in per_scene.groupby("scan"):
            scenes[sc] = {r.model: dict(cur=r3(r.cur), auroc=r3(r.auroc)) for r in g.itertuples()}
        composites = []
        for run in runs:
            ex_dir = os.path.join(RES, run, f"ex_{run}")
            for f in sorted(glob.glob(os.path.join(ex_dir, "*.png"))):
                m = PNG_RE.match(os.path.basename(f))
                if not m:
                    continue
                model, scan, rf = m["model"], m["scan"], float(m["rf"])
                row = df[(df.model == model) & (df.scan == scan) & (df.seed == 0) & (df.fill == "gray") & (df.radius_frac == rf)]
                rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
                composites.append(dict(
                    model=model, scan=scan, rf=rf, run=run, url=raw_base + rel,
                    row=None if row.empty else {k: r3(row.iloc[0][k]) for k in ["auroc", "cur", "cur_err", "absrel_U", "absrel_S", "absrel_U_ctrl", "conf_ratio_U"]},
                ))
            out["generated_from"].append(os.path.relpath(ex_dir, ROOT).replace(os.sep, "/"))
        scan_order = sorted(scenes, key=lambda s: (int(s) if s.isdigit() else 10**9, s))
        out["datasets"][ds] = dict(
            table=table, leaderboard=board, scenes=scenes, scene_order=scan_order,
            scene_labels={s: scene_label(ds, s) for s in scan_order}, composites=composites,
            n_rows=int(len(df)), n_scenes=int(df.scan.nunique()),
        )
    return out


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HalluBench-3D explorer</title>
<meta name="description" content="Per-scene explorer for HalluBench-3D: recorded model outputs for every (model, scene) with the CSV rows they came from.">
<style>
:root{--paper:#F7F8F6;--ink:#17201F;--slate:#4C5A58;--rule:#D9DEDC;--observed:#1E8A7A;--unsupported:#18B8E0;--pending:#B86E00;--panel:#FFFFFF}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#121716;--ink:#E7ECEA;--slate:#A3B0AD;--rule:#2A3331;--observed:#4FC3B1;--unsupported:#5FD3F0;--panel:#1A2120}}
:root[data-theme="dark"]{--paper:#121716;--ink:#E7ECEA;--slate:#A3B0AD;--rule:#2A3331;--observed:#4FC3B1;--unsupported:#5FD3F0;--panel:#1A2120}
*{box-sizing:border-box}
html{background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:18px;line-height:1.5;font-variant-numeric:tabular-nums}
body{margin:0;background:var(--paper)}
main{padding:0 16px 96px;max-width:1100px;margin:0 auto}
header{padding:48px 0 16px}
h1{font-weight:400;font-size:clamp(1.8rem,5vw,3rem);line-height:1.05;margin:0 0 .3em}
h2{font-weight:500;font-size:1.4rem;margin:2.4rem 0 .6rem}
p{margin:0 0 1em;max-width:70ch}
a{color:inherit;text-decoration-color:var(--unsupported);text-decoration-thickness:2px;text-underline-offset:3px}
.sub{color:var(--slate)}
.small{font-size:.9rem;color:var(--slate)}
table{border-collapse:collapse;width:100%;font-size:.95rem;margin:.6rem 0 1rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:500;color:var(--slate)}
td.num,th.num{text-align:right;white-space:nowrap}
.ci{color:var(--slate);font-size:.85em}
.controls{display:flex;flex-wrap:wrap;gap:.6rem 1rem;align-items:end;margin:1rem 0}
.controls label{display:flex;flex-direction:column;font-size:.85rem;color:var(--slate);gap:.2rem}
select,button{font:inherit;font-size:1rem;padding:.4rem .6rem;border:1px solid var(--ink);background:var(--panel);color:var(--ink);border-radius:3px}
button{cursor:pointer}
.stage{background:var(--panel);border:1px solid var(--rule);padding:8px}
.stage img{display:block;width:100%;height:auto;min-height:120px;background:#e9ecea}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem 1rem;margin:.8rem 0}
.metrics div{border-top:2px solid var(--rule);padding-top:.3rem}
.metrics b{display:block;font-weight:500;font-size:1.3rem}
.metrics span{font-size:.85rem;color:var(--slate)}
.cur b{color:var(--unsupported)}
.badge{display:inline-block;background:var(--rule);color:var(--ink);padding:.05em .5em;border-radius:3px;font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:1rem 0}
.grid figure{margin:0;border:1px solid var(--rule);background:var(--panel);padding:6px;cursor:pointer}
.grid img{width:100%;height:auto;display:block;background:#e9ecea}
.grid figcaption{font-size:.8rem;color:var(--slate);margin-top:.3rem}
.grid figure.on{outline:2px solid var(--unsupported)}
@media (max-width:640px){html{font-size:16px}}
</style>
</head>
<body>
<main>
<header>
<p class="small"><a href="index.html">Observe, Imagine, Abstain</a> / per-scene explorer</p>
<h1>HalluBench-3D, scene by scene</h1>
<p class="sub">Every panel below is a model output recorded during a logged benchmark run (Kaggle T4, 2026-10-28). The numbers next to it are the CSV row that run wrote for the same (model, scene, seed 0, gray fill, radius). Nothing on this page is typed by hand: it is generated by <code>scripts/make_site.py</code> from <code>results/</code>.</p>
</header>

<h2>Leaderboard</h2>
<p class="small">CUR pooled over blur and noise fills, five seeds, two radii; 95% cluster-bootstrap intervals over scenes (10,000 resamples). Lower CUR is better; higher AUROC is better. CUR-err: confident and wrong by more than 10% depth.</p>
<div id="boards"></div>

<h2>Explorer</h2>
<div class="controls">
  <label>dataset<select id="ds"></select></label>
  <label>model<select id="model"></select></label>
  <label>scene<select id="scene"></select></label>
  <label>radius (x object diagonal)<select id="rf"></select></label>
  <button id="prev" type="button">previous scene</button>
  <button id="next" type="button">next scene</button>
</div>
<div class="stage">
  <img id="img" alt="recorded composite: masked input, ground-truth depth, model depth, model log-confidence; cyan outline marks the hidden region">
</div>
<p class="small">Left to right: masked input (view 1), ground-truth depth, model depth after median scaling on supported pixels, model log-confidence. The cyan contour is the hidden region U.</p>
<div class="metrics" id="metrics"></div>
<p class="small" id="src"></p>

<h3>Every model on this scene</h3>
<p class="small">Mean over blur and noise fills, five seeds and both radii, from the merged CSV.</p>
<div id="scene_table"></div>

<h3>Every scene for this model and radius</h3>
<div class="grid" id="grid"></div>

<p class="small" id="foot"></p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('data').textContent);
  const $ = id => document.getElementById(id);
  const ORDER = __ORDER__; const NAME = __NAME__;
  const fmt = v => v == null ? 'n/a' : v.toFixed(3);
  const st = {ds: Object.keys(D.datasets)[0], model: 'vggt', scene: null, rf: null};
  try { Object.assign(st, JSON.parse(localStorage.getItem('hb3d_explorer') || '{}')); } catch (e) {}
  function save(){ try { localStorage.setItem('hb3d_explorer', JSON.stringify(st)); } catch (e) {} }
  function opts(sel, items, value){ sel.innerHTML=''; for (const [v,l] of items){ const o=document.createElement('option'); o.value=v; o.textContent=l; sel.appendChild(o);} sel.value=value; }
  function boards(){
    let h='';
    for (const [ds,d] of Object.entries(D.datasets)){
      h += `<h3>${ds} <span class="badge">${d.n_scenes} scenes, ${d.n_rows} rows, ${d.table}</span></h3><table><thead><tr><th>model</th><th class="num">CUR</th><th class="num">AUROC</th><th class="num">CUR-err</th></tr></thead><tbody>`;
      for (const r of d.leaderboard) h += `<tr><td>${r.name}</td><td class="num">${fmt(r.cur)} <span class="ci">[${fmt(r.cur_lo)}, ${fmt(r.cur_hi)}]</span></td><td class="num">${fmt(r.auroc)} <span class="ci">[${fmt(r.auroc_lo)}, ${fmt(r.auroc_hi)}]</span></td><td class="num">${fmt(r.cur_err)}</td></tr>`;
      h += '</tbody></table>';
    }
    $('boards').innerHTML = h;
  }
  function comps(){ return D.datasets[st.ds].composites; }
  function render(){
    const d = D.datasets[st.ds];
    const models = ORDER.filter(m => comps().some(c => c.model === m));
    if (!models.includes(st.model)) st.model = models[0];
    const rfs = [...new Set(comps().filter(c => c.model === st.model).map(c => c.rf))].sort();
    if (!rfs.includes(st.rf)) st.rf = rfs[rfs.length - 1];
    const scenes = d.scene_order.filter(s => comps().some(c => c.model === st.model && c.scan === s && c.rf === st.rf));
    if (!scenes.includes(st.scene)) st.scene = scenes[0];
    opts($('ds'), Object.keys(D.datasets).map(k => [k, k]), st.ds);
    opts($('model'), models.map(m => [m, NAME[m]]), st.model);
    opts($('scene'), scenes.map(s => [s, d.scene_labels[s]]), st.scene);
    opts($('rf'), rfs.map(r => [String(r), r.toFixed(2)]), String(st.rf));
    const c = comps().find(x => x.model === st.model && x.scan === st.scene && x.rf === st.rf);
    $('img').src = c ? c.url : '';
    const r = c && c.row;
    const cells = r ? [['CUR', r.cur, 'cur'], ['AUROC', r.auroc], ['CUR-err', r.cur_err], ['AbsRel U', r.absrel_U], ['AbsRel S', r.absrel_S], ['AbsRel U (clean)', r.absrel_U_ctrl], ['conf ratio U', r.conf_ratio_U]] : [];
    $('metrics').innerHTML = cells.map(([l, v, k]) => `<div class="${k || ''}"><b>${fmt(v)}</b><span>${l}</span></div>`).join('');
    $('src').textContent = c ? `run ${c.run}; row: model=${c.model}, scan=${c.scan}, seed=0, fill=gray, radius_frac=${c.rf}; file ${c.url.split('/results/')[1] || c.url}` : 'no composite recorded for this selection';
    const sc = d.scenes[st.scene] || {};
    $('scene_table').innerHTML = '<table><thead><tr><th>model</th><th class="num">CUR</th><th class="num">AUROC</th></tr></thead><tbody>' +
      ORDER.filter(m => sc[m]).map(m => `<tr${m === st.model ? ' style="font-weight:500"' : ''}><td>${NAME[m]}</td><td class="num">${fmt(sc[m].cur)}</td><td class="num">${fmt(sc[m].auroc)}</td></tr>`).join('') + '</tbody></table>';
    $('grid').innerHTML = scenes.map(s => { const cc = comps().find(x => x.model === st.model && x.scan === s && x.rf === st.rf); const v = (d.scenes[s] || {})[st.model];
      return `<figure data-s="${s}" class="${s === st.scene ? 'on' : ''}"><img loading="lazy" src="${cc ? cc.url : ''}" alt="${NAME[st.model]} on ${d.scene_labels[s]}"><figcaption>${d.scene_labels[s]} · CUR ${v ? fmt(v.cur) : 'n/a'}</figcaption></figure>`; }).join('');
    for (const f of $('grid').querySelectorAll('figure')) f.addEventListener('click', () => { st.scene = f.dataset.s; save(); render(); window.scrollTo({top: $('img').offsetTop - 80, behavior: 'smooth'}); });
    save();
  }
  $('ds').addEventListener('change', e => { st.ds = e.target.value; st.scene = null; render(); });
  $('model').addEventListener('change', e => { st.model = e.target.value; render(); });
  $('scene').addEventListener('change', e => { st.scene = e.target.value; render(); });
  $('rf').addEventListener('change', e => { st.rf = parseFloat(e.target.value); render(); });
  function step(k){ const o = $('scene'); const i = (o.selectedIndex + k + o.options.length) % o.options.length; st.scene = o.options[i].value; render(); }
  $('prev').addEventListener('click', () => step(-1)); $('next').addEventListener('click', () => step(1));
  $('foot').textContent = 'Generated from: ' + D.generated_from.join(', ') + '. CSV bundle: assets/hallubench3d_results.zip.';
  boards(); render();
})();
</script>
</body>
</html>
"""


def write_explorer(data):
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__ORDER__", json.dumps(ORDER)).replace("__NAME__", json.dumps(NAME)))
    path = os.path.join(DOCS, "explorer.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1024:.0f} KB, "
          f"{sum(len(d['composites']) for d in data['datasets'].values())} composites)")


def write_bundle():
    path = os.path.join(DOCS, "assets", "hallubench3d_results.zip")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(RES, "**", "*"), recursive=True)):
            if os.path.isfile(f) and f.endswith((".csv", ".json")) and os.sep + "ex_" not in f:
                z.write(f, os.path.relpath(f, ROOT).replace(os.sep, "/")); n += 1
    print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1024:.0f} KB, {n} files)")


def check_index(data):
    """The headline bars and table in docs/index.html are written by hand; make sure they equal the CSVs."""
    html = open(os.path.join(DOCS, "index.html"), encoding="utf-8").read()
    bad = []
    for ds, d in data["datasets"].items():
        for r in d["leaderboard"]:
            for col in ("cur", "auroc"):
                v = f"{r[col]:.3f}"
                if v not in html and v.lstrip("0") not in html:
                    bad.append(f"{ds} {r['name']} {col} {v}")
    if bad:
        print("index.html does NOT contain these recomputed values:", *bad, sep="\n  ")
        return False
    print("index.html headline numbers match the CSVs (all pooled CUR / AUROC values present)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-base", default=RAW_BASE, help="URL prefix the explorer uses for results/ images")
    ap.add_argument("--check-only", action="store_true", help="only verify docs/index.html against the CSVs")
    a = ap.parse_args()
    data = build_data(a.raw_base)
    ok = check_index(data)
    if not a.check_only:
        write_explorer(data)
        write_bundle()
    sys.exit(0 if ok else 1)
