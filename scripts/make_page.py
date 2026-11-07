"""The project page, generated: every number comes from docs/assets/numbers.json (scripts/make_assets.py), the charts
from the same script, the videos from scripts/make_video.py and scripts/make_demo.py.

  python scripts/make_assets.py && python scripts/make_page.py      -> docs/index.html
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = json.load(open(os.path.join(ROOT, "docs", "assets", "numbers.json"), encoding="utf-8"))
NAME = {"vggt": "VGGT-1B", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAnything", "da3": "Depth Anything 3"}
DS = [("dtu", "DTU"), ("co3d", "CO3Dv2"), ("eth3d", "ETH3D"), ("imc", "Phototourism")]


def v(path, k=0, nd=3, default="--"):
    """N['vis']['dtu']['vggt_oiavisself']['auroc'][0] as v('vis/dtu/vggt_oiavisself/auroc')."""
    x = N
    for p in path.split("/"):
        x = x.get(p) if isinstance(x, dict) else None
        if x is None:
            return default
    if isinstance(x, list):
        x = x[k]
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else default


def pct(path, k=0):
    x = v(path, k, 3)
    return "--" if x == "--" else f"{float(x) * 100:.0f}%"


def rng(paths, nd=2):
    vals = [float(v(p, 0, 3)) for p in paths if v(p, 0, 3) != "--"]
    if not vals:
        return "--"
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if abs(hi - lo) < 10 ** -nd else f"{lo:.{nd}f} to {hi:.{nd}f}"


CSS = r"""
:root{--paper:#F7F8F6;--ink:#17201F;--slate:#4C5A58;--rule:#D9DEDC;--observed:#1E8A7A;--unsupported:#18B8E0;--warn:#B86E00;--warn-bg:#FFF4E3;--measure:68ch}
*{box-sizing:border-box}
html{background:var(--paper);color:var(--ink);font-family:"Newsreader",Georgia,serif;font-size:19px;line-height:1.55;font-variant-numeric:tabular-nums;-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0}
a{color:inherit;text-decoration-color:var(--unsupported);text-decoration-thickness:2px;text-underline-offset:3px}
a:hover{color:var(--observed)}
:focus-visible{outline:3px solid var(--unsupported);outline-offset:3px}
main{padding:0 20px 120px}
.measure{max-width:var(--measure);margin:0 auto}
.wide{max-width:1180px;margin:0 auto}
h1{font-weight:400;font-size:clamp(2.2rem,6vw,4.4rem);line-height:1.02;letter-spacing:-0.015em;margin:0 0 .4em}
h1 em{font-style:italic;font-weight:300}
h2{font-weight:500;font-size:1.65rem;line-height:1.2;margin:3.4rem 0 .8rem}
h2 .num{display:block;font-size:.8rem;letter-spacing:.14em;color:var(--slate);font-family:ui-monospace,Menlo,Consolas,monospace;margin-bottom:.4rem}
h3{font-weight:500;font-size:1.15rem;margin:1.8rem 0 .4rem}
p{margin:0 0 1em}
.lede{font-size:1.32rem;line-height:1.4;max-width:40ch}
.sub{color:var(--slate)}
.small{font-size:.9rem;color:var(--slate)}
header{padding:56px 0 20px}
nav.top{position:sticky;top:0;z-index:5;background:rgba(247,248,246,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--rule);font-size:.9rem}
nav.top .in{max-width:1180px;margin:0 auto;display:flex;flex-wrap:wrap;gap:.2rem 1.3rem;padding:.55rem 20px;align-items:center}
nav.top .brand{font-weight:500;margin-right:auto}
nav.top a{text-decoration:none;color:var(--slate)}
nav.top a:hover{color:var(--ink)}
.buttons{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.4rem 0 2.2rem}
.buttons a{display:inline-block;border:1px solid var(--ink);padding:.45rem .95rem;text-decoration:none;font-size:.95rem;border-radius:3px}
.buttons a:hover{background:var(--ink);color:var(--paper)}
.buttons a.primary{background:var(--ink);color:var(--paper)}
.buttons a.primary:hover{background:var(--observed);border-color:var(--observed)}
.reveal{max-width:960px;margin:0 auto 1.5rem}
.reveal figure{margin:0}
.stage{position:relative;aspect-ratio:432/324;background:#e9ecea;border:1px solid var(--rule);overflow:hidden}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .25s ease}
.stage img.on{opacity:1}
.seg{display:flex;flex-wrap:wrap;border:1px solid var(--ink);width:max-content;max-width:100%;margin:14px 0 8px}
.seg button{font:inherit;font-size:1rem;background:none;border:0;border-right:1px solid var(--ink);padding:.5rem 1rem;cursor:pointer;color:var(--ink);flex:1 1 auto}
.seg button:last-child{border-right:0}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--paper)}
.reveal figcaption{color:var(--slate);font-size:.95rem;max-width:70ch}
.cyan{color:var(--unsupported);font-weight:500}
figure.fig{margin:1.6rem auto;max-width:1180px}
figure.fig img{width:100%;height:auto;display:block;border:1px solid var(--rule);background:#fff}
figure.fig figcaption{max-width:var(--measure);margin:.6rem auto 0;color:var(--slate);font-size:.95rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin:1.4rem 0}
.kpi{border-top:2px solid var(--ink);padding-top:.5rem}
.kpi b{display:block;font-size:2rem;font-weight:400;line-height:1.1}
.kpi span{font-size:.9rem;color:var(--slate)}
.callout{border-left:3px solid var(--observed);padding:.2rem 0 .2rem 1rem;margin:1.4rem 0;color:var(--ink)}
.warnbox{background:var(--warn-bg);border-left:3px solid var(--warn);padding:.6rem 1rem;margin:1.4rem 0}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:1rem 0}
th,td{text-align:left;padding:.42rem .6rem .42rem 0;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:500;color:var(--slate)}
td.n,th.n{text-align:right;padding-right:1rem;white-space:nowrap}
tr.hl td{font-weight:500}
tr.grp td{color:var(--slate);font-style:italic;border-bottom:0;padding-top:.8rem}
.tabs{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--ink);width:max-content;max-width:100%;margin:1rem 0}
.tabs button{font:inherit;font-size:.95rem;background:none;border:0;border-right:1px solid var(--ink);padding:.4rem .9rem;cursor:pointer;color:var(--ink)}
.tabs button:last-child{border-right:0}
.tabs button[aria-pressed=true]{background:var(--ink);color:var(--paper)}
.tabpane{display:none}.tabpane.on{display:block}
video{width:100%;height:auto;display:block;background:#000;border:1px solid var(--rule)}
.video{max-width:1180px;margin:1.2rem auto}
pre{font-size:.8rem;line-height:1.4;overflow:auto;background:#EEF1EF;padding:1rem;border:1px solid var(--rule)}
footer{max-width:var(--measure);margin:4rem auto 0;padding-top:1.5rem;border-top:1px solid var(--rule);color:var(--slate);font-size:.9rem}
@media (max-width:640px){.seg{width:100%}.seg button{border-right:0;border-bottom:1px solid var(--ink);flex-basis:100%;text-align:left}.seg button:last-child{border-bottom:0}}
@media (prefers-reduced-motion:reduce){.stage img{transition:none}}
"""

JS = r"""
(function(){
  var stage=document.getElementById('stage');
  if(stage){
    var imgs=stage.querySelectorAll('img'), btns=document.querySelectorAll('.seg button');
    function show(k){imgs.forEach(function(i){i.classList.toggle('on',i.dataset.k===k)});btns.forEach(function(b){b.setAttribute('aria-pressed',String(b.dataset.k===k))})}
    btns.forEach(function(b){b.addEventListener('click',function(){show(b.dataset.k)})});
    var order=['input','depth','conf'],idx=0,touched=false,reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
    btns.forEach(function(b){b.addEventListener('click',function(){touched=true})});
    if(!reduce){var t=setInterval(function(){if(touched){clearInterval(t);return}idx=(idx+1)%3;show(order[idx])},2600)}
  }
  document.querySelectorAll('[data-tabs]').forEach(function(g){
    var bs=g.querySelectorAll('button'), panes=document.querySelectorAll('[data-pane="'+g.dataset.tabs+'"]');
    bs.forEach(function(b){b.addEventListener('click',function(){
      bs.forEach(function(x){x.setAttribute('aria-pressed',String(x===b))});
      panes.forEach(function(p){p.classList.toggle('on',p.dataset.k===b.dataset.k)});
    })});
  });
  var cp=document.getElementById('copybib');
  if(cp){cp.addEventListener('click',function(){navigator.clipboard.writeText(document.getElementById('bib').textContent).then(function(){cp.textContent='Copied'})})}
})();
"""


def leak_table():
    rows = []
    for m in ("vggt", "pi3", "pi3x", "mapanything", "da3"):
        cells = "".join(f'<td class="n">{v(f"leak/{k}/{m}/auroc")}</td><td class="n">{v(f"leak/{k}/{m}/cur")}</td>' for k, _ in [("dtu", 0), ("dtuext", 0), ("co3d", 0), ("eth3d", 0)])
        rows.append(f"<tr><td>{NAME[m]}</td>{cells}</tr>")
    head = "<tr><th>model</th>" + "".join(f'<th class="n" colspan="2">{lab}</th>' for lab in ("DTU, 18 scans", "DTU, 22 more", "CO3Dv2", "ETH3D")) + "</tr>"
    sub = "<tr><th></th>" + '<th class="n">AUROC</th><th class="n">CUR</th>' * 4 + "</tr>"
    return "<table>" + head + sub + "".join(rows) + "</table>"


def vis_table(base):
    rows = [("", "native confidence"), ("border", "distance to image border"), ("oiavispatch", "per-image head (control)"),
            ("reproj", "reprojection consistency"), ("oiavislin", "linear probe, cross-view tokens"), ("oiavis", "support head, ground-truth labels"),
            ("oiavisselfpatch", "per-image control, self labels"), ("oiavisself", "support head, label-free"), ("oiavisself_x", "label-free head trained on the target capture")]
    out = []
    for suf, lab in rows:
        m = base if suf == "" else f"{base}_{suf}"
        cells = "".join(f'<td class="n">{v(f"vis/{k}/{m}/auroc")}</td><td class="n">{v(f"vis/{k}/{m}/cur")}</td>' for k, _ in DS)
        if cells.count(">--<") == 2 * len(DS):
            continue
        cls = ' class="hl"' if suf == "oiavisself" else ""
        out.append(f"<tr{cls}><td>{lab}</td>{cells}</tr>")
    head = "<tr><th>detector</th>" + "".join(f'<th class="n" colspan="2">{lab}</th>' for _, lab in DS) + "</tr>"
    sub = "<tr><th></th>" + '<th class="n">AUROC</th><th class="n">CUR</th>' * 4 + "</tr>"
    return "<table>" + head + sub + "".join(out) + "</table>"


def bands_table():
    rows = []
    for k, lab in DS:
        for m, name in (("vggt", "native confidence"), ("vggt_reproj", "reprojection"), ("vggt_oiavisself", "head, label-free")):
            B = N["bands"].get(k, {}).get(m)
            if not B:
                continue
            f = lambda key, nd=3: ("--" if B.get(key) is None else f"{B[key]:.{nd}f}")
            bands = "--" if B.get("frac_observed") is None else f"{B['frac_observed']:.2f} / {B['frac_abstain']:.2f} / {B['frac_imagined']:.2f}"
            rows.append(f'<tr><td>{lab}</td><td>{name}</td><td class="n">{f("aurc")}</td><td class="n">{f("absrel_all", 4)}</td><td class="n">{f("absrel50", 4)}</td>'
                        f'<td class="n">{f("cur_loso", 2)} / {f("keep_loso", 2)}</td><td class="n">{bands}</td><td class="n">{f("risk_observed")}</td></tr>')
    head = ("<tr><th>dataset</th><th>detector (VGGT-1B)</th><th class=\"n\">AURC</th><th class=\"n\">AbsRel, all</th><th class=\"n\">AbsRel, top half</th>"
            "<th class=\"n\">CUR / kept at a threshold set on the other scenes</th><th class=\"n\">observed / abstain / imagined</th><th class=\"n\">single-view share inside observed</th></tr>")
    return "<table>" + head + "".join(rows) + "</table>"


def page():
    H = []
    H.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Observe, Imagine, Abstain</title>
<meta name="description" content="HalluBench-3D and the label-free support head: measuring and detecting geometry that feed-forward 3D models return without having seen it. CVPR 2027 submission.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<nav class="top" aria-label="Sections"><div class="in">
  <span class="brand">Observe, Imagine, Abstain</span>
  <a href="#finding">Finding</a><a href="#benchmark">Benchmark</a><a href="#method">Method</a><a href="#results">Results</a><a href="#controls">Controls</a><a href="#abstain">Abstaining</a><a href="#limit">Limit</a><a href="#video">Video</a><a href="#cite">Cite</a>
  <a href="https://github.com/prajak002/DTU-3D-reconstruction">GitHub &#8599;</a>
</div></nav>
<main>

<header class="measure">
  <h1>Observe, Imagine, <em>Abstain</em></h1>
  <p class="lede">When a 3D model has not seen a surface, does it say so? We measured five of them. Mostly, no. Then we taught them to say so, without any labels.</p>
  <p class="sub">Detecting hallucinated geometry in feed-forward 3D reconstruction. Anonymous submission to CVPR 2027.</p>
  <div class="buttons">
    <a class="primary" href="assets/paper.pdf">Paper (PDF)</a>
    <a href="assets/supp.pdf">Supplement</a>
    <a href="https://github.com/prajak002/DTU-3D-reconstruction">Code, heads, run log</a>
    <a href="assets/hallubench3d_results.zip">Every result CSV</a>
    <a href="explorer.html">Per-scene explorer</a>
    <a href="explorer_io.html">Input &rarr; output explorer</a>
    <a href="#video">Video</a>
  </div>
</header>

<section class="reveal" aria-labelledby="reveal-h">
  <figure>
    <div class="stage" id="stage">
      <img class="on" src="assets/hero_input.png" alt="DTU scan 17, one of ten input views, with a region of the courtyard inpainted so no seam remains" data-k="input">
      <img src="assets/hero_depth.png" alt="VGGT depth for the same view: the inpainted region is filled with plausible facade geometry; a cyan contour marks pixels no camera saw" data-k="depth">
      <img src="assets/hero_conf.png" alt="VGGT confidence for the same view: high across most of the hidden region" data-k="conf">
    </div>
    <div class="seg" role="group" aria-label="Choose what to show">
      <button type="button" aria-pressed="true" data-k="input">What the model saw</button>
      <button type="button" aria-pressed="false" data-k="depth">The depth it returned</button>
      <button type="button" aria-pressed="false" data-k="conf">How sure it was</button>
    </div>
    <figcaption id="reveal-h">DTU scan 17, VGGT. A 3D region of the real surface was hidden in all ten views and inpainted. The <span class="cyan">cyan contour</span> encloses pixels that no camera observed. The model fills them with a plausible facade and is about as confident there as anywhere else.</figcaption>
  </figure>
</section>

<section class="measure">
  <h2><span class="num">ABSTRACT</span>In one paragraph</h2>
  <p>Feed-forward 3D models (VGGT, Pi3, Pi3X, MapAnything, Depth Anything 3) reconstruct a scene from unposed photographs in one pass. Where no photograph shows a surface they fill the gap from learned priors, and they assign the invented geometry a confidence close to that of observed geometry. We define a predicted point as hallucinated when it lies on a surface no input view shows, and build a benchmark that labels such points for free by hiding a 3D region of the ground-truth surface in every view. Across five models and three datasets, native confidence lets {pct('leak/dtu/vggt/cur')} to {pct('leak/co3d/mapanything/cur')} of unsupported pixels through a filter that keeps 90% of observed ones. Cheap fixes (perturbation variance, reprojection consistency) do not help, and a head trained on the benchmark's own fills learns the fill. We then change what the detector must learn: a small <em>support head</em> on the frozen cross-view features predicts whether each pixel has multi-view support, with labels computed from the backbone's own prediction, so it needs no ground truth. On three backbones and four datasets it never saw it separates single-view from supported geometry at AUROC {rng(['vis/co3d/vggt_oiavisself/auroc', 'vis/dtu/vggt_oiavisself/auroc', 'vis/imc/pi3_oiavisself/auroc'])} against {rng(['vis/co3d/pi3/auroc', 'vis/eth3d/pi3x/auroc'])} for native confidence, matches a head trained on ground-truth visibility, and beats a per-image control that falls below chance off the lab rig. One limit, measured: a consistent invention that every view agrees on passes every test, this head included.</p>
</section>

<section class="measure" id="finding">
  <h2><span class="num">01 / THE FINDING</span>Native confidence does not know what the model has seen</h2>
  <p>We hide one 3D region of the ground-truth surface in every input view (three fills, five placements, two sizes, plus an inpainted fill that leaves no seam), run the model, and ask whether its confidence separates the pixels inside the hidden region (unsupported) from those far outside (supported). AUROC measures the ranking; CUR, the Confident-Unsupported Rate, is the share of unsupported pixels that survive a threshold keeping 90% of supported ones.</p>
  <div class="kpis">
    <div class="kpi"><b>{pct('leak/dtu/vggt/cur')}</b><span>of unseen geometry passes for the best model on the lab rig (VGGT, DTU)</span></div>
    <div class="kpi"><b>{pct('leak/co3d/mapanything/cur')}</b><span>for the worst on phone captures (MapAnything, CO3Dv2)</span></div>
    <div class="kpi"><b>{pct('leak/eth3d/vggt/cur')}</b><span>for VGGT on building-scale ETH3D scans</span></div>
    <div class="kpi"><b>5 of 5</b><span>models leak; only Pi3 behaves the same on the rig and on phone video</span></div>
  </div>
</section>
<figure class="fig"><img src="assets/charts/leak_light.png" alt="Bar chart of CUR per model on DTU, 22 more DTU scans, CO3Dv2 and ETH3D"><figcaption>Share of unseen geometry that passes a confidence filter keeping 90% of observed geometry, pooled over blur and noise fills with 95% cluster-bootstrap intervals over scenes. Runs P1-full, P2-full, P2-co3d, P3-dtuext, P3-eth3d.</figcaption></figure>
<section class="measure">
  {leak_table()}
  <p class="small">Pooled over blur and noise fills; five seeds and two radii on the 18-scene sets, seed 0 and radius 0.18 on the 22 extension scans. Intervals in the paper and in <code>results/P3-stats.md</code>.</p>
</section>

<section class="measure" id="benchmark">
  <h2><span class="num">02 / THE BENCHMARK</span>Labels for free, with controls on the same pixels</h2>
  <p>One sphere, placed on the ground-truth surface, is hidden in every view with the true cameras. Pixels inside it are unsupported in every view; pixels outside a margin are supported. A clean run of the same views gives a control for every pixel, so confident-and-wrong is separated from confident-and-imagined. Two correctness guards run before any model: the ground truth must reproject consistently between views, and unsupported points of one view must land inside the hidden mask of the others.</p>
  <ol class="steps">
    <li>Stress tests: an inpainted fill raises the leak rather than lowering it; dropping views, low parallax and a distractor image change nothing for VGGT and Pi3X; the imagined depth is as accurate as the observed depth on DTU, so the failure is "cannot tell", not "wrong".</li>
    <li>The ranking changes between the rig and phone video: VGGT, MapAnything and DA3 leak far more on CO3Dv2.</li>
    <li>Cheap fixes fail: perturbation variance (K=4, and K=8 with geometric augmentation) is worse than native confidence on all five backbones; reprojection consistency with the model's own cameras is at chance; MapAnything's multi-view confidence leaks more than its learned one.</li>
  </ol>
</section>
<figure class="fig"><img src="assets/fig2_pipeline.png" alt="The masking protocol: sphere placed on the ground-truth surface, hidden in every view, fills, labels, controls"><figcaption>The protocol. Everything downstream of the sphere is computed from the ground truth and the cameras; no model is involved in the labels.</figcaption></figure>
<figure class="fig"><img src="assets/charts/compare_light.png" alt="AUROC of every detector on the filled patch and on real single-view geometry"><figcaption>Left: on the benchmark's filled patch, the only detectors that score are the ones trained on the fill, and their per-image control scores the same, so they found the patch, not the missing support. Right: on real single-view geometry from clean views, the cross-view tokens carry the answer, a per-image head does not, and the fill-trained head scores below native confidence.</figcaption></figure>

<section class="measure" id="method">
  <h2><span class="num">03 / THE METHOD</span>Change what the detector must learn</h2>
  <p>A head trained on the benchmark's fills learns to see the fill, which a single image reveals even when inpainted. So the label was changed to the thing the patch imitates, multi-view support, and the input to clean views: a pixel that only one view supports is the positive class, a pixel two or more views support is the negative class. The decisive control is the same head on per-image tokens, before any cross-view attention.</p>
  <p>The method of record goes one step further and needs no ground truth at all. The backbone is run once on clean views; each predicted point is projected into the other views with the predicted cameras and counted as supported where it lands on their predicted surface. That count is the label. The reprojection test is the teacher, and the head has to generalise better than its teacher to be worth having. It does: on phone orbits and tourist photographs, where the model's own cameras are least reliable, the head beats the test it learned from.</p>
</section>
<figure class="fig"><img src="assets/charts/method_light.png" alt="Schematic of the label-free support head"><figcaption>The label-free support head. 3.3 M parameters on VGGT-1B (1.7 M on Pi3X and Pi3): a projection per layer, two convolutions at patch resolution, one logit per patch, under 1% of the backbone's FLOPs. Trained in ten minutes on one GPU, 1500 steps, on the 79 DTU training scans, which the benchmark never uses.</figcaption></figure>

<section class="measure" id="results">
  <h2><span class="num">04 / RESULTS</span>Three backbones, four datasets the heads never saw</h2>
  <p>Clean inputs of 3, 5 and 10 views, three rotations of the view set per scene. DTU: 18 benchmark scans. CO3Dv2: 18 phone orbits. ETH3D: nine building-scale DSLR scans with laser ground truth. Phototourism: nine landmarks from Internet photo collections (British Museum, Florence cathedral, Lincoln Memorial, London Bridge, Milan cathedral, Mount Rushmore, Piazza San Marco, Sagrada Familia, St Paul's) with people and vehicles in front of the monument and COLMAP depth as ground truth. Pre-registered pass conditions, written before the heads were trained, with a 0.02 margin against the control: both pass on every dataset and backbone.</p>
  <div class="kpis">
    <div class="kpi"><b>{v('vis/dtu/vggt_oiavisself/auroc', nd=2)}</b><span>AUROC of the label-free head on DTU (VGGT); native {v('vis/dtu/vggt/auroc', nd=2)}, per-image control {v('vis/dtu/vggt_oiavispatch/auroc', nd=2)}</span></div>
    <div class="kpi"><b>{v('vis/co3d/vggt_oiavisself/auroc', nd=2)}</b><span>on CO3Dv2 phone orbits; the per-image control falls to {v('vis/co3d/vggt_oiavispatch/auroc', nd=2)}, below chance</span></div>
    <div class="kpi"><b>{v('vis/imc/pi3x_oiavisself/auroc', nd=2)}</b><span>on tourist photographs (Pi3X); reprojection consistency {v('vis/imc/pi3x_reproj/auroc', nd=2)}, native {v('vis/imc/pi3x/auroc', nd=2)}</span></div>
    <div class="kpi"><b>{pct('vis/dtu/vggt_oiavisself/cur')}</b><span>of single-view pixels pass the head's 90%-kept threshold on DTU, against {pct('vis/dtu/vggt/cur')} for native confidence</span></div>
  </div>
</section>
<figure class="fig"><img src="assets/charts/vis_light.png" alt="AUROC per detector family, per backbone and dataset"><figcaption>Every detector family on the four datasets, one panel per backbone, pooled over scenes and view counts with 95% intervals. Runs P4-vis, P4-vis-co3d, P4-vis-eth3d, P7-vis-imc.</figcaption></figure>
<section class="measure">
  <div class="tabs" data-tabs="vis" role="group" aria-label="Backbone">
    <button type="button" aria-pressed="true" data-k="vggt">VGGT-1B</button><button type="button" aria-pressed="false" data-k="pi3x">Pi3X</button><button type="button" aria-pressed="false" data-k="pi3">Pi3</button><button type="button" aria-pressed="false" data-k="da3">Depth Anything 3</button>
  </div>
  <div class="tabpane on" data-pane="vis" data-k="vggt">{vis_table('vggt')}</div>
  <div class="tabpane" data-pane="vis" data-k="pi3x">{vis_table('pi3x')}</div>
  <div class="tabpane" data-pane="vis" data-k="pi3">{vis_table('pi3')}</div>
  <div class="tabpane" data-pane="vis" data-k="da3">{vis_table('da3')}</div>
  <p class="small">AUROC and CUR pooled over scenes and view counts (3 and 5 views on CO3Dv2, where ten views around an object leave no single-view pixel). Every paired difference quoted on this page is at the permutation floor unless a p-value is given; the full tables with intervals are in the supplement and in <code>results/P3-stats.md</code>. The last row is the head trained on one half of the target dataset, still without labels, and scored on the other half.</p>
</section>

<section class="measure" id="controls">
  <h2><span class="num">05 / CONTROLS AND ABLATIONS</span>What the head reads, and what it does not</h2>
  <p><strong>Not the frame edge.</strong> On the lab rig single-view pixels sit at the image border, and a per-image head learns that: the distance to the border alone scores {v('vis/dtu/vggt_border/auroc', nd=2)} on DTU. Off the rig it collapses ({v('vis/co3d/vggt_border/auroc', nd=2)} on CO3Dv2), and so does the per-image control ({v('vis/co3d/vggt_oiavispatch/auroc', nd=2)}, indistinguishable from the border score). Depth edges score {rng(['vis/dtu/vggt_dedge/auroc', 'vis/co3d/vggt_dedge/auroc', 'vis/eth3d/vggt_dedge/auroc'])}. The cross-view head keeps {rng(['vis/co3d/vggt_oiavisself/auroc', 'vis/eth3d/vggt_oiavisself/auroc'], 2)} on the same scenes.</p>
  <p><strong>Where the signal lives.</strong> A linear probe on the same cross-view tokens scores within 0.015 of the convolutional head, so the information is in the features; the head is a reader. One probe per VGGT layer says where it appears: layer 4 of 24 scores {v('layers/dtu/4', nd=2)} on DTU and {v('layers/co3d/4', nd=2)} on CO3Dv2, like the per-image control; layer 17 scores {v('layers/dtu/17', nd=2)} and {v('layers/co3d/17', nd=2)}. Support becomes readable in the second half of the aggregator.</p>
  <p><strong>Stable.</strong> Two extra initialisation seeds are within 0.005 of the head. With 20 views, twice the most it saw in training, the head scores {v('ext/vggt_oiavisself', nd=3)} (VGGT) while native confidence falls to {v('ext/vggt', nd=2)}.</p>
  <p><strong>Adaptation without labels.</strong> Because the labels come from the backbone itself, the head can be trained on the target capture. On one half of a dataset and scored on the other half, this helps where the backbone's own geometry on that capture is reliable (Pi3X on CO3Dv2: {v('vis/co3d/pi3x_oiavisself/auroc', nd=3)} to {v('vis/co3d/pi3x_oiavisself_x/auroc', nd=3)}; on ETH3D: {v('vis/eth3d/pi3x_oiavisself/auroc', nd=3)} to {v('vis/eth3d/pi3x_oiavisself_x/auroc', nd=3)}), changes nothing on the landmarks, and hurts VGGT on CO3Dv2 ({v('vis/co3d/vggt_oiavisself/auroc', nd=3)} to {v('vis/co3d/vggt_oiavisself_x/auroc', nd=3)}), whose reprojection test, the teacher, scores only {v('vis/co3d/vggt_reproj/auroc', nd=2)} there. The DTU-trained head is the safe default.</p>
</section>
<figure class="fig"><img src="assets/charts/layers_light.png" alt="Linear probe AUROC per VGGT layer on four datasets"><figcaption>One linear probe per VGGT aggregator layer, trained on ground-truth visibility labels. The early layers carry only the per-image cue that fails off the rig.</figcaption></figure>
<figure class="fig"><img src="assets/charts/adapt_light.png" alt="Label-free adaptation: DTU-trained head against a head trained on the target capture"><figcaption>Label-free adaptation on CO3Dv2, ETH3D and the landmarks (runs P6-adapt-co3d, P6-adapt-eth3d, P7-adapt-imc): the target-trained head is scored on scenes it never saw.</figcaption></figure>

<section class="measure" id="abstain">
  <h2><span class="num">06 / ABSTAINING</span>What a user gets from it</h2>
  <p>A detector is used with a threshold. CUR sets it on the test scene itself, which a deployed filter cannot do, so from the per-pixel scores (run P6-dump) three deployable readings. Ranking: keeping the half of the pixels the head trusts most lowers the depth error of what remains by 15 to 44%, and the area under the single-view rate of the kept pixels against coverage is within 0.02 of the oracle on every dataset. Fixed bands, set before any evaluation at 0.3 and 0.7: the head marks {pct('bands/dtu/vggt_oiavisself/frac_observed')} of the pixels observed on DTU with {pct('bands/dtu/vggt_oiavisself/risk_observed')} single-view inside, {pct('bands/co3d/vggt_oiavisself/frac_observed')} with {pct('bands/co3d/vggt_oiavisself/risk_observed')} on CO3Dv2. Frozen thresholds: a threshold set to keep 90% of the supported pixels of the <em>other</em> scenes lets {pct('bands/dtu/vggt_oiavisself/cur_loso')} of single-view pixels through on DTU and {pct('bands/co3d/vggt_oiavisself/cur_loso')} on CO3Dv2, against {pct('bands/dtu/vggt/cur_loso')} and {pct('bands/co3d/vggt/cur_loso')} for native confidence. VGGT's native confidence has no scale that transfers between datasets at all; the head's probability does.</p>
  {bands_table()}
</section>
<figure class="fig"><img src="assets/charts/rc_light.png" alt="Risk-coverage curves on four datasets"><figcaption>Keep the pixels a detector trusts most: how much single-view geometry survives at each coverage (VGGT-1B). Lower is better; the head stays near the floor on every dataset.</figcaption></figure>

<section class="measure" id="limit">
  <h2><span class="num">07 / THE LIMIT</span>A consistent invention passes</h2>
  <div class="warnbox"><p>The benchmark defines unsupported geometry as a surface no view shows; the detector was trained on surfaces only one view shows. Put back to the benchmark's filled patches, the visibility heads score AUROC {rng(['onfill/dtu/vggt_oiavisself', 'onfill/dtu/vggt_oiavis'], 2)} on DTU and {rng(['onfill/co3d/vggt_oiavisself', 'onfill/co3d/vggt_oiavis'], 2)} on CO3Dv2, below native confidence on the same rows. A head trained to find single-view geometry does not fire on a patch that every view shows, for the reason the reprojection and perturbation baselines do not: the invented surface is coherent across views and across the features. The head certifies agreement between views, and a consistent invention has it. The detector answers "which pixels lack corroboration", not "which surfaces were invented"; on real captures, the first question is the one with an answer.</p></div>
</section>
<figure class="fig"><img src="assets/charts/onfill_light.png" alt="Visibility detectors on the filled patch"><figcaption>Run P6-vis-on-fill: the visibility detectors scored zero-shot on the benchmark's own filled patches.</figcaption></figure>

<section class="measure" id="video">
  <h2><span class="num">08 / VIDEO</span>Three minutes, no audio</h2>
  <p>The story of this page in order: the question, the protocol, the leak, why the cheap fixes and the fill-trained head fail, the label-free head, the four datasets, the controls, what abstaining buys, and the limit. Every chart in it is recomputed from the released CSVs when the video is built.</p>
</section>
<div class="video"><video controls preload="metadata" playsinline poster="assets/charts/vis_dark.png"><source src="assets/overview.mp4" type="video/mp4">Your browser cannot play this video. <a href="assets/overview.mp4">Download it</a> instead.</video></div>
<section class="measure">
  <h3>Earlier videos</h3>
  <p class="small">The <a href="assets/demo.mp4">benchmark demo</a> (the masking protocol run live on DTU and CO3Dv2 data, then the five models' recorded outputs) and the <a href="assets/walkthrough.mp4">end-to-end walkthrough</a> (the CPU self-test live, the recorded runs, the charts, chaptered) predate the support head and show the benchmark side in detail.</p>
</section>

<section class="measure">
  <h2><span class="num">09 / REPRODUCIBILITY</span>Every number from a logged run</h2>
  <p>Each table cell maps to a run id whose CSV, configuration and log are in the repository (<code>results/</code>), together with the dated log of every failed run and every correction (<code>LOG.md</code>, <code>EXPERIMENTS.md</code>). The pass conditions of the heads were written into that registry, with dates and commit hashes, before the heads were trained. The benchmark is 1000 lines of Python with no training and runs on one 16 GB GPU in about eight hours; a head trains in ten minutes on one H200. Everything on this page, including the charts and the video, is regenerated by the scripts in <code>scripts/</code>.</p>
</section>

<section class="measure" id="cite">
  <h2><span class="num">CITATION</span>Cite</h2>
  <button type="button" id="copybib" class="seg" style="padding:.4rem .9rem;font:inherit;font-size:.9rem;cursor:pointer;background:none;color:var(--ink)">Copy BibTeX</button>
<pre id="bib">@inproceedings{{observeimagineabstain2027,
  title     = {{Observe, Imagine, Abstain: Detecting Hallucinated Geometry
               in Feed-Forward 3D Reconstruction}},
  author    = {{Anonymous}},
  booktitle = {{CVPR submission}},
  year      = {{2027}}
}}</pre>
</section>

<footer>
  Built from the released result files by <code>scripts/make_page.py</code>; no number on this page was typed by hand except the citation year. Charts by <code>scripts/make_assets.py</code>, video by <code>scripts/make_video.py</code>.
</footer>
</main>
<script>{JS}</script>
</body>
</html>
""")
    return "".join(H)


if __name__ == "__main__":
    out = os.path.join(ROOT, "docs", "index.html")
    open(out, "w", encoding="utf-8", newline="\n").write(page())
    print("->", out)
