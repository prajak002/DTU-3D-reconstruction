"""The paper's diagrammatic figures, drawn at print size with one palette shared across the paper:
  fig_teaser    page-1 teaser (full width): identical patches, native confidence vs the support head, headline ranges
  fig_arch      architecture (full width): frozen backbone, tapped cross-view layers, support head, bands; label-free
                training labels from the backbone's own reprojection; the per-image control tap
  fig_protocol  the benchmark and its matched split (column width): one 3D sphere hidden in every view; the
                three-sphere visibility grid
  fig_results   true-visibility results (column width): AUROC range over four backbones per detector and dataset

Image panels are cropped from figures already rendered from logged runs (fig1_teaser, fig_matched, fig_qual); every
number comes from docs/assets/numbers.json and docs/assets/matched.json. Layouts are in inches at print size.

  python scripts/make_fig_design.py [--out paper/figures]"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")

# one palette for the whole paper
IMAG = "#D1495B"     # imagined / unsupported
OBS = "#2A9D8F"      # observed / supported / ours
ABST = "#E9C46A"     # abstain
BASE = "#8D99AE"     # native confidence and other baselines
REPRO = "#5B8DB8"    # reprojection
CTRL = "#E39B4C"     # per-image control
INK = "#1D2B36"
MUTED = "#5C6B77"
FROZEN_FC, FROZEN_EC = "#E7EEF6", "#7C9CBF"
HEAD_FC, HEAD_EC = "#E3F3F1", OBS
PANEL_EC = "#C9D1D9"
LANE_FC, LANE_EC = "#FBFAF7", "#B9B3A6"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300, "mathtext.default": "regular"})

MATCHED_COLS = [(23, 384), (408, 771), (794, 1157), (1179, 1542), (1564, 1927)]
MATCHED_ROW = (66, 352)
QUAL_COLS = [(44, 317), (319, 592), (594, 867), (868, 1142), (1143, 1416), (1418, 1691)]
QUAL_ROWS = [(29, 248), (266, 421), (457, 640), (657, 849)]
TEASER_COLS = [(4, 332), (345, 674), (694, 994), (1028, 1356)]
TEASER_ROW = (60, 311)


def crop(png, x0, x1, y0, y1):
    im = Image.open(os.path.join(FIG, png)).convert("RGB")
    return np.asarray(im.crop((x0, y0, x1, y1)))


def mpanel(i):
    return crop("fig_matched.png", *MATCHED_COLS[i], *MATCHED_ROW)


def qpanel(r, c):
    return crop("fig_qual.png", *QUAL_COLS[c], *QUAL_ROWS[r])


def tpanel(c):
    return crop("fig1_teaser.png", *TEASER_COLS[c], *TEASER_ROW)


def R(W, H, x, y, w, h):
    """Axes rectangle in figure fractions from inches."""
    return [x / W, y / H, w / W, h / H]


def image(fig, img, rect, title=None, ec=PANEL_EC, lw=0.6, tcolor=INK, tfs=6.6, tweight="bold"):
    ax = fig.add_axes(rect)
    ax.imshow(img, interpolation="lanczos", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ec); sp.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=tfs, weight=tweight, color=tcolor, pad=2.5)
    return ax


class Canvas:
    """Full-figure axes in inch coordinates with box and arrow helpers."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off"); self.ax.set_xlim(0, W); self.ax.set_ylim(0, H)

    def box(self, x, y, w, h, t="", fc="white", ec=INK, lw=0.8, fs=6.0, weight="normal", color=INK, ls="-", r=0.05):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        if t:
            self.ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight=weight, color=color, zorder=3,
                         linespacing=1.18)

    def arrow(self, x0, y0, x1, y1, color=INK, lw=0.9, ls="-", ms=6, rad=0.0):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                          zorder=4, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def text(self, x, y, t, **kw):
        kw.setdefault("fontsize", 6.0); kw.setdefault("color", INK)
        self.ax.text(x, y, t, **kw)

    def image(self, img, x, y, w, h, title=None, **kw):
        return image(self.fig, img, R(self.W, self.H, x, y, w, h), title, **kw)

    def save(self, out):
        self.fig.savefig(out + ".pdf"); self.fig.savefig(out + ".png", dpi=220)
        plt.close(self.fig)


# ----------------------------------------------------------------------------- teaser
def mvec(i):
    """Panel i of fig_matched.pdf at source resolution: the embedded image and its outlines as vector polylines
    (pixel coordinates, colour, dashed), so the teaser carries sharp outlines instead of a crop of a rendered page."""
    import fitz
    page = fitz.open(os.path.join(FIG, "fig_matched.pdf"))[0]
    xref = sorted((page.get_image_rects(im[0])[0].x0, im[0]) for im in page.get_images(full=True))[i][1]
    r = page.get_image_rects(xref)[0]
    pix = fitz.Pixmap(page.parent, xref)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    lines = []
    for d in page.get_drawings():
        if d["type"] != "s" or not r.contains(d["rect"]):
            continue
        dashed = "[]" not in str(d.get("dashes"))
        cur = []
        for it in d["items"]:
            if it[0] != "l":
                continue
            if cur and (abs(cur[-1][0] - it[1].x) > 1e-3 or abs(cur[-1][1] - it[1].y) > 1e-3):
                lines.append((np.array(cur), d["color"], dashed)); cur = []
            if not cur:
                cur.append((it[1].x, it[1].y))
            cur.append((it[2].x, it[2].y))
        if cur:
            lines.append((np.array(cur), d["color"], dashed))
    out = []
    for xy, col, dashed in lines:
        px = np.stack([(xy[:, 0] - r.x0) / r.width * pix.width - 0.5, (xy[:, 1] - r.y0) / r.height * pix.height - 0.5], 1)
        out.append((px, col, dashed))
    return img, out


def _smooth(px, k=7):
    """Moving average along an outline: removes the pixel staircase of the low-resolution mask, keeps its shape."""
    if len(px) < 2 * k:
        return px
    closed = np.allclose(px[0], px[-1], atol=1e-3)
    ker = np.ones(k) / k
    if closed:
        q = np.concatenate([px[-k:-1], px, px[1:k]])
        out = np.stack([np.convolve(q[:, j], ker, "same") for j in (0, 1)], 1)[k - 1:k - 1 + len(px)]
        out[-1] = out[0]
        return out
    q = np.concatenate([np.repeat(px[:1], k // 2, 0), px, np.repeat(px[-1:], k // 2, 0)])
    return np.stack([np.convolve(q[:, j], ker, "valid") for j in (0, 1)], 1)


def vpanel(c, i, x, y, w, h, title, lw=1.15, photo=False, up=6, **kw):
    """One teaser panel. The source image is upsampled (Lanczos) so that it prints smooth; a photograph also gets an
    unsharp mask (display only, no content is generated). Outlines are vector, smoothed, with a white halo."""
    import matplotlib.patheffects as pe
    from PIL import ImageFilter
    img, lines = mvec(i)
    H0, W0 = img.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(img)).resize((W0 * up, H0 * up), Image.LANCZOS if photo else Image.BICUBIC)
    if photo:
        im = im.filter(ImageFilter.UnsharpMask(radius=1.2 * up, percent=110, threshold=2))
        im = im.filter(ImageFilter.UnsharpMask(radius=0.5 * up, percent=60, threshold=2))
    ax = c.image(np.asarray(im), x, y, w, h, title, **kw)
    ax.images[0].set_extent((-0.5, W0 - 0.5, H0 - 0.5, -0.5))
    for px, col, dashed in lines:
        px = _smooth(px)
        if not dashed:
            ax.plot(px[:, 0], px[:, 1], color="white", lw=lw + 0.6, solid_capstyle="round", solid_joinstyle="round", alpha=0.9)
        ax.plot(px[:, 0], px[:, 1], color=col, lw=lw, ls=(0, (3.0, 1.5)) if dashed else "-", solid_capstyle="round",
                solid_joinstyle="round", dash_capstyle="round",
                path_effects=[pe.Stroke(linewidth=lw + 0.6, foreground="white", alpha=0.9), pe.Normal()] if dashed else None)
    ax.set_xlim(-0.5, W0 - 0.5); ax.set_ylim(H0 - 0.5, -0.5)
    return ax


def _interior(mask):
    """(x, y) of the most interior pixel of the largest connected region of a boolean mask, or None."""
    import cv2
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    d = cv2.distanceTransform((lab == big).astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x), float(y)


def fig_teaser(N, M, out, P14=None):
    """Hero figure. Left: two captures fused in 3D from logged predictions (scripts/make_fig_gallery3d.py: predicted
    depth, dataset cameras), a static DTU capture and a video with a walking person, each beside the points the
    support head keeps. Right: the matched patches (2D) and the headline ranges."""
    c = Canvas(6.875, 2.45)
    T = json.load(open(os.path.join(ROOT, "docs", "assets", "teaser3d.json"), encoding="utf-8"))
    cw, ch, gx, gy, gx0 = 1.55, 1.06, 0.03, 0.05, 0.17
    box = dict(fc="black", ec="none", alpha=0.6, pad=1.4)
    rows3d = [("teaser3d_all.png", "teaser3d_keep.png", T, "static scene, 5 views",
               "red: seen by one view only ({:.0f}%)", "keeps {:.0f}% of red, {:.0f}% of all points", IMAG),
              ("teaser3d_video_all.png", "teaser3d_video_keep.png", T["video"], "video, 10 frames",
               "purple: moved during the video ({:.0f}%)", "keeps {:.0f}% of purple, {:.0f}% of all points", "#8E44AD")]
    calls = [("seen by one view only:\nno second view confirms it", "removed by the head", None),
             ("the walking person:\none copy per frame", "person removed", "also removed:\nsome background")]
    for r, (fa, fk, st, name, na, nk_, col) in enumerate(rows3d):
        call_all, call_keep, call_also = calls[r]
        ia = np.asarray(Image.open(os.path.join(FIG, fa)).convert("RGB"))
        ik = np.asarray(Image.open(os.path.join(FIG, fk)).convert("RGB"))
        nz = np.argwhere((ia < 250).any(2))                      # one crop per row: the extent of the full cloud
        (y0, x0), (y1, x1) = np.maximum(nz.min(0) - 10, 0), nz.max(0) + 10
        y = c.H - 0.15 - ch - r * (ch + gy)
        for k, (im, note) in enumerate(((ia, na.format(100 * st["bad_share"])),
                                        (ik, nk_.format(100 * st["bad_kept"], 100 * st["kept"])))):
            ax = c.fig.add_axes(R(c.W, c.H, gx0 + k * (cw + gx), y, cw, ch))
            ax.imshow(im[y0:y1, x0:x1], interpolation="lanczos"); ax.set_anchor("C")
            # callout: an arrow to the most interior point of the region it names, found in the renders themselves
            A, K_ = ia[y0:y1, x0:x1].astype(int), ik[y0:y1, x0:x1].astype(int)
            rgbc = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)])
            marked = np.abs(A - rgbc).sum(2) < 40                       # the red / purple points of the left render
            removed = (A < 250).any(2) & (K_ >= 250).all(2)             # drawn on the left, empty on the right
            if k == 0:
                mask, lab, lc = marked, call_all, col
            else:                                                       # where the marked points were and are no longer
                mask, lab, lc = marked & removed, call_keep, col
            pt = _interior(mask)
            if k == 1 and call_also:                                    # what else the head dropped: say so on the figure
                import cv2
                other = removed & ~(cv2.dilate(marked.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
                po = _interior(other)
                if po is not None:
                    ax.annotate(call_also, xy=po, xycoords="data", xytext=(0.03, 0.955), textcoords="axes fraction", ha="left",
                                va="top", fontsize=5.0, color=MUTED, zorder=30, annotation_clip=False, linespacing=1.15,
                                bbox=dict(fc="white", ec=MUTED, lw=0.5, boxstyle="round,pad=0.25"),
                                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, shrinkA=1, shrinkB=1, mutation_scale=5,
                                                connectionstyle="arc3,rad=-0.25"))
            if pt is not None:
                ax.annotate(lab, xy=pt, xycoords="data", xytext=(0.975, 0.955), textcoords="axes fraction",
                            ha="right", va="top", fontsize=5.5, weight="bold", color=lc, zorder=30, annotation_clip=False,
                            bbox=dict(fc="white", ec=lc, lw=0.6, pad=1.6, boxstyle="round,pad=0.25"),
                            arrowprops=dict(arrowstyle="-|>", color=lc, lw=0.9, shrinkA=1, shrinkB=1, mutation_scale=6,
                                            connectionstyle=f"arc3,rad={-0.18 if k else 0.18}"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            c.ax.add_patch(Rectangle((gx0 + k * (cw + gx), y), cw, ch, fc="none", ec=PANEL_EC, lw=0.6, zorder=6))
            # figure-level text: the image axes sit above the canvas axes
            c.fig.text((gx0 + k * (cw + gx) + 0.045) / c.W, (y + 0.045) / c.H, note, fontsize=5.3, color="white", weight="bold",
                       va="bottom", ha="left", bbox=box, zorder=20)
            if k == 0:              # row name, upright beside the row
                c.text(gx0 - 0.07, y + ch / 2, name, fontsize=6.2, color=INK, weight="bold", va="center", ha="center", rotation=90)
            if r == 0:
                c.text(gx0 + k * (cw + gx) + cw / 2, y + ch + 0.03, ("predicted points, all views fused", "points the support head keeps")[k],
                       fontsize=6.8, weight="bold", ha="center", va="bottom", color=(INK, OBS)[k])
    # matched patches
    xr = gx0 + 2 * cw + gx + 0.14
    c.ax.plot([xr - 0.07, xr - 0.07], [0.06, c.H - 0.04], color=PANEL_EC, lw=0.6)
    wr = c.W - 0.02 - xr
    ph = 0.75
    pw = ph * 242 / 194
    x0p = xr + (wr - 3 * pw - 2 * 0.04) / 2
    yp = c.H - 0.15 - ph
    tk = dict(tfs=6.6, lw=0.7)
    nk = dict(ha="center", va="top", fontsize=5.7, linespacing=1.2)
    vpanel(c, 0, x0p, yp, pw, ph, "input view (gray fill)", photo=True, **tk)
    c.text(x0p + pw / 2, yp - 0.05, "red: hidden in every view\ngreen: seen in another view", color=MUTED, **nk)
    vpanel(c, 2, x0p + pw + 0.04, yp, pw, ph, "native confidence", tcolor=IMAG, **tk)
    c.text(x0p + 1.5 * pw + 0.04, yp - 0.05, "dark on both patches:\nreads the patch", color=IMAG, weight="bold", **nk)
    vpanel(c, 4, x0p + 2 * pw + 0.08, yp, pw, ph, "support head (ours)", tcolor=OBS, **tk)
    c.text(x0p + 2.5 * pw + 0.08, yp - 0.05, "dark only where\nno view sees", color=OBS, weight="bold", **nk)
    # headline ranges
    aw = 2.05
    ax = c.fig.add_axes(R(c.W, c.H, xr + 0.78, 0.27, aw, yp - 0.27 - 0.30))
    V = N["vis"]
    MC = {k: M[k]["cells"] for k in ("DTU", "CO3Dv2")}
    bases = ("vggt", "pi3x", "pi3", "da3")
    rows = [("imagined vs observed patch", None, None),
            ("native", [MC[d][b]["auroc_m"][0] for d in MC for b in ("vggt", "pi3x", "pi3", "da3", "mapanything") if b in MC[d]], BASE),
            ("per-image", [MC[d][f"{b}_oiamatchselfpatch"]["auroc_m"][0] for d in MC for b in bases if f"{b}_oiamatchselfpatch" in MC[d]], CTRL),
            ("ours", [MC[d][f"{b}_oiamatchself"]["auroc_m"][0] for d in MC for b in bases if f"{b}_oiamatchself" in MC[d]], OBS),
            ("single-view geometry, unseen data", None, None),
            ("native", [V[d][b]["auroc"][0] for d in V for b in bases if b in V[d]], BASE),
            ("ours", [V[d][f"{b}_oiavisself"]["auroc"][0] for d in V for b in bases if f"{b}_oiavisself" in V[d]], OBS),
            ("real unseen surfaces, three datasets", None, None),
            ("native", [c["native"] for k, c in P14["cells"].items() if k.startswith(("DTU", "ETH3D", "TUM")) and c["qualifies"]], BASE),
            ("ours", [c["head"] for k, c in P14["cells"].items() if k.startswith(("DTU", "ETH3D", "TUM")) and c["qualifies"]], OBS)]
    yt, yl = [], []
    y = 0.0
    for k, (lab, vals, col) in enumerate(rows):
        if vals is None:                      # a heading takes a taller slot and sits at its bottom, clear of the bar above
            y -= 1.55 if k else 1.0
            ax.text(-0.74 / aw, y, lab, fontsize=6.0, weight="bold", color=INK, va="center", ha="left",
                    transform=ax.get_yaxis_transform(), zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))
            continue
        y -= 1.0
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [y, y], color=col, lw=4.2, solid_capstyle="round")
        ax.text(hi + 0.045, y, f"{lo:.2f}–{hi:.2f}", fontsize=5.8, va="center", color=col, weight="bold")
        yt.append(y); yl.append(lab)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.9, color=MUTED); ax.tick_params(axis="y", length=0, pad=1)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=":", zorder=1)
    ax.set_xlim(0.36, 1.24); ax.set_ylim(y - 0.75, -0.45)
    ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"]); ax.tick_params(axis="x", labelsize=5.9, length=2, pad=1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_bounds(0.36, 1.0)
    ax.set_xlabel("AUROC (0.5: chance)", fontsize=5.9, labelpad=1, x=0.36)
    c.fig.savefig(out + ".pdf", dpi=600); c.fig.savefig(out + ".png", dpi=600)
    plt.close(c.fig)


# ----------------------------------------------------------------------------- architecture
def fig_arch(out):
    c = Canvas(6.875, 2.45)
    # inputs
    cw, chh = 0.56, 0.56 * 0.76
    for k, img in enumerate([tpanel(0), qpanel(0, 0), tpanel(3)]):
        c.image(img, 0.06 + 0.07 * k, 1.42 - 0.15 * k, cw, chh, lw=0.8)
    c.text(0.44, 2.10, "N unposed\nphotos", ha="center", fontsize=7, weight="bold", linespacing=1.1)
    c.text(0.44, 0.92, "3 to 20 views\nno cameras", ha="center", fontsize=5.8, color=MUTED, linespacing=1.1)
    c.arrow(0.84, 1.47, 0.98, 1.47)
    # frozen backbone
    c.box(1.0, 0.70, 1.48, 1.62, fc=FROZEN_FC, ec=FROZEN_EC, lw=0.9, r=0.08)
    c.text(1.74, 2.18, "frozen 3D backbone", ha="center", fontsize=7.2, weight="bold")
    c.text(1.74, 2.05, "VGGT-1B, Pi3, Pi3X, Depth Anything 3", ha="center", fontsize=5.6, color=MUTED)
    c.box(1.08, 0.98, 0.40, 0.98, "per-image\nencoder", ec=FROZEN_EC, fs=6.0)
    c.box(1.56, 0.98, 0.84, 0.98, ec=FROZEN_EC)
    c.text(1.98, 1.86, "cross-view attention", ha="center", fontsize=6.0)
    taps = []
    for k, yy in enumerate(np.linspace(1.04, 1.70, 8)):
        tapped = k in (1, 3, 5, 7)
        c.ax.add_patch(Rectangle((1.62, yy), 0.72, 0.06, fc=OBS if tapped else "#C8D6E6", ec="none", zorder=3, alpha=0.9 if tapped else 1))
        if tapped:
            taps.append(yy + 0.03)
            c.text(1.98, yy + 0.03, f"layer {(4, 11, 17, 23)[len(taps) - 1]}", ha="center", va="center", fontsize=5.0, color="white",
                   weight="bold", zorder=4)
    c.arrow(1.48, 1.47, 1.56, 1.47, ms=5)
    c.box(1.08, 0.76, 1.32, 0.16, "depth, cameras, native confidence", ec=FROZEN_EC, fs=5.6)
    # support head
    hx, hy, hw, hh = 2.72, 1.12, 2.05, 1.12
    c.box(hx, hy, hw, hh, fc=HEAD_FC, ec=HEAD_EC, lw=1.0, r=0.08)
    c.text(hx + hw / 2, hy + hh - 0.11, "support head", ha="center", fontsize=7.2, weight="bold", color=OBS)
    c.text(hx + hw / 2, hy + hh - 0.23, "1.7 to 3.3 M parameters, under 1% of the backbone's FLOPs", ha="center", fontsize=5.4, color=MUTED)
    chips = [("LayerNorm\n+ 1×1 proj\nper layer", 0.48), ("Σ", 0.17), ("3×3 conv\n× 2", 0.34), ("logit\nper patch", 0.34),
             ("bilinear\nupsample", 0.36)]
    x, cy, ch = hx + 0.06, hy + 0.22, 0.52
    starts = []
    for k, (t, w) in enumerate(chips):
        c.box(x, cy, w, ch, t, ec=HEAD_EC, fs=5.6)
        starts.append(x)
        if k < len(chips) - 1:
            c.arrow(x + w, cy + ch / 2, x + w + 0.07, cy + ch / 2, color=OBS, ms=4.5)
        x += w + 0.07
    for k, yy in enumerate(taps):
        c.arrow(2.34, yy, starts[0], cy + 0.08 + k * 0.12, color=OBS, lw=0.7, ms=4)
    c.text(hx + hw / 2, hy + 0.09, "score = 1 − P(imagined) per pixel, one forward", ha="center", fontsize=5.6, color=MUTED)
    # output
    c.arrow(hx + hw, cy + ch / 2, 4.93, cy + ch / 2, color=OBS)
    ow = 0.80
    c.image(qpanel(0, 5), 4.95, 1.08, ow, ow * 219 / 273, "head output", tfs=6.4)
    for k, (lab, col, rg) in enumerate((("observed", OBS, "score > 0.7"), ("abstain", ABST, "0.3 to 0.7"), ("imagined", IMAG, "score < 0.3"))):
        yy = 1.86 - k * 0.27
        c.box(5.88, yy, 0.13, 0.13, fc=col, ec="none", r=0.02)
        c.text(6.06, yy + 0.085, lab, fontsize=6.4, va="center", weight="bold")
        c.text(6.06, yy - 0.01, rg, fontsize=5.4, va="center", color=MUTED)
    c.text(6.30, 1.04, "fixed in advance", fontsize=5.4, color=MUTED, ha="center", style="italic")
    # per-image control
    c.box(2.72, 0.70, 1.62, 0.30, "per-image control: the same head on the encoder\ntokens, before any cross-view attention", ec=CTRL,
          ls="--", fs=5.4)
    c.arrow(1.30, 0.98, 2.72, 0.86, color=CTRL, lw=0.8, ls="--", ms=4.5, rad=0.18)
    # training lane
    c.box(1.0, 0.05, 5.82, 0.56, fc=LANE_FC, ec=LANE_EC, ls="--", lw=0.8, r=0.06)
    c.text(1.08, 0.505, "training labels, no ground truth", fontsize=6.6, weight="bold", va="center")
    steps = [("run the backbone\nonce on clean views", 1.08, 0.86),
             ("project each predicted point into the\nother views, predicted depth and cameras", 2.06, 1.42),
             ("count the views whose\npredicted surface it hits", 3.60, 1.06)]
    for t, x, w in steps:
        c.box(x, 0.12, w, 0.30, t, ec=LANE_EC, fs=5.5)
    for x0, x1 in ((1.94, 2.06), (3.48, 3.60), (4.66, 4.76)):
        c.arrow(x0, 0.27, x1, 0.27, color=MUTED, ms=4.5)
    c.box(4.76, 0.285, 1.05, 0.14, "0 other views: imagined (+)", fc="#FBE8EA", ec=IMAG, fs=5.4, r=0.03)
    c.box(4.76, 0.12, 1.05, 0.14, "2 or more: observed (−)", fc="#E3F3F1", ec=OBS, fs=5.4, r=0.03)
    c.box(5.88, 0.12, 0.88, 0.30, "matched split: identical\npatches another view\nsees are negatives too", ec=OBS, fs=5.2)
    c.arrow(5.28, 0.43, 4.62, hy, color=OBS, lw=0.8, ls="--", ms=4.5, rad=-0.15)
    c.text(5.12, 0.84, "class-balanced BCE,\n1500 steps, 79 DTU scans", fontsize=5.3, color=OBS, ha="left", va="center")
    c.save(out)


# ----------------------------------------------------------------------------- protocol
def fig_protocol(out):
    c = Canvas(3.3, 2.25)
    pw, ph = 1.0, 1.0 * 251 / 328
    for k, (i, t) in enumerate(((0, "clean view"), (2, "sphere hidden, view 1"), (3, "same sphere, view 28"))):
        c.image(tpanel(i), 0.06 + k * 1.07, 1.30, pw, ph, t, tfs=6.2)
    c.text(1.65, 1.17, "(a) one 3D sphere on the ground-truth surface, hidden in every view and filled", ha="center", fontsize=5.9, color=MUTED)
    c.text(0.06, 1.04, "(b) the matched split: three identical patches per scene", fontsize=6.4, weight="bold")
    nv, ev = 6, 3
    gx0, cs, rh, gap = 1.12, 0.31, 0.17, 0.035
    gy_top = 0.94
    for i in range(nv):
        c.text(gx0 + i * (cs + gap) + cs / 2, gy_top - 0.02, "evidence" if i == ev else f"view {i + 1}", ha="center", va="top",
               fontsize=5.4, color=OBS if i == ev else MUTED, weight="bold" if i == ev else "normal")
    rows = [("A  imagined (U)", IMAG, [True] * nv), ("B  observed ($S_m$)", OBS, [i != ev for i in range(nv)]),
            ("D  keeps two per view", "#7E8A97", [i == ev for i in range(nv)])]
    y0 = gy_top - 0.32
    c.ax.add_patch(Rectangle((gx0 + ev * (cs + gap) - 0.02, y0 - 2 * (rh + 0.04) - 0.03), cs + 0.04, 3 * (rh + 0.04) - 0.01,
                             fc="#E3F3F1", ec=OBS, lw=0.6, zorder=1))
    for r, (lab, col, hidden) in enumerate(rows):
        yy = y0 - r * (rh + 0.04)
        c.text(gx0 - 0.05, yy + rh / 2, lab, ha="right", va="center", fontsize=5.8, color=col, weight="bold")
        for i, h in enumerate(hidden):
            xx = gx0 + i * (cs + gap)
            if h:
                c.box(xx, yy, cs, rh, "hidden", fc=col, ec="none", fs=5.0, color="white", r=0.025)
            else:
                c.box(xx, yy, cs, rh, "seen", fc="white", ec=col, fs=5.0, color=col, r=0.025)
    c.text(1.65, 0.03, "Same fill, size, depth and texture: one image cannot tell A from B.", ha="center", fontsize=5.8, color=MUTED,
           style="italic")
    c.save(out)


# ----------------------------------------------------------------------------- results
def fig_results(N, out):
    V = N["vis"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.85, bottom=0.17)
    dets = [("", "native confidence", BASE), ("_oiavisselfpatch", "per-image control", CTRL), ("_reproj", "reprojection", REPRO),
            ("_oiavisself", "support head (ours)", OBS)]
    dsets = [("dtu", "DTU"), ("co3d", "CO3Dv2"), ("eth3d", "ETH3D"), ("imc", "Phototourism")]
    bases = ("vggt", "pi3x", "pi3", "da3")
    yt, yl = [], []
    for g, (k, lab) in enumerate(dsets):
        for j, (suf, dl, col) in enumerate(dets):
            vals = [V[k][f"{b}{suf}"]["auroc"][0] for b in bases if f"{b}{suf}" in V[k]]
            y = -(g * 5.2 + j)
            ax.plot([min(vals), max(vals)], [y, y], color=col, lw=3.4, solid_capstyle="round", alpha=0.85)
            ax.plot(vals, [y] * len(vals), "o", ms=2.4, color="white", mec=col, mew=0.7)
        yt.append(-(g * 5.2 + 1.5)); yl.append(lab)
        if g:
            ax.axhline(-(g * 5.2 - 1.6), color="#E3E7EB", lw=0.6)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=":")
    ax.set_xlim(0.38, 1.005); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("AUROC, single-view vs supported geometry (range over 4 backbones)", fontsize=6.0, labelpad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    hs = [Line2D([0], [0], color=col, lw=3.4, label=l) for _, l, col in dets]
    fig.legend(handles=hs, loc="upper center", ncol=2, fontsize=6.2, frameon=False, bbox_to_anchor=(0.58, 1.01),
               handlelength=1.6, columnspacing=1.2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- small data figures (column width)
def _axstyle(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.grid(axis="y", color="#E3E7EB", lw=0.5); ax.set_axisbelow(True)


def _save(fig, out):
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
    plt.close(fig)


NAMES = {"vggt": "VGGT", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAny.", "da3": "DA3"}


def fig_leak(N, M, out):
    """CUR of native confidence per model: three datasets against clean pixels, and the matched split on DTU."""
    L = N["leak"]
    models = ["vggt", "pi3", "pi3x", "mapanything", "da3"]
    series = [("dtu", "DTU", "#B8C4D6"), ("co3d", "CO3Dv2", "#7F93AE"), ("eth3d", "ETH3D", "#4F6584")]
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.14)
    x = np.arange(len(models)); w = 0.19
    for i, (k, lab, col) in enumerate(series):
        v = [L[k][m]["cur"] for m in models]
        ax.bar(x + (i - 1.5) * w, [a[0] for a in v], w * 0.92, color=col, label=f"{lab}, vs clean",
               yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]], ecolor=MUTED, error_kw=dict(lw=0.5, capsize=1.2))
    mc = M["DTU"]["cells"]
    v = [mc[m]["cur_m"] for m in models]
    ax.bar(x + 1.5 * w, [a[0] for a in v], w * 0.92, color=IMAG, label="DTU, vs observed patch",
           yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]], ecolor=MUTED, error_kw=dict(lw=0.5, capsize=1.2))
    ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], fontsize=6.4)
    ax.set_ylim(0, 1); ax.set_ylabel("CUR (lower is better)", fontsize=6.2)
    _axstyle(ax)
    fig.legend(loc="upper center", ncol=2, fontsize=6.0, frameon=False, bbox_to_anchor=(0.55, 1.0), handlelength=1.0, columnspacing=1.0)
    _save(fig, out)


def fig_layers(N, out):
    """Linear probe on one VGGT layer at a time (true-visibility labels), per dataset."""
    Ly = N["layers"]
    xs = ["patch", "4", "11", "17", "23", "all"]
    labs = ["per-image\ntokens", "layer 4", "11", "17", "23", "all four"]
    cols = {"dtu": ("DTU", "#264653"), "co3d": ("CO3Dv2", OBS), "eth3d": ("ETH3D", "#8AB17D"), "imc": ("Phototourism", "#E39B4C")}
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.27)
    for k, (lab, col) in cols.items():
        v = [Ly[k][x][0] for x in xs]
        ax.plot(range(len(xs)), v, "-o", color=col, ms=2.8, lw=1.2, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=":")
    ax.axvspan(-0.3, 0.3, color="#FDF3E7", zorder=0)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(labs, fontsize=6.0)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("probe AUROC", fontsize=6.2)
    _axstyle(ax)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.8)
    _save(fig, out)


def fig_evidence(M, out):
    """Matched split: AUROC of imagined against observed patches by how much evidence the observed patch gets (DTU)."""
    C = M["DTU"]["cells"]
    buckets = [("auroc_m_ev1lo", "1 view,\nlow parallax"), ("auroc_m_ev1hi", "1 view,\nhigh parallax"), ("auroc_m_ev2", "2 views"),
               ("auroc_m_ev3p", "3 or more")]
    dets = [("", "native confidence", BASE, "o"), ("_oiamatchselfpatch", "per-image control", CTRL, "s"),
            ("_oiamatchself", "matched head, label-free", OBS, "D"), ("_oiamatch", "matched head, GT placement", "#264653", "^")]
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.25)
    for suf, lab, col, mk in dets:
        v = [np.mean([C[f"{b}{suf}"][k][0] for b in ("vggt", "pi3x", "pi3", "da3") if C.get(f"{b}{suf}", {}).get(k)]) for k, _ in buckets]
        ax.plot(range(4), v, "-", marker=mk, color=col, ms=3.2, lw=1.3, label=lab)
    if C.get("vggt_oiamatchselfev"):
        v = [C["vggt_oiamatchselfev"][k][0] for k, _ in buckets]
        ax.plot(range(4), v, "--", marker="D", color=OBS, mfc="white", ms=3.0, lw=1.0, label="VGGT label-free, one-view fine-tune")
    ax.axhline(0.75, color=IMAG, lw=0.5, ls="--")
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([b for _, b in buckets], fontsize=6.0)
    ax.set_ylim(0.35, 0.95); ax.set_ylabel("AUROC$_m$", fontsize=6.4)
    ax.set_xlabel("evidence for the observed patch (mean of four backbones, DTU)", fontsize=5.9, labelpad=1)
    _axstyle(ax)
    fig.legend(loc="upper center", ncol=2, fontsize=5.9, frameon=False, bbox_to_anchor=(0.55, 1.01), handlelength=1.4, columnspacing=1.0)
    _save(fig, out)


def fig_abstain(N, out):
    """(a) single-view rate of the kept pixels against coverage, VGGT on DTU; (b) accuracy of the fused DTU cloud."""
    rc, fu = N["rc"]["dtu"], N["fuse"]["dtu"]
    dets = [("vggt", "native", BASE), ("vggt_reproj", "reprojection", REPRO), ("vggt_oiavisselfpatch", "per-image control", CTRL),
            ("vggt_oiavisself", "support head", OBS)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.3, 1.55))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.78, bottom=0.24, wspace=0.45)
    for k, lab, col in dets:
        if k in rc:
            a.plot(rc[k]["coverage"], rc[k]["risk"], color=col, lw=1.2, label=lab)
    a.set_xlabel("coverage (share kept)", fontsize=5.9, labelpad=1); a.set_ylabel("single-view share", fontsize=5.9, labelpad=1)
    a.set_xlim(0, 1); a.set_ylim(0, None)
    a.set_title("(a) what a filter keeps", fontsize=6.3, weight="bold", loc="left", pad=2)
    covs = ["0.6", "0.7", "0.8", "0.9", "1.0"]
    for k, lab, col in dets:
        kf = k if k in fu else k.replace("oiavisselfpatch", "oiavispatch")
        if kf in fu:
            b.plot([float(c) for c in covs], [fu[kf][c]["accuracy"] for c in covs], "-o", color=col, ms=2.2, lw=1.2)
    if "vggt_random" in fu:
        b.plot([float(c) for c in covs], [fu["vggt_random"][c]["accuracy"] for c in covs], ":", color=MUTED, lw=1.0, label="random")
    b.set_xlabel("coverage", fontsize=5.9, labelpad=1); b.set_ylabel("fused-cloud accuracy", fontsize=5.9)
    b.set_title("(b) fused DTU cloud", fontsize=6.3, weight="bold", loc="left", pad=2)
    for x in (a, b):
        _axstyle(x)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=5.6, frameon=False, bbox_to_anchor=(0.55, 1.01), handlelength=1.2, columnspacing=0.8)
    _save(fig, out)


def fig_realocc(P14, out):
    """Real occlusion against dense references: (a) share of predicted points on surfaces no input view saw, per
    dataset, backbone and view count; (b) the head's AUROC gain over native confidence on those points, filled where
    the share is at least 0.5% and the gain is significant (p < 0.05), hollow otherwise. Hole-robust labels (P17,
    Gate 18): points in a gap of the reference depth are left out, which lets ETH3D be drawn with the other datasets."""
    g = P14["cells"]
    views = [3, 5, 10]
    cols = {"vggt": "#4F6584", "pi3x": "#8FA3BF", "pi3": "#C9D3E2", "da3": "#E3B8A0"}
    dsets = [("DTU benchmark", "o", None), ("DTU test", "D", "...."), ("ETH3D", "^", "xxxx"), ("TUM RGB-D", "v", "++++"),
             ("CO3Dv2", "s", "////")]
    groups = [(ds, b) for ds, _, _ in dsets for b in ("vggt", "pi3x", "pi3", "da3")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.3, 1.7))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.70, bottom=0.2, wspace=0.42)
    w = 0.84 / len(groups)
    mk = {d: m for d, m, _ in dsets}; ht = {d: h for d, _, h in dsets}
    for i, (ds, bb) in enumerate(groups):
        x = np.arange(3) + (i - (len(groups) - 1) / 2) * w
        v = [100 * g.get(f"{ds}:{n}:{bb}", {}).get("share", np.nan) for n in views]
        a.bar(x, v, w * 0.92, color=cols[bb], hatch=ht[ds], edgecolor="white", lw=0.25)
        ec = cols[bb] if bb != "pi3" else "#9AA8BD"
        for xi, n in zip(x, views):
            c = g.get(f"{ds}:{n}:{bb}")
            if not c:
                continue
            b.plot([xi], [c["diff"][0]], marker=mk[ds], ms=2.3, color=ec, mfc=ec if c["holds"] else "white", mew=0.6, ls="none")
    a.axhline(0.5, color=IMAG, lw=0.7, ls="--")
    a.set_yscale("log"); a.set_ylim(0.06, 14); a.set_yticks([0.1, 1, 10]); a.set_yticklabels(["0.1", "1", "10"])
    a.text(2.45, 0.54, "0.5%", fontsize=5.4, color=IMAG, ha="right", va="bottom", bbox=dict(fc="white", ec="none", pad=0.3))
    a.set_xticks(range(3)); a.set_xticklabels([f"{n} views" for n in views], fontsize=6.0)
    a.set_ylabel("% of points unobserved", fontsize=5.8, labelpad=1)
    a.set_title("(a) how often", fontsize=6.3, weight="bold", loc="left", pad=2)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_xticks(range(3)); b.set_xticklabels([f"{n} views" for n in views], fontsize=6.0)
    b.set_ylabel("head - native AUROC", fontsize=5.8, labelpad=1)
    b.set_title("(b) detecting them", fontsize=6.3, weight="bold", loc="left", pad=2)
    for x in (a, b):
        _axstyle(x)
    hs = [Patch(color=cols[bb], label=NAMES[bb]) for bb in ("vggt", "pi3x", "pi3", "da3")]
    hs += [Line2D([], [], marker=m, ls="none", color=MUTED, mfc="white", ms=3, label=d) for d, m, _ in dsets]
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=4.9, frameon=False, bbox_to_anchor=(0.55, 1.03), handlelength=1.0,
               columnspacing=0.7, labelspacing=0.2)
    fig.text(0.55, 0.815, "(b) filled: share >= 0.5% and gain significant (p < 0.05)", fontsize=5.0, color=MUTED, ha="center")
    _save(fig, out)


def fig_limit(N, out):
    """What separates the benchmark's filled patch (no view shows it) from supported geometry, DTU."""
    of, xm, rf = N["onfill"]["dtu"], N["xmodel"]["fill_dtu"], N["refill"]["dtu"]
    rows = [("native confidence", [of["vggt"][0], of["pi3x"][0]], BASE),
            ("per-image control", [of["vggt_oiavisselfpatch"][0], of["pi3x_oiavisselfpatch"][0]], CTRL),
            ("support head", [of["vggt_oiavisself"][0], of["pi3x_oiavisself"][0]], OBS),
            ("two models disagree", [v["auroc"][0] for v in xm.values()], REPRO),
            ("refill: fill the hole 3 ways", [rf[b]["auroc"][0] for b in rf], IMAG)]
    fig, ax = plt.subplots(figsize=(3.3, 1.45))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.97, bottom=0.21)
    for i, (lab, vals, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([min(vals), max(vals)], [y, y], color=col, lw=4.0, solid_capstyle="round")
        ax.plot(vals, [y] * len(vals), "o", ms=2.4, color="white", mec=col, mew=0.7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=":")
    ax.set_xlim(0.38, 1.0); ax.set_xlabel("AUROC, invented patch vs supported geometry (DTU)", fontsize=5.9, labelpad=1)
    _axstyle(ax); ax.grid(False); ax.spines["left"].set_visible(False)
    _save(fig, out)


def fig_stress(out):
    """Stress tests on DTU (blur, radius 0.18, seeds 0 and 1): CUR under an inpainted fill, fewer views, low parallax and a
    distractor image, against the matching reference cells of the full run. From the logged CSVs."""
    import sys
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import hallu3d as H
    ref = pd.read_csv(os.path.join(ROOT, "results", "table1_all_models.csv"))
    ref = ref[(ref.fill == "blur") & (ref.radius_frac == 0.18) & ref.seed.isin([0, 1])]
    tests = [("reference", None), ("inpainted\nfill", "inpaint10"), ("5 views", "blur5"), ("3 views", "blur3"),
             ("low\nparallax", "adjacent10"), ("distractor\nimage", "distractor10")]
    models = [("vggt", "VGGT", "#4F6584"), ("pi3x", "Pi3X", "#8D99AE"), ("da3", "DA3", "#C9D1D9")]
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.25)
    x = np.arange(len(tests)); w = 0.26
    for j, (m, lab, col) in enumerate(models):
        vals = []
        for _, t in tests:
            d = ref if t is None else pd.read_csv(os.path.join(ROOT, "results", "P2-stress", f"P2-stress_{t}.csv"))
            vals.append(H.cluster_bootstrap_ci(d[d.model == m], "cur", n_boot=2000))
        ax.bar(x + (j - 1) * w, [v[0] for v in vals], w * 0.92, color=col, label=lab, edgecolor="white", lw=0.3,
               yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], ecolor=MUTED, error_kw=dict(lw=0.5, capsize=1.0))
    ax.axvspan(-0.5, 0.5, color="#F3F5F7", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([t for t, _ in tests], fontsize=5.9)
    ax.set_xlim(-0.5, len(tests) - 0.5); ax.set_ylim(0, 1); ax.set_ylabel("CUR", fontsize=6.2)
    _axstyle(ax)
    fig.legend(loc="upper center", ncol=3, fontsize=6.0, frameon=False, bbox_to_anchor=(0.55, 1.01), handlelength=1.0)
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIG)
    a = ap.parse_args()
    N = json.load(open(os.path.join(ROOT, "docs", "assets", "numbers.json"), encoding="utf-8"))
    M = json.load(open(os.path.join(ROOT, "docs", "assets", "matched.json"), encoding="utf-8"))
    # real occlusion with the hole-robust labels (scripts/make_stats_p17.py); same cell layout as the earlier p14.json
    P17 = json.load(open(os.path.join(ROOT, "docs", "assets", "p17.json"), encoding="utf-8"))["cells"]
    P17.update(json.load(open(os.path.join(ROOT, "docs", "assets", "p18.json"), encoding="utf-8"))["cells"])   # TUM RGB-D
    P14 = {"cells": {k: dict(v, diff=[v["diff"], v["p"]]) for k, v in P17.items() if "adapted" not in k}}
    fig_teaser(N, M, os.path.join(a.out, "fig_teaser"), P14)
    fig_arch(os.path.join(a.out, "fig_arch"))
    fig_protocol(os.path.join(a.out, "fig_protocol"))
    fig_results(N, os.path.join(a.out, "fig_results"))
    fig_leak(N, M, os.path.join(a.out, "fig_leak"))
    fig_layers(N, os.path.join(a.out, "fig_layers"))
    fig_evidence(M, os.path.join(a.out, "fig_evidence"))
    fig_abstain(N, os.path.join(a.out, "fig_abstain"))
    fig_realocc(P14, os.path.join(a.out, "fig_realocc"))
    fig_limit(N, os.path.join(a.out, "fig_limit"))
    fig_stress(os.path.join(a.out, "fig_stress"))
    print("-> fig_teaser, fig_arch, fig_protocol, fig_results, fig_leak, fig_layers, fig_evidence, fig_abstain, fig_realocc, fig_limit in", a.out)


if __name__ == "__main__":
    main()
