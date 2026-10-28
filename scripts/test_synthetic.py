"""CPU test of hallu3d data/mask/metric pipeline on a synthetic DTU-format scene.
Two fake models:
  oracle : correct depth (x2 scale), conf drops where input pixels are hidden  -> auroc ~1, cur ~0
  blind  : correct depth on visible, smooth guess in hidden, constant-ish conf -> auroc ~0.5, cur ~0.9
"""
import os, sys, shutil, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import hallu3d as H

ROOT = "/tmp/claude-0/fake_dtu"
shutil.rmtree(ROOT, ignore_errors=True)
scan, light = 3, 3
for d in ["Cameras/train", f"Rectified/scan{scan}_train", f"Depths/scan{scan}_train"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

h, w = 128, 160
K_low = np.array([[180.0, 0, 80.5], [0, 180.0, 64.5], [0, 0, 1]])   # depth-res intrinsics, like Cameras/train

def look_at(cam_pos, target=np.zeros(3)):
    z = target - cam_pos; z /= np.linalg.norm(z)
    up = np.array([0, -1.0, 0]); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z])                    # world->cam rotation rows
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ cam_pos
    return E

def render_depth(E, K):
    # scene: plane z=0.0 facing camera side (cameras at z<0), sphere bump center (0,0,-0.15) r=0.3
    Einv = np.linalg.inv(E); o = Einv[:3, 3]
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dw = dc @ Einv[:3, :3].T
    t_plane = (0.0 - o[2]) / dw[..., 2]
    P = o + t_plane[..., None] * dw
    t = np.where((np.abs(P[..., 0]) < 1.2) & (np.abs(P[..., 1]) < 1.0), t_plane, np.inf)
    c, r = np.array([0, 0, -0.15]), 0.3
    oc = o - c; b = (dw * oc).sum(-1); a = (dw * dw).sum(-1); cc = (oc * oc).sum() - r * r
    disc = b * b - a * cc
    ts = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    t = np.minimum(t, np.where(ts > 0, ts, np.inf))
    depth = np.where(np.isfinite(t), t, 0.0)    # dc has z=1 so t equals camera-frame depth
    return depth.astype(np.float32)

def write_pfm(path, img):
    with open(path, "wb") as f:
        f.write(b"Pf\n"); f.write(f"{img.shape[1]} {img.shape[0]}\n".encode()); f.write(b"-1\n")
        np.flipud(img).astype("<f").tofile(f)

rng = np.random.default_rng(0)
tex = (rng.random((512, 640, 3)) * 255).astype(np.uint8)
tex = cv2.GaussianBlur(tex, (0, 0), 2)
for v in range(49):
    ang = np.deg2rad(-40 + 80 * v / 48)
    cam = np.array([2.5 * np.sin(ang), 0.3 * np.cos(3 * ang), -2.5 * np.cos(ang)])
    E = look_at(cam)
    with open(os.path.join(ROOT, "Cameras/train", f"{v:08d}_cam.txt"), "w") as f:
        f.write("extrinsic\n" + "\n".join(" ".join(f"{x:.8f}" for x in row) for row in E) + "\n\nintrinsic\n"
                + "\n".join(" ".join(f"{x:.8f}" for x in row) for row in K_low) + "\n\n425 2.5\n")
    d = render_depth(E, K_low)
    write_pfm(os.path.join(ROOT, f"Depths/scan{scan}_train", f"depth_map_{v:04d}.pfm"), d)
    img = cv2.resize(((d > 0)[..., None] * 200).astype(np.uint8).repeat(3, -1), (640, 512), interpolation=cv2.INTER_NEAREST)
    img = np.clip(img.astype(int) // 2 + tex // 2, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(ROOT, f"Rectified/scan{scan}_train", f"rect_{v + 1:03d}_{light}_r5000.png"), img)

# -------- checks
root = H.find_dtu_root((ROOT,))
data = H.load_scan(root, scan, H.select_views(10), light)
print("K_img scale ok:", np.allclose(data["K_img"][0][0, 0], 720.0), "K_dep:", data["K_dep"][0][0, 0])
cfg = {**H.DEFAULT_CFG, "scans": [scan], "seeds": [0, 1], "radius_fracs": [0.15], "fills": ["gray", "noise"]}
masks = H.build_masks(data, 0, 0.15, cfg)
print("U per view:", masks["U"].sum((1, 2)), " S per view:", masks["S"].sum((1, 2))[:3])
print("overlap U&S (must be 0):", int((masks["U"] & masks["S"]).sum()))
print("consistency:", H.consistency_check(data, masks))
m2 = H.build_masks(data, 0, 0.15, cfg)
print("deterministic:", np.array_equal(masks["U"], m2["U"]))

class Fake:
    def __init__(self, kind): self.kind = kind
    def predict(self, imgs, model_hw, out_hw):
        # recover GT for this fake from the loaded data (test only)
        gt = data["depths"]
        hidden_img = (np.abs(imgs - data["imgs"]).sum(-1) > 1e-6)
        hid = hidden_img.reshape(len(imgs), out_hw[0], 4, out_hw[1], 4).mean((2, 4)) > 0.5
        bg = 3.0
        d = np.where(gt > 0, gt, bg) * 2.0
        if self.kind == "oracle":
            c = np.where(hid, 0.1, 5.0) + rng.random(d.shape) * 0.01
        else:
            guess = cv2.GaussianBlur(d.transpose(1, 2, 0), (0, 0), 6).transpose(2, 0, 1)
            d = np.where(hid, guess * 1.15, d)
            c = 5.0 + rng.random(d.shape) * 0.01
        return d.astype(np.float32), c.astype(np.float32)

csv = "/tmp/claude-0/fake_results.csv"
if os.path.exists(csv): os.remove(csv)
H.run_benchmark(root, ["oracle", "blind"], cfg, out_csv=csv, model_factory=Fake,
                save_examples_dir="/tmp/claude-0/fake_examples")
import pandas as pd
df = pd.read_csv(csv)
print(df.groupby("model")[["auroc", "cur", "cur_err", "absrel_S", "absrel_U", "absrel_U_ctrl", "scale"]].mean().round(3))
print(H.summarize(csv).to_string())
print("perm oracle vs blind auroc:", H.paired_permutation(df, "auroc", "oracle", "blind"))
