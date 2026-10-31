"""Anonymised code + results archive for the CVPR supplementary (issue #11). Run from anywhere:
  python scripts/make_supp_archive.py [--out supp/hallubench3d_supp.zip] [--with-examples] [--scrub TOKEN ...]

Includes scripts/, requirements.txt, README/EXPERIMENTS/LOG, every result CSV, config, consistency JSON and run
log. Identifying strings (GitHub user, Kaggle user, author name, project URL) are replaced by ANON in every text
file, and the script fails if any token survives. Kernel logs that Kaggle wrote are included only after scrubbing.
Composite images (results/*/ex_*, about 200 MB) are left out unless --with-examples is given.
Paper sources, docs/ and the demo video are never included: they are submitted through their own channels."""
import argparse
import glob
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCRUB = ["https://prajak002.github.io/DTU-3D-reconstruction/", "https://github.com/prajak002/DTU-3D-reconstruction",
                 "prajak002", "popsensharma", "prajaksen", "Prajak Sen", "Prajak"]
TEXT_EXT = (".py", ".md", ".txt", ".csv", ".json", ".log", ".cfg", ".yaml", ".yml", ".toml")


def files_to_pack(with_examples):
    top = ["README.md", "EXPERIMENTS.md", "LOG.md", "requirements.txt", ".gitignore"]
    out = [os.path.join(ROOT, f) for f in top if os.path.exists(os.path.join(ROOT, f))]
    out += sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        is_example = (os.sep + "ex_") in f
        if is_example and not with_examples:
            continue
        if is_example or f.endswith((".csv", ".json", ".log")):
            out.append(f)
    return out


def scrub_text(s, tokens):
    for t in sorted(tokens, key=len, reverse=True):
        s = s.replace(t, "ANON")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "supp", f"hallubench3d_supp_{time.strftime('%Y%m%d')}.zip"))
    ap.add_argument("--with-examples", action="store_true", help="also pack the composite PNGs under results/*/ex_*")
    ap.add_argument("--scrub", nargs="*", default=None, help="identifying strings to replace (default: project defaults)")
    a = ap.parse_args()
    tokens = a.scrub if a.scrub is not None else DEFAULT_SCRUB
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    leaks, n_text, n_bin = [], 0, 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files_to_pack(a.with_examples):
            arc = "hallubench3d/" + os.path.relpath(f, ROOT).replace(os.sep, "/")
            if f.endswith(TEXT_EXT):
                s = open(f, encoding="utf-8", errors="replace").read()
                s = scrub_text(s, tokens)
                for t in tokens:
                    if t.lower() in s.lower():
                        leaks.append((arc, t))
                z.writestr(arc, s); n_text += 1
            else:
                z.write(f, arc); n_bin += 1
    size = os.path.getsize(a.out) / 1e6
    print(f"wrote {os.path.relpath(a.out, ROOT)}: {n_text} text files (scrubbed), {n_bin} binary files, {size:.1f} MB")
    if leaks:
        print("identifying strings survived scrubbing (case-insensitive match):")
        for arc, t in leaks:
            print(f"  {arc}: {t!r}")
        sys.exit(1)
    print("no identifying strings left in text files; remember to check the paper PDF metadata separately")


if __name__ == "__main__":
    main()
