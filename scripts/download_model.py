"""Download the released model (~2.1 GB) next to the code, and verify its checksums.

    python scripts/download_model.py                       # -> models/solar_suitability_model_v1.0/
    python scripts/download_model.py --source <URL or folder>

The model is archived with the dataset at CEDA (see README). Files: model.json (~1 kB), mlp.npz (~2 MB),
rf.joblib (~2.1 GB). Interrupted downloads resume; finished files are checked against model.json.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

# TODO(release): replace with the CEDA archive URL of the model folder once the DOI is minted.
DEFAULT_SOURCE = "https://data.ceda.ac.uk/PLACEHOLDER/solar_suitability_v1.0/model/solar_suitability_model_v1.0"
FILES = ("model.json", "mlp.npz", "rf.joblib")


def fetch(src, dst):
    if os.path.isdir(os.path.dirname(src)) or os.path.exists(src):          # local folder (e.g. a mounted archive)
        shutil.copyfile(src, dst + ".part"); os.replace(dst + ".part", dst); return
    part = dst + ".part"; have = os.path.getsize(part) if os.path.exists(part) else 0
    from solarsiting.aef import USER_AGENT
    hdr = {"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})}
    req = urllib.request.Request(src, headers=hdr)
    with urllib.request.urlopen(req, timeout=300) as r:
        total = have + int(r.headers.get("Content-Length", 0))
        mode = "ab" if (have and r.status == 206) else "wb"
        done = have if mode == "ab" else 0
        with open(part, mode) as f:
            while True:
                b = r.read(1 << 23)
                if not b:
                    break
                f.write(b); done += len(b)
                if total > 1 << 26:
                    print(f"\r  {os.path.basename(dst)}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    if total > 1 << 26:
        print()
    os.replace(part, dst)


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="URL or local folder holding the model files")
    ap.add_argument("--out", default=os.path.join("models", "solar_suitability_model_v1.0"))
    a = ap.parse_args()
    if "PLACEHOLDER" in a.source:
        sys.exit("The model's CEDA archive link is not public yet (it will be added to this script and the README at "
                 "release).\nIf you already have the files, run:  python scripts/download_model.py --source <folder or URL>")
    from solarsiting.model import sha256_16
    os.makedirs(a.out, exist_ok=True)
    for f in FILES:
        dst = os.path.join(a.out, f)
        if not os.path.exists(dst):
            print(f"downloading {f} ...", flush=True)
            fetch(a.source.rstrip("/") + "/" + f, dst)
    meta = json.load(open(os.path.join(a.out, "model.json")))
    for f, expected in meta["sha256_16"].items():
        got = sha256_16(os.path.join(a.out, f))
        if got != expected:
            os.replace(os.path.join(a.out, f), os.path.join(a.out, f + ".bad"))
            sys.exit(f"checksum mismatch for {f} ({got} != {expected}); moved it to {f}.bad - please re-run")
        print(f"  {f}: checksum OK")
    print(f"model ready in {a.out}")


if __name__ == "__main__":
    main()
