import os, glob, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import orjson as _json
    def jloads(b: bytes): return _json.loads(b)
    def jdumps(o): return _json.dumps(o)
except Exception:
    import json as _json
    def jloads(b: bytes): return _json.loads(b.decode("utf-8"))
    def jdumps(o): return _json.dumps(o).encode("utf-8")

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--annos_dir", required=True)
parser.add_argument("--out_json", required=True)
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.annos_dir, "*.json")))
merged = {}

def load_one(fp: str):
    with open(fp, "rb") as f:
        data = jloads(f.read())
    img_name = data.get("image_path") or os.path.splitext(os.path.basename(fp))[0] + ".jpg"
    return img_name, data

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(load_one, fp) for fp in files]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Merging"):
        k, v = fut.result()
        merged[k] = v

with open(args.out_json, "wb") as f:
    f.write(jdumps(merged))

print(f"merged {len(files)} files -> {args.out_json}")