import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

DF2_CLASSES = [
    "short_sleeved_shirt","long_sleeved_shirt","short_sleeved_outwear","long_sleeved_outwear",
    "vest","sling","shorts","trousers","skirt","short_sleeved_dress","long_sleeved_dress","vest_dress","sling_dress",
]

def build_categories():
    return [{"id": i+1, "name": n, "supercategory": "clothes"} for i, n in enumerate(DF2_CLASSES)]

def get_wh(img_path: str):
    with Image.open(img_path) as im:
        return im.size  # (w,h)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--ann_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    with open(args.ann_file, "r", encoding="utf-8") as f:
        df2 = json.load(f)  # {"000001.jpg": {...}, ...}

    file_names = [fn for fn in df2.keys() if fn.lower().endswith((".jpg",".jpeg",".png"))]
    img_paths = [os.path.join(args.images_dir, fn) for fn in file_names]

    # Parallel size reads
    sizes = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(get_wh, pth): fn for fn, pth in zip(file_names, img_paths) if os.path.isfile(pth)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Reading image sizes"):
            fn = futs[fut]
            try:
                w, h = fut.result()
                sizes[fn] = (int(w), int(h))
            except Exception:
                pass

    images = []
    annotations = []
    categories = build_categories()
    ann_id = 1
    img_id_counter = 1

    for fn in file_names:
        img_path = os.path.join(args.images_dir, fn)
        if not os.path.isfile(img_path):
            continue
        if fn not in sizes:
            # fallback (open single-threaded)
            try:
                w, h = get_wh(img_path)
                sizes[fn] = (int(w), int(h))
            except Exception:
                continue

        w, h = sizes[fn]
        img_id = img_id_counter
        img_id_counter += 1

        images.append({"id": img_id, "file_name": fn, "width": w, "height": h})

        rec = df2[fn]
        pair_id = rec.get("pair_id", 0)
        for k, obj in rec.items():
            if k in ("source","pair_id"):
                continue
            cat = int(obj["category_id"])
            x1, y1, x2, y2 = obj["bounding_box"]
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat,
                "bbox": [float(x1), float(y1), bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
                "segmentation": obj.get("segmentation", []),
                "pair_id": pair_id,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories, "licenses": [], "info": {}}
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"Wrote {len(images)} images, {len(annotations)} annotations -> {args.out_file}")

if __name__ == "__main__":
    main()