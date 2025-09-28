import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification


DEFAULT_MODEL = "prithivMLmods/Fashion-Product-Season"


ID2LABEL = {
    0: "Fall",
    1: "Spring", 
    2: "Summer",
    3: "Winter"
}


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiglipForImageClassification.from_pretrained(model_name)
    model.to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor, device


def predict_season(image_path: Path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    top_idx = int(torch.tensor(probs).argmax().item())
    return {
        "label": ID2LABEL[top_idx],
        "score": round(probs[top_idx], 4),
        "scores": {ID2LABEL[i]: round(probs[i], 4) for i in range(len(probs))},
    }


def main():
    parser = argparse.ArgumentParser(description="Classify seasons for crop images in a directory")
    parser.add_argument("--crops_dir", type=str, default="fashion_out", help="Directory with crop images")
    parser.add_argument("--pattern", type=str, default="det_*_crop.jpg", help="Glob for crop images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--output", type=str, default="fashion_out/fashion_seasons.json", help="Output JSON path")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    crops = sorted(crops_dir.glob(args.pattern))
    if not crops:
        print(f"No crops found in {crops_dir} matching {args.pattern}")
        return

    model, processor, device = load_model(args.model)

    results = {}
    for crop_path in crops:
        try:
            pred = predict_season(crop_path, model, processor, device)
            results[crop_path.name] = pred
            print(f"{crop_path.name}: {pred['label']} ({pred['score']})")
        except Exception as exc:
            print(f"Failed on {crop_path}: {exc}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
