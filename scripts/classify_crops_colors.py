import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification


DEFAULT_MODEL = "prithivMLmods/Fashion-Product-baseColour"


ID2LABEL = {
    0: "Beige", 1: "Black", 2: "Blue", 3: "Bronze", 4: "Brown", 5: "Burgundy",
    6: "Charcoal", 7: "Coffee Brown", 8: "Copper", 9: "Cream", 10: "Fluorescent Green",
    11: "Gold", 12: "Green", 13: "Grey", 14: "Grey Melange", 15: "Khaki", 16: "Lavender",
    17: "Lime Green", 18: "Magenta", 19: "Maroon", 20: "Mauve", 21: "Metallic",
    22: "Multi", 23: "Mushroom Brown", 24: "Mustard", 25: "Navy Blue", 26: "Nude",
    27: "Off White", 28: "Olive", 29: "Orange", 30: "Peach", 31: "Pink", 32: "Purple",
    33: "Red", 34: "Rose", 35: "Rust", 36: "Sea Green", 37: "Silver", 38: "Skin",
    39: "Steel", 40: "Tan", 41: "Taupe", 42: "Teal", 43: "Turquoise Blue", 44: "White", 45: "Yellow"
}


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiglipForImageClassification.from_pretrained(model_name)
    model.to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor, device


def predict_color(image_path: Path, model, processor, device):
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
    parser = argparse.ArgumentParser(description="Classify colors for crop images in a directory")
    parser.add_argument("--crops_dir", type=str, default="fashion_out", help="Directory with crop images")
    parser.add_argument("--pattern", type=str, default="det_*_crop.jpg", help="Glob for crop images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--output", type=str, default="fashion_out/fashion_colors.json", help="Output JSON path")
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
            pred = predict_color(crop_path, model, processor, device)
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


