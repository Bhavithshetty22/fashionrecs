import os
import argparse
import json
import time
from pathlib import Path
import torch
import cv2
from torchvision.transforms import functional as F

# Robust import of training module regardless of how script is launched
try:
    from .train_df2_maskrcnn import build_model, DF2_CLASSES  # when run as module: -m scripts.infer_df2
except Exception:
    try:
        from scripts.train_df2_maskrcnn import build_model, DF2_CLASSES  # when package resolves
    except Exception:
        import importlib.util
        here = os.path.dirname(__file__)
        train_path = os.path.join(here, "train_df2_maskcrnn.py")
        spec = importlib.util.spec_from_file_location("train_df2_maskrcnn", train_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        build_model, DF2_CLASSES = mod.build_model, mod.DF2_CLASSES


def main():
    parser = argparse.ArgumentParser("DeepFashion2 inference")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="out_vis.jpg", help="Output image path")
    parser.add_argument("--score_thr", type=float, default=0.35, help="Score threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (avoid CUDA DLL issues)")
    parser.add_argument("--max_dets", type=int, default=10, help="Max detections to draw (by score)")
    parser.add_argument("--no_masks", action="store_true", help="Disable mask overlay for clarity")
    parser.add_argument("--save_crops_dir", type=str, default="fashion_out", help="Directory to save detection crops")
    parser.add_argument("--run_color_after", action="store_true", help="Run color classification on saved crops")
    parser.add_argument("--color_min_score", type=float, default=0.60, help="Min score for color classification candidates")
    parser.add_argument("--color_nms_iou", type=float, default=0.5, help="IoU threshold for suppressing overlapping color candidates")
    parser.add_argument("--color_top_k", type=int, default=0, help="If no candidates selected, fallback to top-K detections by score")
    parser.add_argument("--run_pattern_local", action="store_true", help="Run local HuggingFace pattern classifier on saved crops and write unified JSON")
    parser.add_argument("--run_season_local", action="store_true", help="Run local HuggingFace season classifier on saved crops and write unified JSON")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Load checkpoint first to detect how many classes it was trained with
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    # Torchvision detection heads: out_features == num_classes (including background)
    cls_w = sd.get("roi_heads.box_predictor.cls_score.weight")
    if cls_w is None:
        raise RuntimeError("Unexpected checkpoint format: missing roi_heads.box_predictor.cls_score.weight")
    num_classes_ckpt = int(cls_w.shape[0])

    model = build_model(num_classes=num_classes_ckpt, pretrained=False)
    model.load_state_dict(sd)
    model.to(device).eval()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = img[:, :, ::-1]
    pil = F.to_pil_image(img_rgb)
    tensor = F.to_tensor(pil).to(device)

    with torch.no_grad():
        out = model([tensor])[0]

    vis = img.copy()
    H, W = vis.shape[:2]

    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    masks = out.get("masks")
    if masks is not None:
        masks = masks.cpu().numpy()  # Nx1xHxW

    # Keep only indices above threshold, then take top-K by score
    keep_idx = (scores >= args.score_thr).nonzero()[0]
    keep_idx = keep_idx[np.argsort(scores[keep_idx])[::-1]]
    if args.max_dets and len(keep_idx) > args.max_dets:
        keep_idx = keep_idx[:args.max_dets]

    # Ensure crop output directory exists if we are saving crops
    save_crops = args.save_crops_dir is not None and len(args.save_crops_dir) > 0
    saved_crops = []
    saved_meta = []  # metadata for unified analysis JSON
    if save_crops:
        os.makedirs(args.save_crops_dir, exist_ok=True)

    # Prepare candidate indices for color classification: score gate + NMS to avoid duplicates
    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    color_candidates = [i for i in keep_idx.tolist() if scores[i] >= args.color_min_score]
    selected_for_color = []
    for idx in color_candidates:
        b = boxes[idx].astype(int)
        suppress = False
        for kept in selected_for_color:
            if iou(b, boxes[kept].astype(int)) >= args.color_nms_iou:
                suppress = True
                break
        if not suppress:
            selected_for_color.append(idx)

    print(f"Detections above score_thr: {len(keep_idx)} | candidates>=color_min_score: {len(color_candidates)} | selected after NMS: {len(selected_for_color)}")
    if len(selected_for_color) == 0 and args.color_top_k > 0:
        fallback = keep_idx.tolist()[: args.color_top_k]
        selected_for_color = fallback
        print(f"No candidates passed thresholds; falling back to top-{args.color_top_k} detections: {selected_for_color}")

    for i in keep_idx.tolist():
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(labels[i])
        # Map label to known names if available; fall back if checkpoint had fewer classes
        if 0 < cls_id <= len(DF2_CLASSES):
            cls_name = DF2_CLASSES[cls_id - 1]
        else:
            cls_name = f"cls_{cls_id}"

        color = (0, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{cls_name}:{scores[i]:.2f}"
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 4)
        cv2.rectangle(vis, (x1, y_text - th - 2), (x1 + tw + 2, y_text), color, -1)
        cv2.putText(vis, label_text, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if (not args.no_masks) and masks is not None and i < masks.shape[0]:
            m = (masks[i, 0] > 0.5).astype("uint8")
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            colored = (np.stack([m * 0, m * 255, m * 0], axis=-1)).astype("uint8")
            vis = cv2.addWeighted(vis, 1.0, colored, 0.3, 0)

        # Save crop only if selected for color classification
        if save_crops and (i in selected_for_color):
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            if x2c > x1c and y2c > y1c:
                crop = img[y1c:y2c, x1c:x2c]
                crop_name = f"det_{i:02d}_crop.jpg"
                crop_path = os.path.join(args.save_crops_dir, crop_name)
                try:
                    cv2.imwrite(crop_path, crop)
                    saved_crops.append(crop_path)
                    saved_meta.append({
                        "crop_file": crop_name,
                        "det_index": i,
                        "label": cls_name,
                        "score": float(scores[i]),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    })
                except Exception as exc:
                    print(f"Failed to save crop {crop_name}: {exc}")

    if save_crops:
        print(f"Saved {len(saved_crops)} crop(s) to: {os.path.abspath(args.save_crops_dir)}")

    cv2.imwrite(args.out, vis)
    print(f"Saved visualization: {args.out}")

    # Optionally run color classification on saved crops
    color_results = {}
    if args.run_color_after and saved_crops:
        try:
            # Robust import similar to model import above
            try:
                from .classify_crops_colors import load_model, predict_color
            except Exception:
                try:
                    from scripts.classify_crops_colors import load_model, predict_color
                except Exception:
                    import importlib.util
                    here = os.path.dirname(__file__)
                    cls_path = os.path.join(here, "classify_crops_colors.py")
                    spec = importlib.util.spec_from_file_location("classify_crops_colors", cls_path)
                    mod = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(mod)
                    load_model, predict_color = mod.load_model, mod.predict_color

            model, processor, device = load_model("prithivMLmods/Fashion-Product-baseColour")
            for cp in saved_crops:
                try:
                    pred = predict_color(Path(cp), model, processor, device)  # type: ignore[name-defined]
                    color_results[os.path.basename(cp)] = pred
                except Exception as exc:
                    print(f"Color prediction failed for {cp}: {exc}")

            colors_json = os.path.join(args.save_crops_dir, "fashion_colors.json")
            with open(colors_json, "w", encoding="utf-8") as f:
                json.dump(color_results, f, indent=2)
            print(f"Saved color predictions for {len(color_results)} crops: {colors_json}")
        except Exception as exc:
            print(f"Failed running color classification: {exc}")

    # Optionally run local pattern and season classification and write unified analysis JSON
    if (args.run_pattern_local or args.run_season_local) and saved_crops:
        try:
            from PIL import Image as _Image
            import torch as _torch
            try:
                from transformers import AutoImageProcessor as _AutoImageProcessor
            except Exception:
                from transformers import AutoFeatureExtractor as _AutoImageProcessor
            from transformers import AutoModelForImageClassification as _AutoCls
            from transformers import SiglipForImageClassification as _SiglipForImageClassification

            pattern_results_local = {}
            season_results_local = {}
            
            # Load pattern model if requested
            if args.run_pattern_local:
                pattern_model_name = "IrshadG/Clothes_Pattern_Classification_v2"
                proc = _AutoImageProcessor.from_pretrained(pattern_model_name)
                pmodel = _AutoCls.from_pretrained(pattern_model_name)
                pmodel.eval()

            # Load season model if requested
            if args.run_season_local:
                season_model_name = "prithivMLmods/Fashion-Product-Season"
                season_proc = _AutoImageProcessor.from_pretrained(season_model_name)
                season_model = _SiglipForImageClassification.from_pretrained(season_model_name)
                season_model.eval()
                season_id2label = {0: "Fall", 1: "Spring", 2: "Summer", 3: "Winter"}

            for cp in saved_crops:
                try:
                    im = _Image.open(cp).convert("RGB")
                    
                    # Pattern classification
                    if args.run_pattern_local:
                        inputs = proc(images=im, return_tensors="pt")
                        with _torch.no_grad():
                            logits = pmodel(**inputs).logits
                            probs = _torch.nn.functional.softmax(logits, dim=-1)
                            pred_idx = int(_torch.argmax(probs, dim=-1).item())
                            label = pmodel.config.id2label[pred_idx]
                            conf = float(probs[0, pred_idx].item())
                        pattern_results_local[os.path.basename(cp)] = {"label": label, "score": round(conf, 4)}
                    
                    # Season classification
                    if args.run_season_local:
                        try:
                            # Try to use improved classifier first
                            from improved_season_classifier import ImprovedSeasonClassifier
                            improved_classifier = ImprovedSeasonClassifier()
                            
                            # Get clothing type from detection results
                            clothing_type = "unknown"
                            for meta in saved_meta:
                                if meta["crop_file"] == os.path.basename(cp):
                                    clothing_type = meta["label"]
                                    break
                            
                            result = improved_classifier.classify_season(im, clothing_type)
                            season_results_local[os.path.basename(cp)] = {
                                "label": result["label"], 
                                "score": result["score"],
                                "method": result["method"]
                            }
                        except Exception as e:
                            print(f"Improved classifier failed, falling back to ML-only: {e}")
                            # Fallback to original ML approach
                            inputs = season_proc(images=im, return_tensors="pt")
                            with _torch.no_grad():
                                logits = season_model(**inputs).logits
                                probs = _torch.nn.functional.softmax(logits, dim=-1)
                                pred_idx = int(_torch.argmax(probs, dim=-1).item())
                                label = season_id2label[pred_idx]
                                conf = float(probs[0, pred_idx].item())
                            season_results_local[os.path.basename(cp)] = {
                                "label": label, 
                                "score": round(conf, 4),
                                "method": "ml_fallback"
                            }
                        
                except Exception as exc:
                    print(f"Local prediction failed for {cp}: {exc}")

            # Build unified analysis entries
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            analysis_path = os.path.join(args.save_crops_dir, f"fashion_analysis_{timestamp}.json")
            entries = []
            for meta in saved_meta:
                crop_file = meta["crop_file"]
                # Color: choose top color if we have per-class scores
                color_item = {"label": None, "score": None}
                c = color_results.get(crop_file)
                if isinstance(c, dict):
                    try:
                        scores_map = c.get("scores", {})
                        if scores_map:
                            top_color = max(scores_map, key=lambda k: scores_map[k])
                            color_item = {"label": top_color, "score": scores_map[top_color]}
                        else:
                            color_item = {"label": c.get("label"), "score": c.get("score")}
                    except Exception:
                        color_item = {"label": c.get("label"), "score": c.get("score")}

                p = pattern_results_local.get(crop_file, {"label": None, "score": None})
                s = season_results_local.get(crop_file, {"label": None, "score": None})
                entries.append({
                    "crop_file": crop_file,
                    "det_label": meta["label"],
                    "det_score": meta["score"],
                    "box": meta["box"],
                    "color": color_item,
                    "pattern": p,
                    "season": s,
                })

            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump({"items": entries}, f, indent=2)
            print(f"Saved unified fashion analysis with {len(entries)} items: {analysis_path}")
        except Exception as exc:
            print(f"Failed running local pattern/season classification: {exc}")


if __name__ == "__main__":
    import numpy as np
    main()