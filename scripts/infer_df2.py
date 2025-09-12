import os
import argparse
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

    cv2.imwrite(args.out, vis)
    print(f"Saved visualization: {args.out}")


if __name__ == "__main__":
    import numpy as np
    main()