import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from tqdm import tqdm

try:
    from pycocotools import mask as maskUtils
    HAS_COCO = True
except Exception:
    HAS_COCO = False


DF2_CLASSES = [
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeved_dress",
    "long_sleeved_dress",
    "vest_dress",
    "sling_dress",
]


def _convert_cxcywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _polygons_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    if not HAS_COCO:
        raise RuntimeError("pycocotools not installed; cannot build masks from polygons.")
    rles = maskUtils.frPyObjects(segmentation, height, width)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)  # HxW or HxWxN
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return m.astype(np.uint8)


class DeepFashion2CocoDataset(Dataset):
    def __init__(self, images_dir: str, ann_file: str, use_masks: bool = True, resize_shorter: int = 0):
        self.images_dir = images_dir
        self.use_masks = use_masks and HAS_COCO
        self.resize_shorter = resize_shorter

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.id_to_img = {img["id"]: img for img in self.images}
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
        # Map DeepFashion2 category names -> contiguous ids 1..K
        self.class_to_contig = {name: i + 1 for i, name in enumerate(DF2_CLASSES)}
        self.cat_id_to_contig = {
            cat_id: self.class_to_contig.get(self.cat_id_to_name.get(cat_id, ""), 0)
            for cat_id in self.cat_id_to_name
        }

        anns = coco["annotations"]
        self.img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            self.img_id_to_anns.setdefault(a["image_id"], []).append(a)

        # Keep only images that have at least one annotation
        self.image_ids: List[int] = [img["id"] for img in self.images if img["id"] in self.img_id_to_anns]

    def __len__(self):
        return len(self.image_ids)

    def _maybe_resize(self, img: Image.Image, masks: List[np.ndarray]) -> Tuple[Image.Image, List[np.ndarray], float]:
        if not self.resize_shorter or self.resize_shorter <= 0:
            return img, masks, 1.0
        w, h = img.size
        scale = self.resize_shorter / min(h, w)
        if abs(scale - 1.0) < 1e-6:
            return img, masks, 1.0
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if masks:
            resized = []
            for m in masks:
                pil_m = Image.fromarray((m * 255).astype(np.uint8))
                pil_m = pil_m.resize((new_w, new_h), Image.NEAREST)
                resized.append((np.array(pil_m) > 127).astype(np.uint8))
            masks = resized
        return img, masks, scale

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.images_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        anns = self.img_id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        masks_np: List[np.ndarray] = []

        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            cat_id = a["category_id"]
            label = self.cat_id_to_contig.get(cat_id, 0)
            if label <= 0:
                continue

            xyxy = _convert_cxcywh_to_xyxy(a["bbox"])
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

            if self.use_masks:
                seg = a.get("segmentation")
                if isinstance(seg, list) and len(seg) > 0:
                    try:
                        m = _polygons_to_mask(seg, height, width)
                        masks_np.append(m)
                    except Exception:
                        # If conversion fails, skip mask for this instance
                        pass

        if len(boxes) == 0:
            # If image has no valid anns after filtering, return a dummy tiny box; dataloader filter could also be added
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]
            if self.use_masks:
                masks_np = [np.zeros((height, width), dtype=np.uint8)]

        img, masks_np, scale = self._maybe_resize(img, masks_np)
        if scale != 1.0:
            boxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]

        img_tensor = F.to_tensor(img)  # [0,1], CxHxW

        target: Dict[str, Any] = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        if self.use_masks and len(masks_np) == len(boxes):
            masks_stack = np.stack(masks_np, axis=0).astype(np.uint8)  # NxHxW
            target["masks"] = torch.from_numpy(masks_stack)

        # area / iscrowd (optional but recommended)
        area = []
        for b in boxes:
            area.append((b[2] - b[0]) * (b[3] - b[1]))
        target["area"] = torch.tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace heads
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def train(
    images_dir_train: str,
    ann_train: str,
    images_dir_val: str,
    ann_val: str,
    epochs: int = 12,
    batch_size: int = 2,
    lr: float = 0.005,
    num_workers: int = 2,
    use_masks: bool = True,
    resize_shorter: int = 800,
    output_dir: str = "./checkpoints_df2",
    device_str: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    train_dataset = DeepFashion2CocoDataset(images_dir_train, ann_train, use_masks=use_masks, resize_shorter=resize_shorter)
    val_dataset = DeepFashion2CocoDataset(images_dir_val, ann_val, use_masks=use_masks, resize_shorter=resize_shorter)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    model = build_model(num_classes=len(DF2_CLASSES) + 1, pretrained=True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", ncols=100)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, tgts)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()}, total=f"{running_loss:.3f}")

        lr_scheduler.step()

        # quick val pass (loss only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", ncols=100):
                images = [img.to(device) for img in images]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, tgts)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path} | train_loss={running_loss:.3f} val_loss={val_loss:.3f}")

    final_path = os.path.join(output_dir, "final.pth")
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
    print(f"Training finished. Final checkpoint: {final_path}")


if __name__ == "__main__":
    # Edit these paths
    images_train = r"E:\datasets\deepfashion2\coco\images\train"
    ann_train = r"E:\datasets\deepfashion2\coco\annotations\instances_train.json"
    images_val = r"E:\datasets\deepfashion2\coco\images\val"
    ann_val = r"E:\datasets\deepfashion2\coco\annotations\instances_val.json"

    # If pycocotools is unavailable on your system, set use_masks=False to train detection-only
    train(
        images_dir_train=images_train,
        ann_train=ann_train,
        images_dir_val=images_val,
        ann_val=ann_val,
        epochs=12,
        batch_size=2,
        lr=0.005,
        num_workers=2,
        use_masks=True,          # set to False if pycocotools is problematic
        resize_shorter=800,      # shorter side resize; set 0 to disable
        output_dir="./checkpoints_df2",
        device_str="cuda",
    )