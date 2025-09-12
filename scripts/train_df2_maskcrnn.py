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

# Speed knobs for NVIDIA GPUs (RTX 40xx): enable autotune and TF32 fast paths
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


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
    # Accessories
    "shoes",
    "watch",
    "necklace",
    "bracelet",
    "ring",
    "earrings",
    "hat",
    "bag",
    "belt",
    "glasses",
    "scarf",
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
    def __init__(self, images_dir: str, ann_file: str, use_masks: bool = True, resize_shorter: int = 0, max_images: int = None):
        self.images_dir = images_dir
        self.use_masks = use_masks and HAS_COCO
        self.resize_shorter = resize_shorter
        self.max_images = max_images

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
        
        # Filter out images with no valid annotations
        valid_image_ids = []
        for img_id in self.image_ids:
            anns = self.img_id_to_anns.get(img_id, [])
            has_valid_ann = False
            for ann in anns:
                if ann.get("iscrowd", 0) == 0:
                    cat_id = ann["category_id"]
                    label = self.cat_id_to_contig.get(cat_id, 0)
                    if label > 0:
                        has_valid_ann = True
                        break
            if has_valid_ann:
                valid_image_ids.append(img_id)
        
        # Limit dataset size if specified
        if self.max_images and len(valid_image_ids) > self.max_images:
            # Shuffle and take first max_images for consistent selection
            import random
            random.seed(42)  # For reproducible results
            valid_image_ids = random.sample(valid_image_ids, self.max_images)
            
        self.image_ids = valid_image_ids

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
                        # If conversion fails, create empty mask
                        masks_np.append(np.zeros((height, width), dtype=np.uint8))
                else:
                    # No segmentation provided, create empty mask
                    masks_np.append(np.zeros((height, width), dtype=np.uint8))

        # Since we filtered out images with no valid annotations, this should not happen
        if len(boxes) == 0:
            raise ValueError(f"Image {img_id} has no valid annotations after filtering")

        img, masks_np, scale = self._maybe_resize(img, masks_np)
        if scale != 1.0:
            boxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]

        img_tensor = F.to_tensor(img)  # [0,1], CxHxW
        new_height, new_width = img_tensor.shape[1], img_tensor.shape[2]

        target: Dict[str, Any] = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        if self.use_masks:
            # Ensure we have masks for all boxes with correct dimensions
            while len(masks_np) < len(boxes):
                masks_np.append(np.zeros((new_height, new_width), dtype=np.uint8))
            # Resize any masks that don't match the new image size
            resized_masks = []
            for mask in masks_np[:len(boxes)]:
                if mask.shape != (new_height, new_width):
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(mask)
                    mask_pil = mask_pil.resize((new_width, new_height), PILImage.NEAREST)
                    mask = np.array(mask_pil)
                resized_masks.append(mask)
            masks_stack = np.stack(resized_masks, axis=0).astype(np.uint8)  # NxHxW
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
    resume_from: str = None,  # Path to checkpoint to resume from
    checkpoint_every_steps: int = 8000,  # step checkpointing safeguard
    max_train_images: int = None,  # Limit training images per epoch
    max_val_images: int = None,  # Limit validation images
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    train_dataset = DeepFashion2CocoDataset(images_dir_train, ann_train, use_masks=use_masks, resize_shorter=resize_shorter, max_images=max_train_images)
    val_dataset = DeepFashion2CocoDataset(images_dir_val, ann_val, use_masks=use_masks, resize_shorter=resize_shorter, max_images=max_val_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
        prefetch_factor=6 if num_workers else None,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    # Print dataset information (avoid Unicode emoji for Windows cp1252 consoles)
    print(f"Dataset Info:")
    print(f"   • Training images: {len(train_dataset)}")
    print(f"   • Validation images: {len(val_dataset)}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Steps per epoch: {len(train_loader)}")
    print(f"   • Total batches: {len(train_loader) * epochs}")
    print(f"   • Categories: {len(DF2_CLASSES)} (13 clothing + 10 accessories)")
    print("=" * 60)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
        prefetch_factor=6 if num_workers else None,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=len(DF2_CLASSES) + 1, pretrained=True).to(device)
    # Channels-last memory format can speed up on Ampere/Lovelace with AMP/TF32
    model = model.to(memory_format=torch.channels_last)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

    # Resume from checkpoint if provided
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, starting from epoch {start_epoch}")

    # Mixed precision scaler for faster training (new API)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        # Calculate total steps for this epoch
        total_steps = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        step_in_epoch = 0
        for images, targets in pbar:
            # Detection API expects a list of 3D tensors (C,H,W); don't convert to channels_last here
            images = [img.to(device, non_blocking=True) for img in images]
            tgts = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

            # Ensure all targets have masks
            for i, tgt in enumerate(tgts):
                if "masks" not in tgt:
                    # Add empty mask with correct dimensions
                    if len(tgt["boxes"]) > 0:
                        h, w = images[i].shape[1], images[i].shape[2]
                        tgt["masks"] = torch.zeros((len(tgt["boxes"]), h, w), dtype=torch.uint8, device=device)

            def forward_backward(batch_imgs, batch_tgts):
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    ld = model(batch_imgs, batch_tgts)
                    ls = sum(l for l in ld.values())
                scaler.scale(ls).backward()
                return ld, ls

            optimizer.zero_grad()
            try:
                loss_dict, losses = forward_backward(images, tgts)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # OOM fallback: micro-batch by halving until it fits
                micro_bs = max(1, len(images) // 2)
                success = False
                while micro_bs >= 1 and not success:
                    try:
                        optimizer.zero_grad()
                        total_loss = None
                        agg_loss_dict: Dict[str, torch.Tensor] = {}
                        for i in range(0, len(images), micro_bs):
                            sub_imgs = images[i:i+micro_bs]
                            sub_tgts = tgts[i:i+micro_bs]
                            ld, ls = forward_backward(sub_imgs, sub_tgts)
                            # accumulate for logging
                            for k, v in ld.items():
                                agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.detach()
                            total_loss = (total_loss + ls) if total_loss is not None else ls
                        scaler.step(optimizer)
                        scaler.update()
                        # replace for logging
                        loss_dict = {k: (v/ (len(images)/micro_bs)) for k, v in agg_loss_dict.items()}
                        losses = total_loss
                        success = True
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        micro_bs = micro_bs // 2
                        if micro_bs == 0:
                            # Save emergency checkpoint and re-raise
                            ckpt_path = os.path.join(output_dir, f"oom_epoch_{epoch}_step.pth")
                            torch.save({
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch
                            }, ckpt_path)
                            raise

            running_loss += losses.item()
            avg_loss = running_loss / (step_in_epoch + 1)
            pbar.set_postfix({
                **{k: f"{v.item():.3f}" for k, v in loss_dict.items()}, 
                'avg_loss': f"{avg_loss:.3f}",
                'step': f"{step_in_epoch + 1}/{len(train_loader)}"
            })
            step_in_epoch += 1

            # Step checkpointing safeguard
            if checkpoint_every_steps and (step_in_epoch % checkpoint_every_steps == 0):
                mid_ckpt = os.path.join(output_dir, f"epoch_{epoch}_step_{step_in_epoch}.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch
                }, mid_ckpt)

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
        torch.save({
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path} | train_loss={running_loss:.3f} val_loss={val_loss:.3f}")

    final_path = os.path.join(output_dir, "final.pth")
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epochs
    }, final_path)
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
        batch_size=4,            # safest for 8 GB VRAM
        lr=0.01,                 # LR for bs=4
        num_workers=8,           # higher parallelism for dataloader
        use_masks=False,         # detection-only for maximum speed today
        resize_shorter=320,      # smaller side for faster throughput
        output_dir="./checkpoints_df2",
        device_str="cuda",
        # Resume from latest epoch 2 checkpoint
        resume_from="./checkpoints_df2/epoch_2_step_40000.pth",
        checkpoint_every_steps=8000,
        # Limit dataset size for faster training
        max_train_images=20000,  # Use only 10k images per epoch
        max_val_images=4000,     # Use 2k validation images
    )