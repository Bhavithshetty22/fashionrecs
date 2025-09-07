import os
import sys
import shutil
import subprocess

import kagglehub

# 1) Download dataset
src_root = kagglehub.dataset_download("thusharanair/deepfashion2-original-with-dataframes")
print("Kaggle dataset path:", src_root)

# 2) Guess typical DeepFashion2 original layout inside that dataset
train_img = os.path.join(src_root, "train", "image")
train_json = os.path.join(src_root, "train", "annos", "train.json")
val_img   = os.path.join(src_root, "validation", "image")
val_json  = os.path.join(src_root, "validation", "annos", "val.json")

# 3) Target COCO layout
dst_root = r"E:\datasets\deepfashion2\coco"
dst_img_train = os.path.join(dst_root, "images", "train")
dst_img_val   = os.path.join(dst_root, "images", "val")
dst_ann_dir   = os.path.join(dst_root, "annotations")
dst_train_json = os.path.join(dst_ann_dir, "instances_train.json")
dst_val_json   = os.path.join(dst_ann_dir, "instances_val.json")

os.makedirs(dst_img_train, exist_ok=True)
os.makedirs(dst_img_val, exist_ok=True)
os.makedirs(dst_ann_dir, exist_ok=True)

# 4) Copy images
if os.path.isdir(train_img) and os.path.isdir(val_img):
    print("Copying train images...")
    shutil.copytree(train_img, dst_img_train, dirs_exist_ok=True)
    print("Copying val images...")
    shutil.copytree(val_img, dst_img_val, dirs_exist_ok=True)
else:
    print("Could not find train/validation image folders under:", src_root)
    print("Expected:", train_img, "and", val_img)
    sys.exit(1)

# 5) Convert annotations to COCO if official JSONs are present
conv_script = os.path.join(os.getcwd(), "evaluation", "deepfashion2_to_coco.py")

def run_convert(images_dir, ann_in, out_json):
    cmd = [
        sys.executable, conv_script,
        "--images_dir", images_dir,
        "--ann_file", ann_in,
        "--out_file", out_json,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

have_train_json = os.path.isfile(train_json)
have_val_json = os.path.isfile(val_json)

if have_train_json and have_val_json and os.path.isfile(conv_script):
    run_convert(dst_img_train, train_json, dst_train_json)
    run_convert(dst_img_val,   val_json,  dst_val_json)
    print("Done. COCO files at:", dst_train_json, "and", dst_val_json)
else:
    print("Official DeepFashion2 JSONs not found in this Kaggle dataset.")
    print("Searched for:", train_json, "and", val_json)
    print("You can:")
    print("  A) Download the official DeepFashion2 annotations (train.json/val.json) and rerun this script; or")
    print("  B) Share the dataframe schema so we can generate COCO JSONs from those dataframes.")