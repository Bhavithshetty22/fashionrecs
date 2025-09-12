"""
Resume training script that handles category changes.
Loads old checkpoint and adapts it to new category structure.
"""

import os
import torch
import glob
from train_df2_maskcrnn import train, build_model, DF2_CLASSES

def find_latest_checkpoint(checkpoint_dir="./checkpoints_df2"):
    """Find the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory {checkpoint_dir} not found!")
        return None
    
    # Look for epoch checkpoints
    epoch_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    
    if not epoch_files:
        print("❌ No epoch checkpoints found!")
        return None
    
    # Sort by epoch number
    def extract_epoch_number(filename):
        basename = os.path.basename(filename)
        if "epoch_" in basename and "_step_" in basename:
            try:
                epoch_part = basename.split("epoch_")[1].split("_step_")[0]
                return int(epoch_part)
            except:
                return 0
        return 0
    
    latest_file = max(epoch_files, key=extract_epoch_number)
    latest_epoch = extract_epoch_number(latest_file)
    
    print(f"✅ Found latest checkpoint: {latest_file}")
    print(f"📊 Latest epoch: {latest_epoch}")
    
    return latest_file, latest_epoch

def adapt_checkpoint_to_new_categories(old_checkpoint_path, new_model, device):
    """
    Adapt old checkpoint to new category structure.
    Loads compatible parts and initializes new parts.
    """
    print("🔄 Adapting checkpoint to new category structure...")
    
    # Load old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location=device)
    old_state_dict = old_checkpoint["model"]
    
    # Get new model state dict
    new_state_dict = new_model.state_dict()
    
    # Create adapted state dict
    adapted_state_dict = {}
    
    # Copy compatible layers
    for key, value in old_state_dict.items():
        if key in new_state_dict:
            if value.shape == new_state_dict[key].shape:
                adapted_state_dict[key] = value
                print(f"   ✅ Copied: {key}")
            else:
                print(f"   ⚠️  Shape mismatch: {key} - old: {value.shape}, new: {new_state_dict[key].shape}")
                # Initialize new layers with pretrained weights
                adapted_state_dict[key] = new_state_dict[key]
        else:
            print(f"   ⚠️  Key not found in new model: {key}")
    
    # Add new layers that weren't in old checkpoint
    for key, value in new_state_dict.items():
        if key not in adapted_state_dict:
            adapted_state_dict[key] = value
            print(f"   🆕 Added new layer: {key}")
    
    # Create new checkpoint - don't load optimizer state to avoid size mismatches
    new_checkpoint = {
        "model": adapted_state_dict,
        "optimizer": {},  # Start with fresh optimizer state
        "lr_scheduler": {},  # Start with fresh scheduler state
        "epoch": old_checkpoint.get("epoch", 0)
    }
    
    # Save adapted checkpoint
    adapted_path = old_checkpoint_path.replace(".pth", "_adapted.pth")
    torch.save(new_checkpoint, adapted_path)
    print(f"💾 Saved adapted checkpoint: {adapted_path}")
    
    return adapted_path

def continue_training_with_adaptation():
    """Continue training with category adaptation."""
    print("🚀 Fashion AI Pipeline - Continue Training (Adapted)")
    print("=" * 60)
    
    # Find latest checkpoint
    checkpoint_info = find_latest_checkpoint()
    if not checkpoint_info:
        print("❌ Cannot continue training without a checkpoint!")
        return
    
    latest_checkpoint, latest_epoch = checkpoint_info
    
    print(f"\n📈 Continuing from Epoch {latest_epoch + 1}")
    print(f"💾 Using checkpoint: {latest_checkpoint}")
    print(f"📊 Progress will show: step/total_steps (e.g., 2/10000)")
    
    # Check if we need to adapt the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build new model with all categories
    print("🏗️ Building new model with all categories...")
    new_model = build_model(num_classes=len(DF2_CLASSES) + 1, pretrained=False)
    
    # Try to load checkpoint directly first
    try:
        old_checkpoint = torch.load(latest_checkpoint, map_location=device)
        new_model.load_state_dict(old_checkpoint["model"])
        print("✅ Checkpoint loaded directly - no adaptation needed!")
        adapted_checkpoint = latest_checkpoint
    except Exception as e:
        print(f"⚠️ Direct loading failed: {e}")
        print("🔄 Adapting checkpoint to new category structure...")
        adapted_checkpoint = adapt_checkpoint_to_new_categories(latest_checkpoint, new_model, device)
    
    # Dataset paths
    images_train = r"E:\datasets\deepfashion2\coco\images\train"
    ann_train = r"E:\datasets\deepfashion2\coco\annotations\instances_train.json"
    images_val = r"E:\datasets\deepfashion2\coco\images\val"
    ann_val = r"E:\datasets\deepfashion2\coco\annotations\instances_val.json"
    
    # Training parameters
    print("\n⚙️ Training Configuration:")
    print(f"   • Epochs: 12 (continuing from {latest_epoch + 1})")
    print(f"   • Batch size: 4")
    print(f"   • Learning rate: 0.01")
    print(f"   • Max train images per epoch: 10,000")
    print(f"   • Max val images: 2,000")
    print(f"   • Device: {device}")
    print(f"   • Categories: {len(DF2_CLASSES)} (13 clothing + 10 accessories)")
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    try:
        train(
            images_dir_train=images_train,
            ann_train=ann_train,
            images_dir_val=images_val,
            ann_val=ann_val,
            epochs=12,
            batch_size=4,
            lr=0.01,
            num_workers=8,
            use_masks=False,
            resize_shorter=320,
            output_dir="./checkpoints_df2",
            device_str="cuda",
            resume_from=adapted_checkpoint,
            checkpoint_every_steps=8000,
            max_train_images=10000,  # Reduced dataset for faster training
            max_val_images=2000,
        )
        
        print("\n✅ Training completed successfully!")
        print("🎉 Your model is ready for the Fashion AI Pipeline!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    continue_training_with_adaptation()
