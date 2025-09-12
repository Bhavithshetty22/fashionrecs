"""
Continue training script for Fashion AI Pipeline
Resumes training from the latest checkpoint with reduced dataset.
"""

import os
import glob
from train_df2_maskcrnn import train

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

def continue_training():
    """Continue training from the latest checkpoint."""
    print("🚀 Fashion AI Pipeline - Continue Training")
    print("=" * 50)
    
    # Find latest checkpoint
    checkpoint_info = find_latest_checkpoint()
    if not checkpoint_info:
        print("❌ Cannot continue training without a checkpoint!")
        return
    
    latest_checkpoint, latest_epoch = checkpoint_info
    
    print(f"\n📈 Continuing from Epoch {latest_epoch + 1}")
    print(f"💾 Using checkpoint: {latest_checkpoint}")
    
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
    print(f"   • Device: CUDA")
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 50)
    
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
            resume_from=latest_checkpoint,
            checkpoint_every_steps=10000,
            max_train_images=10000,  # Reduced dataset for faster training
            max_val_images=2000,
        )
        
        print("\n✅ Training completed successfully!")
        print("🎉 Your model is ready for the Fashion AI Pipeline!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    continue_training()
