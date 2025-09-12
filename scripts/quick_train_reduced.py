"""
Quick training script with reduced dataset and new categories.
Starts fresh but uses pretrained weights for faster convergence.
"""

import os
from train_df2_maskcrnn import train

def quick_train():
    """Quick training with reduced dataset."""
    print("🚀 Fashion AI Pipeline - Quick Training (Reduced Dataset)")
    print("=" * 60)
    
    print("📊 Training Configuration:")
    print("   • Starting: Fresh with pretrained weights")
    print("   • Epochs: 12")
    print("   • Batch size: 4")
    print("   • Learning rate: 0.01")
    print("   • Max train images per epoch: 10,000")
    print("   • Max val images: 2,000")
    print("   • Categories: 23 (13 clothing + 10 accessories)")
    print("   • Device: CUDA")
    print("   • Progress: Shows step/total_steps (e.g., 2/2500)")
    
    # Dataset paths
    images_train = r"E:\datasets\deepfashion2\coco\images\train"
    ann_train = r"E:\datasets\deepfashion2\coco\annotations\instances_train.json"
    images_val = r"E:\datasets\deepfashion2\coco\images\val"
    ann_val = r"E:\datasets\deepfashion2\coco\annotations\instances_val.json"
    
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
            resume_from=None,  # Start fresh
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
    quick_train()

