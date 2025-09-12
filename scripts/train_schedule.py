import os
import sys

def train_part(part_num, resume_from=None):
    """Train one part of the dataset"""
    print(f"\n=== TRAINING PART {part_num} ===")
    
    # Import the training function
    sys.path.append(os.path.dirname(__file__))
    from train_df2_maskrcnn import train
    
    # Dataset paths
    images_train = r"E:\datasets\deepfashion2\coco\images\train"
    ann_train = r"E:\datasets\deepfashion2\coco\annotations\instances_train.json"
    images_val = r"E:\datasets\deepfashion2\coco\images\val"
    ann_val = r"E:\datasets\deepfashion2\coco\annotations\instances_val.json"
    
    # Training parameters
    epochs_per_part = 2  # Each part trains for 2 epochs (12 total / 6 parts = 2 each)
    total_epochs = 12
    
    if resume_from:
        start_epoch = int(resume_from.split('epoch_')[1].split('.')[0]) + 1
        epochs_to_run = min(epochs_per_part, total_epochs - start_epoch + 1)
        print(f"Resuming from epoch {start_epoch}, running {epochs_to_run} epochs")
    else:
        start_epoch = 1
        epochs_to_run = epochs_per_part
        print(f"Starting fresh, running {epochs_to_run} epochs")
    
    train(
        images_dir_train=images_train,
        ann_train=ann_train,
        images_dir_val=images_val,
        ann_val=ann_val,
        epochs=start_epoch + epochs_to_run - 1,  # End epoch
        batch_size=2,
        lr=0.005,
        num_workers=2,
        use_masks=True,
        resize_shorter=800,
        output_dir="./checkpoints_df2",
        device_str="cuda",
        resume_from=resume_from,
    )
    
    print(f"Part {part_num} completed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_schedule.py <part_number> [resume_from_checkpoint]")
        print("Example: python train_schedule.py 1")
        print("Example: python train_schedule.py 2 ./checkpoints_df2/epoch_2.pth")
        sys.exit(1)
    
    part_num = int(sys.argv[1])
    resume_from = sys.argv[2] if len(sys.argv) > 2 else None
    
    train_part(part_num, resume_from)