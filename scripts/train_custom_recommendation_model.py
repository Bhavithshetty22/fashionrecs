#!/usr/bin/env python3
"""
Custom Fashion Recommendation Model Training
===========================================

This script trains a recommendation model specifically optimized for the custom dataset.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pickle
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from custom_dataset_adapter import CustomFashionDatasetAdapter
from fashion_triplet_model import FashionEmbeddingModel, TripletLoss, TripletDataset

class OptimizedFashionEmbeddingModel(nn.Module):
    """Optimized model architecture for the custom dataset"""
    
    def __init__(self, feature_dim: int, embedding_dim: int = 256, dropout_rate: float = 0.3):
        super(OptimizedFashionEmbeddingModel, self).__init__()
        
        # Enhanced image encoder with residual connections
        self.image_encoder = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Enhanced feature encoder with attention
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(512 + 128, num_heads=8, dropout=dropout_rate)
        
        # Fusion layer with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Normalize embeddings
        self.normalize = True
    
    def forward(self, image, features):
        # Encode image
        image_emb = self.image_encoder(image)
        
        # Encode features
        feature_emb = self.feature_encoder(features)
        
        # Combine representations
        combined = torch.cat([image_emb, feature_emb], dim=1)
        
        # Apply attention
        attended, _ = self.attention(
            combined.unsqueeze(1), 
            combined.unsqueeze(1), 
            combined.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Fuse with residual connection
        embedding = self.fusion(attended) + combined[:, :512]  # Residual connection
        
        # Layer normalization
        embedding = self.layer_norm(embedding)
        
        # Normalize embeddings
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

class AdaptiveTripletLoss(nn.Module):
    """Adaptive triplet loss that adjusts margin based on difficulty"""
    
    def __init__(self, base_margin: float = 1.0, adaptive: bool = True):
        super(AdaptiveTripletLoss, self).__init__()
        self.base_margin = base_margin
        self.adaptive = adaptive
        self.margin = base_margin
    
    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        
        # Compute triplet loss
        loss = torch.nn.functional.relu(pos_dist - neg_dist + self.margin)
        
        # Adaptive margin adjustment
        if self.adaptive:
            # Increase margin for hard triplets
            hard_triplets = loss > 0
            if hard_triplets.sum() > 0:
                self.margin = min(self.margin * 1.01, 2.0)  # Gradually increase margin
            else:
                self.margin = max(self.margin * 0.99, 0.5)  # Gradually decrease margin
        
        return loss.mean()

class CustomFashionTrainer:
    """Custom trainer for the fashion recommendation model"""
    
    def __init__(self, model, dataset, device='auto'):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device if device != 'auto' else 
                                 ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self, dataloader, criterion, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            anchor_img = batch['anchor']['image'].to(self.device)
            anchor_feat = batch['anchor']['features'].to(self.device)
            positive_img = batch['positive']['image'].to(self.device)
            positive_feat = batch['positive']['features'].to(self.device)
            negative_img = batch['negative']['image'].to(self.device)
            negative_feat = batch['negative']['features'].to(self.device)
            
            # Forward pass
            anchor_emb = self.model(anchor_img, anchor_feat)
            positive_emb = self.model(positive_img, positive_feat)
            negative_emb = self.model(negative_img, negative_feat)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Update learning rate
        if scheduler:
            scheduler.step(avg_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def evaluate(self, dataloader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                anchor_img = batch['anchor']['image'].to(self.device)
                anchor_feat = batch['anchor']['features'].to(self.device)
                positive_img = batch['positive']['image'].to(self.device)
                positive_feat = batch['positive']['features'].to(self.device)
                negative_img = batch['negative']['image'].to(self.device)
                negative_feat = batch['negative']['features'].to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor_img, anchor_feat)
                positive_emb = self.model(positive_img, positive_feat)
                negative_emb = self.model(negative_img, negative_feat)
                
                # Compute loss
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs=100, batch_size=32, learning_rate=0.001, 
              weight_decay=1e-4, patience=15, min_delta=1e-4):
        """Train the model with early stopping"""
        
        # Create data loaders
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        criterion = AdaptiveTripletLoss(base_margin=1.0, adaptive=True)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        print(f"🔧 Device: {self.device}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📈 Learning rate: {learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        if self.learning_rates:
            plt.subplot(1, 3, 2)
            plt.plot(self.learning_rates, label='Learning Rate', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss distribution
        plt.subplot(1, 3, 3)
        plt.hist(self.train_losses, bins=20, alpha=0.7, label='Training', density=True)
        plt.hist(self.val_losses, bins=20, alpha=0.7, label='Validation', density=True)
        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.title('Loss Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training curves saved to: {save_path}")
        
        plt.show()
    
    def visualize_embeddings(self, sample_size=1000, save_path=None):
        """Visualize embeddings using t-SNE"""
        self.model.eval()
        
        # Sample data
        indices = torch.randperm(len(self.dataset))[:sample_size]
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for idx in indices:
                item = self.dataset[idx]
                image = item['image'].unsqueeze(0).to(self.device)
                features = item['features'].unsqueeze(0).to(self.device)
                
                embedding = self.model(image, features)
                embeddings.append(embedding.cpu().numpy())
                
                # Get label for coloring
                item_data = item['item_data']
                label = f"{item_data.get('det_label', 'unknown')}_{item_data.get('color', {}).get('label', 'unknown')}"
                labels.append(label)
        
        embeddings = np.vstack(embeddings)
        
        # t-SNE
        print("🔄 Computing t-SNE embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Color by category
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=20)
        
        plt.title('Fashion Item Embeddings (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Embeddings visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train custom fashion recommendation model')
    parser.add_argument('--custom_json', type=str, default='dataset/fashion_dataset_1000 (3).json',
                       help='Path to custom JSON dataset')
    parser.add_argument('--output_dir', type=str, default='models/custom_recommendation',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    print("Custom Fashion Recommendation Model Training")
    print("=" * 60)
    
    # Step 1: Convert custom dataset
    print("\nStep 1: Converting custom dataset...")
    adapter = CustomFashionDatasetAdapter(args.custom_json, "dataset/adapted")
    adapted_json_path = adapter.convert_to_expected_format()
    
    # Step 2: Create dataset
    print("\nStep 2: Creating dataset...")
    dataset = TripletDataset([adapted_json_path], "dataset/adapted/images")
    
    # Get feature dimension
    sample_item = dataset[0]
    feature_dim = sample_item['features'].shape[0]
    
    print(f"Dataset created with {len(dataset)} items")
    print(f"Feature dimension: {feature_dim}")
    
    # Step 3: Create model
    print("\nStep 3: Creating optimized model...")
    model = OptimizedFashionEmbeddingModel(
        feature_dim=feature_dim,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    trainer = CustomFashionTrainer(model, dataset)
    train_losses, val_losses = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience
    )
    
    # Step 5: Save model
    print("\nStep 5: Saving model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'custom_fashion_model.pth')
    
    # Save encoders
    with open(output_dir / 'encoders.pkl', 'wb') as f:
        pickle.dump({
            'color_encoder': dataset.color_encoder,
            'pattern_encoder': dataset.pattern_encoder,
            'season_encoder': dataset.season_encoder,
            'clothing_encoder': dataset.clothing_encoder
        }, f)
    
    # Save training info
    training_info = {
        'model_architecture': 'OptimizedFashionEmbeddingModel',
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_epochs': len(train_losses),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None,
        'training_time': time.time(),
        'dataset_size': len(dataset),
        'feature_dim': feature_dim
    }
    
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Model saved to: {output_dir}")
    
    # Step 6: Visualizations
    if args.visualize:
        print("\nStep 6: Creating visualizations...")
        
        # Training curves
        trainer.plot_training_curves(output_dir / 'training_curves.png')
        
        # Embeddings visualization
        trainer.visualize_embeddings(save_path=output_dir / 'embeddings_visualization.png')
    
    print("\nTraining completed successfully!")
    print(f"Model directory: {output_dir}")
    print(f"Use the model for recommendations with the fashion_recommender.py script")

if __name__ == "__main__":
    main()
