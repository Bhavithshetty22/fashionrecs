# Fashion Triplet Loss Recommendation System

This system implements a triplet loss-based recommendation system for fashion items using the data from your fashion analysis JSON files.

## 🎯 Overview

The system learns embeddings for fashion items that capture their visual and categorical features, enabling:
- **Similarity-based recommendations**: Find items similar to a query item
- **Attribute-based filtering**: Recommend by color, pattern, season, or clothing type
- **Seasonal recommendations**: Get seasonal fashion suggestions
- **Color coordination**: Find items that match a base item's color

## 🏗️ Architecture

### 1. **Fashion Item Dataset** (`fashion_triplet_model.py`)
- Loads fashion analysis JSON data
- Creates triplets (anchor, positive, negative) for training
- Encodes categorical features (color, pattern, season, clothing type)
- Handles image preprocessing and augmentation

### 2. **Embedding Model** (`fashion_triplet_model.py`)
- **Image Encoder**: CNN-based feature extraction from images
- **Feature Encoder**: MLP for categorical features
- **Fusion Layer**: Combines image and categorical features
- **Output**: Normalized 128-dimensional embeddings

### 3. **Triplet Loss** (`fashion_triplet_model.py`)
- Implements triplet loss with configurable margin
- Ensures similar items are closer in embedding space
- Ensures dissimilar items are farther apart

### 4. **Recommendation System** (`fashion_recommender.py`)
- Uses trained embeddings for similarity search
- Provides various recommendation strategies
- Supports clustering and visualization

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_triplet.txt
```

### 2. Train the Model
```bash
# Basic training with existing data
python scripts/run_fashion_triplet_training.py

# With additional data collection
python scripts/run_fashion_triplet_training.py --collect_data

# Custom parameters
python scripts/run_fashion_triplet_training.py \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0005
```

### 3. Use the Recommendation System
```bash
# Run the demo
python fashion_demo.py

# Or use the recommender directly
python scripts/fashion_recommender.py \
    --model_path models/fashion_triplet/fashion_embedding_model.pth \
    --encoders_path models/fashion_triplet/encoders.pkl \
    --json_files fashion_out/fashion_analysis_*.json \
    --image_dir fashion_out
```

## 📊 Data Format

The system expects JSON files with this structure:
```json
{
  "items": [
    {
      "crop_file": "det_00_crop.jpg",
      "det_label": "trousers",
      "det_score": 0.997,
      "box": [155, 247, 301, 570],
      "color": {"label": "Blue", "score": 0.8481},
      "pattern": {"label": "Denim", "score": 0.9992},
      "season": {"label": "Spring", "score": 0.8}
    }
  ]
}
```

## 🔧 Training Process

### 1. **Data Preparation**
- Loads fashion analysis JSON files
- Creates item groups based on attributes
- Generates triplets for training

### 2. **Model Training**
- Trains CNN + MLP architecture
- Uses triplet loss with margin
- Includes learning rate scheduling
- Saves model checkpoints

### 3. **Evaluation**
- Computes training/validation loss
- Creates embedding visualizations
- Tests recommendation quality

## 🎨 Recommendation Types

### 1. **Similarity Search**
```python
similar_items = recommender.find_similar_items(
    query_item, query_image_path, top_k=5
)
```

### 2. **Attribute-based Filtering**
```python
items = recommender.recommend_by_attributes(
    color="Blue", season="Summer", top_k=10
)
```

### 3. **Seasonal Recommendations**
```python
winter_items = recommender.get_seasonal_recommendations("Winter")
```

### 4. **Color Coordination**
```python
coordinated = recommender.get_color_coordinated_items(base_item)
```

## 📈 Model Performance

### Training Metrics
- **Loss**: Triplet loss with configurable margin
- **Learning Rate**: Adaptive scheduling
- **Convergence**: Typically 50-100 epochs

### Recommendation Quality
- **Similarity**: Cosine similarity in embedding space
- **Clustering**: K-means clustering of embeddings
- **Visualization**: t-SNE plots for embedding analysis

## 🛠️ Customization

### Model Architecture
```python
# Modify embedding dimension
model = FashionEmbeddingModel(feature_dim, embedding_dim=256)

# Adjust CNN architecture
# Modify image_encoder in FashionEmbeddingModel
```

### Training Parameters
```python
# Adjust triplet loss margin
criterion = TripletLoss(margin=2.0)

# Modify learning rate schedule
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=10, factor=0.3
)
```

### Data Augmentation
```python
# Add more transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

## 📁 File Structure

```
scripts/
├── fashion_triplet_model.py      # Core model and dataset classes
├── train_triplet_model.py        # Training script
├── fashion_recommender.py        # Recommendation system
├── collect_fashion_dataset.py    # Data collection from web
└── run_fashion_triplet_training.py  # Complete pipeline

models/fashion_triplet/
├── fashion_embedding_model.pth   # Trained model weights
├── encoders.pkl                  # Label encoders
├── training_curves.png           # Training progress
└── embeddings_visualization.png  # t-SNE visualization

recommendations/
├── recommendations.png           # Recommendation visualizations
└── clusters/                     # Clustering results
```

## 🔍 Advanced Usage

### 1. **Custom Triplet Sampling**
```python
# Modify triplet sampling strategy in FashionItemDataset
def get_triplet(self, idx):
    # Custom logic for selecting positive/negative samples
    pass
```

### 2. **Multi-Modal Features**
```python
# Add more features to the model
features = torch.cat([
    color_onehot,
    pattern_onehot,
    season_onehot,
    clothing_onehot,
    price_onehot,      # Add price range
    brand_onehot       # Add brand information
])
```

### 3. **Ensemble Recommendations**
```python
# Combine multiple recommendation strategies
def ensemble_recommend(self, query_item, weights=[0.4, 0.3, 0.3]):
    similarity_rec = self.find_similar_items(query_item, top_k=10)
    attribute_rec = self.recommend_by_attributes(**query_attributes)
    seasonal_rec = self.get_seasonal_recommendations(query_season)
    
    # Combine with weights
    return weighted_combination(similarity_rec, attribute_rec, seasonal_rec)
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Use CPU: `--device cpu`

2. **Poor Recommendations**
   - Increase training epochs: `--epochs 100`
   - Adjust triplet margin: `--margin 2.0`
   - Collect more diverse data

3. **Data Loading Errors**
   - Check image paths in JSON files
   - Ensure all images exist
   - Verify JSON format

### Performance Tips

1. **Faster Training**
   - Use GPU: `--device cuda`
   - Increase batch size (if memory allows)
   - Use mixed precision training

2. **Better Recommendations**
   - Collect more diverse data
   - Fine-tune hyperparameters
   - Use ensemble methods

## 📚 References

- [Triplet Loss Paper](https://arxiv.org/abs/1503.03832)
- [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)
- [PyTorch Triplet Loss Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 🤝 Contributing

To improve the recommendation system:

1. **Add new features** to the embedding model
2. **Implement new sampling strategies** for triplets
3. **Create new recommendation algorithms**
4. **Add evaluation metrics** for recommendation quality
5. **Improve data collection** from online sources

## 📄 License

This project extends the DeepFashion2 work and follows the same licensing terms.



