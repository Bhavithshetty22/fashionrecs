# Fashion Datasets Guide for Triplet Loss Recommendation System

This guide helps you find and download fashion datasets compatible with your triplet loss recommendation system.

## 🎯 **Top Recommended Datasets**

### **1. Fashion Product Images Dataset (Kaggle) - BEST CHOICE**
- **URL**: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- **Size**: ~44,000 images
- **Features**: Color, pattern, season, gender, category, subcategory
- **Format**: Images + CSV metadata
- **Why it's perfect**: Has all the attributes you need (color, pattern, season)

#### **How to Download:**
```bash
# Install Kaggle API
pip install kaggle

# Get API key from https://www.kaggle.com/account
# Place kaggle.json in ~/.kaggle/ (or C:\Users\YourName\.kaggle\ on Windows)

# Download dataset
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset

# Extract
unzip fashion-product-images-dataset.zip
```

### **2. DeepFashion2 Dataset (Kaggle) - You're Already Using This!**
- **URL**: https://www.kaggle.com/datasets/switchablenorms/deepfashion2
- **Size**: ~800,000 images
- **Features**: Detailed annotations, categories, keypoints
- **Format**: Images + JSON annotations
- **Note**: This is what you're currently using!

### **3. Fashion MNIST (Kaggle) - Good for Basic Classification**
- **URL**: https://www.kaggle.com/datasets/zalando-research/fashionmnist
- **Size**: 70,000 grayscale images
- **Features**: 10 clothing categories
- **Format**: Images + labels
- **Good for**: Basic clothing classification

## 🌐 **Alternative Sources**

### **Hugging Face Datasets**
1. **Fashion Product Images Small**
   - URL: https://huggingface.co/datasets/Transformersx/fashion-product-images-small
   - Features: Color, season, gender, category
   - Easy Python integration

2. **Fashion Dataset (nreimers)**
   - URL: https://huggingface.co/datasets/nreimers/fashion-dataset
   - Features: Gender, category, color, season, year, usage

### **Roboflow Datasets**
1. **Fashion Seasons Classification**
   - URL: https://universe.roboflow.com/capstone-zn0iu/fashion-seasons/dataset/1
   - Size: 2,559 images
   - Features: Season labels (Spring, Summer, Fall, Winter)

### **Academic Datasets**
1. **DeepFashion Dataset**
   - URL: https://github.com/switchablenorms/DeepFashion
   - Size: ~800,000 images
   - Features: Categories, attributes, keypoints

## 🚀 **Quick Start with Our Downloader**

### **1. Create Sample Dataset (For Testing)**
```bash
python scripts/download_fashion_datasets.py --dataset sample --sample_size 1000
```

### **2. Download from Hugging Face**
```bash
python scripts/download_fashion_datasets.py --dataset huggingface_fashion
```

### **3. Download from Kaggle (After API Setup)**
```bash
python scripts/download_fashion_datasets.py --dataset kaggle_fashion
```

## 📊 **Dataset Format Compatibility**

Your triplet loss system expects this JSON format:
```json
{
  "items": [
    {
      "crop_file": "image.jpg",
      "det_label": "shirt",
      "det_score": 0.9,
      "box": [0, 0, 224, 224],
      "color": {"label": "Blue", "score": 0.8},
      "pattern": {"label": "Solid", "score": 0.8},
      "season": {"label": "Summer", "score": 0.8}
    }
  ]
}
```

## 🔧 **Integration Steps**

### **1. Download Dataset**
```bash
# Using our script
python scripts/download_fashion_datasets.py --dataset sample --sample_size 5000
```

### **2. Train with New Data**
```bash
# Train with downloaded data
python scripts/simple_triplet_training.py \
    --json_files fashion_datasets/metadata/*.json \
    --image_dir fashion_datasets/images \
    --epochs 50 \
    --batch_size 16
```

### **3. Use for Recommendations**
```python
from scripts.fashion_recommender import FashionRecommender

# Load trained model
recommender = FashionRecommender("models/simple_triplet/model.pth", 
                                "models/simple_triplet/encoders.pkl")

# Load new dataset
recommender.load_item_database(["fashion_datasets/metadata/*.json"], 
                              "fashion_datasets/images")

# Get recommendations
similar_items = recommender.find_similar_items(query_item, query_image, top_k=5)
```

## 📈 **Dataset Size Recommendations**

### **For Testing (Small)**
- **Size**: 100-1,000 items
- **Use**: Sample dataset or small subset
- **Training time**: 5-10 minutes

### **For Development (Medium)**
- **Size**: 1,000-10,000 items
- **Use**: Fashion Product Images (subset)
- **Training time**: 30-60 minutes

### **For Production (Large)**
- **Size**: 10,000+ items
- **Use**: Full Fashion Product Images or DeepFashion2
- **Training time**: 2-6 hours

## 🎯 **Specific Dataset Recommendations by Use Case**

### **For Season Classification**
- **Best**: Roboflow Fashion Seasons (2,559 images)
- **Alternative**: Fashion Product Images (has season column)

### **For Color/Pattern Analysis**
- **Best**: Fashion Product Images (has color and pattern columns)
- **Alternative**: DeepFashion2 (detailed annotations)

### **For General Fashion Recommendations**
- **Best**: Fashion Product Images (44K images, good variety)
- **Alternative**: DeepFashion2 (800K images, more complex)

### **For Quick Testing**
- **Best**: Sample dataset (generated by our script)
- **Alternative**: Fashion MNIST (70K images, simple)

## 💡 **Pro Tips**

1. **Start Small**: Begin with sample dataset for testing
2. **Gradual Scale**: Increase dataset size as you improve the model
3. **Quality over Quantity**: Better to have 10K high-quality items than 100K poor ones
4. **Diverse Sources**: Combine multiple datasets for better generalization
5. **Data Augmentation**: Use image augmentation to increase effective dataset size

## 🔍 **Finding More Datasets**

### **Search Terms to Use**
- "fashion dataset" + "color" + "season"
- "clothing images" + "metadata" + "attributes"
- "fashion product" + "classification" + "kaggle"
- "garment dataset" + "style" + "pattern"

### **Platforms to Check**
- **Kaggle**: https://www.kaggle.com/datasets
- **Hugging Face**: https://huggingface.co/datasets
- **Roboflow**: https://universe.roboflow.com
- **Papers with Code**: https://paperswithcode.com/datasets
- **Google Dataset Search**: https://datasetsearch.research.google.com

## 🚨 **Important Notes**

1. **Licensing**: Always check dataset licenses before commercial use
2. **Attribution**: Give proper credit to dataset creators
3. **Privacy**: Ensure datasets don't contain personal information
4. **Quality**: Verify image quality and annotation accuracy
5. **Updates**: Check if datasets are actively maintained

## 📞 **Need Help?**

If you need help with specific datasets or encounter issues:

1. Check the dataset documentation
2. Look for example notebooks on Kaggle
3. Search for similar projects on GitHub
4. Ask questions on relevant forums (Kaggle, Stack Overflow)

Happy dataset hunting! 🎽✨



