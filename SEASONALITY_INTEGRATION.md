# Fashion Seasonality Detection Integration

This document explains how to use the newly integrated seasonality detection feature in the DeepFashion2 project.

## 🆕 New Features

### 1. Seasonality Detection Script
- **File**: `scripts/classify_crops_seasons.py`
- **Model**: `prithivMLmods/Fashion-Product-Season`
- **Classes**: Fall, Spring, Summer, Winter

### 2. Integrated Inference Pipeline
- **File**: `scripts/infer_df2.py` (updated)
- **New flag**: `--run_season_local`
- **Output**: JSON includes season predictions

### 3. Gradio Web Applications
- **File**: `scripts/season_app.py` - Standalone season detection
- **File**: `scripts/fashion_analysis_app.py` - Complete fashion analysis suite

### 4. Easy-to-Use Runner Script
- **File**: `scripts/run_fashion_analysis.py`
- **Features**: Command-line interface for all analysis types

## 🚀 Quick Start

### Option 1: Standalone Season Detection
```bash
# Launch simple season detection app
python scripts/season_app.py
```

### Option 2: Complete Fashion Analysis
```bash
# Run all analyses (detection + color + pattern + season)
python scripts/run_fashion_analysis.py --image your_image.jpg --all

# Run only season detection
python scripts/run_fashion_analysis.py --image your_image.jpg --season-only

# Launch web interface
python scripts/run_fashion_analysis.py --gradio
```

### Option 3: Direct Inference Script
```bash
# Run with season detection
python scripts/infer_df2.py \
    --ckpt checkpoints_df2/epoch_2_step_16000.pth \
    --image your_image.jpg \
    --save_crops_dir fashion_out \
    --run_color_after \
    --run_pattern_local \
    --run_season_local
```

## 📊 Output Format

The analysis now includes season predictions in the JSON output:

```json
{
  "items": [
    {
      "crop_file": "det_00_crop.jpg",
      "det_label": "trousers",
      "det_score": 0.997,
      "box": [155, 247, 301, 570],
      "color": {
        "label": "Blue",
        "score": 0.8481
      },
      "pattern": {
        "label": "Denim",
        "score": 0.9992
      },
      "season": {
        "label": "Fall",
        "score": 0.7234
      }
    }
  ]
}
```

## 🔧 Model Details

### Season Classification Model
- **Model**: `prithivMLmods/Fashion-Product-Season`
- **Architecture**: SigLIP (Sigmoid Loss for Language-Image Pre-training)
- **Classes**: 4 seasons (Fall, Spring, Summer, Winter)
- **Input**: RGB images (any size, auto-resized)
- **Output**: Probability scores for each season

### Integration Points
1. **Detection**: DeepFashion2 detects clothing items
2. **Cropping**: Individual items are cropped from the image
3. **Classification**: Each crop is analyzed for:
   - Color (46 classes)
   - Pattern (various fabric patterns)
   - Season (4 seasons)

## 🎯 Use Cases

### Fashion Retail
- **Seasonal Inventory**: Automatically categorize clothing by season
- **Recommendation Systems**: Suggest seasonal items to customers
- **Trend Analysis**: Track seasonal fashion trends

### Personal Styling
- **Wardrobe Organization**: Sort clothes by season
- **Outfit Planning**: Choose appropriate seasonal clothing
- **Shopping Assistance**: Find season-appropriate items

### Fashion Research
- **Trend Analysis**: Study seasonal fashion patterns
- **Design Inspiration**: Understand seasonal color/pattern preferences
- **Market Research**: Analyze seasonal product performance

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install transformers torch torchvision pillow opencv-python gradio
```

### Hardware
- **GPU**: Recommended for faster processing
- **RAM**: 4GB+ for model loading
- **Storage**: ~2GB for model downloads

### Model Download
Models are automatically downloaded on first use:
- `prithivMLmods/Fashion-Product-Season` (~500MB)
- `prithivMLmods/Fashion-Product-baseColour` (~500MB)
- `IrshadG/Clothes_Pattern_Classification_v2` (~1GB)

## 📝 Examples

### Example 1: Basic Season Detection
```python
from scripts.classify_crops_seasons import load_model, predict_season
from pathlib import Path

# Load model
model, processor, device = load_model("prithivMLmods/Fashion-Product-Season")

# Predict season for an image
result = predict_season(Path("fashion_out/det_00_crop.jpg"), model, processor, device)
print(f"Season: {result['label']} (confidence: {result['score']:.3f})")
```

### Example 2: Batch Processing
```python
import glob
from scripts.classify_crops_seasons import load_model, predict_season

model, processor, device = load_model("prithivMLmods/Fashion-Product-Season")

for image_path in glob.glob("fashion_out/det_*_crop.jpg"):
    result = predict_season(Path(image_path), model, processor, device)
    print(f"{image_path}: {result['label']} ({result['score']:.3f})")
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running with `--verbose` flag

2. **CUDA Out of Memory**
   - Use `--cpu` flag to force CPU processing
   - Reduce batch size in processing

3. **Import Errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Batch Processing**: Process multiple images together when possible
3. **Model Caching**: Models are cached after first download

## 📈 Future Enhancements

- [ ] Multi-season classification (items suitable for multiple seasons)
- [ ] Weather-specific recommendations
- [ ] Regional season variations
- [ ] Historical trend analysis
- [ ] Real-time video analysis

## 🤝 Contributing

To add new seasonality features:
1. Modify `scripts/classify_crops_seasons.py` for core functionality
2. Update `scripts/infer_df2.py` for pipeline integration
3. Extend `scripts/fashion_analysis_app.py` for UI features
4. Add tests and documentation

## 📚 References

- [DeepFashion2 Paper](https://arxiv.org/abs/1901.07973)
- [SigLIP Model](https://arxiv.org/abs/2203.15353)
- [Fashion-Product-Season Model](https://huggingface.co/prithivMLmods/Fashion-Product-Season)
