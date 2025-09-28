import gradio as gr
import torch
import cv2
import numpy as np
import json
import os
import tempfile
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModelForImageClassification

# Import the inference functions
try:
    from .infer_df2 import main as run_inference
    from .classify_crops_colors import load_model as load_color_model, predict_color
    from .classify_crops_seasons import load_model as load_season_model, predict_season
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from infer_df2 import main as run_inference
    from classify_crops_colors import load_model as load_color_model, predict_color
    from classify_crops_seasons import load_model as load_season_model, predict_season

# Global variables for models
color_model = None
color_processor = None
color_device = None
season_model = None
season_processor = None
season_device = None
pattern_model = None
pattern_processor = None
pattern_device = None

def load_all_models():
    """Load all fashion analysis models"""
    global color_model, color_processor, color_device
    global season_model, season_processor, season_device
    global pattern_model, pattern_processor, pattern_device
    
    try:
        # Load color model
        color_model, color_processor, color_device = load_color_model("prithivMLmods/Fashion-Product-baseColour")
        print("✓ Color model loaded")
        
        # Load season model
        season_model, season_processor, season_device = load_season_model("prithivMLmods/Fashion-Product-Season")
        print("✓ Season model loaded")
        
        # Load pattern model
        pattern_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pattern_processor = AutoImageProcessor.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")
        pattern_model = AutoModelForImageClassification.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")
        pattern_model.to(pattern_device).eval()
        print("✓ Pattern model loaded")
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def analyze_fashion_image(image, detection_threshold=0.35, run_color=True, run_pattern=True, run_season=True):
    """Analyze a fashion image for detection, color, pattern, and season"""
    if image is None:
        return None, "Please upload an image"
    
    try:
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            temp_image_path = tmp_file.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run DeepFashion2 inference
            import sys
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    'infer_df2.py',
                    '--ckpt', 'checkpoints_df2/epoch_2_step_16000.pth',  # Use your best checkpoint
                    '--image', temp_image_path,
                    '--out', os.path.join(temp_dir, 'output.jpg'),
                    '--score_thr', str(detection_threshold),
                    '--save_crops_dir', temp_dir,
                    '--run_color_after' if run_color else '',
                    '--run_pattern_local' if run_pattern else '',
                    '--run_season_local' if run_season else ''
                ]
                # Remove empty strings
                sys.argv = [arg for arg in sys.argv if arg]
                
                # Run inference
                run_inference()
                
                # Load results
                results = []
                crop_files = list(Path(temp_dir).glob("det_*_crop.jpg"))
                
                for crop_file in sorted(crop_files):
                    item = {
                        "crop_file": crop_file.name,
                        "det_label": "Unknown",
                        "det_score": 0.0,
                        "box": [0, 0, 100, 100],
                        "color": {"label": None, "score": None},
                        "pattern": {"label": None, "score": None},
                        "season": {"label": None, "score": None}
                    }
                    
                    # Run individual classifications on crops
                    if run_color and color_model is not None:
                        try:
                            color_pred = predict_color(crop_file, color_model, color_processor, color_device)
                            item["color"] = {
                                "label": color_pred["label"],
                                "score": color_pred["score"]
                            }
                        except Exception as e:
                            print(f"Color prediction failed for {crop_file}: {e}")
                    
                    if run_pattern and pattern_model is not None:
                        try:
                            im = Image.open(crop_file).convert("RGB")
                            inputs = pattern_processor(images=im, return_tensors="pt").to(pattern_device)
                            with torch.no_grad():
                                logits = pattern_model(**inputs).logits
                                probs = torch.softmax(logits, dim=-1)
                                pred_idx = int(torch.argmax(probs, dim=-1).item())
                                label = pattern_model.config.id2label[pred_idx]
                                score = float(probs[0, pred_idx].item())
                            item["pattern"] = {"label": label, "score": round(score, 4)}
                        except Exception as e:
                            print(f"Pattern prediction failed for {crop_file}: {e}")
                    
                    if run_season and season_model is not None:
                        try:
                            season_pred = predict_season(crop_file, season_model, season_processor, season_device)
                            item["season"] = {
                                "label": season_pred["label"],
                                "score": season_pred["score"]
                            }
                        except Exception as e:
                            print(f"Season prediction failed for {crop_file}: {e}")
                    
                    results.append(item)
                
                # Create summary text
                summary = f"Found {len(results)} fashion items:\n\n"
                for i, item in enumerate(results):
                    summary += f"Item {i+1}:\n"
                    summary += f"  Detection: {item['det_label']} (score: {item['det_score']:.3f})\n"
                    if item['color']['label']:
                        summary += f"  Color: {item['color']['label']} (score: {item['color']['score']:.3f})\n"
                    if item['pattern']['label']:
                        summary += f"  Pattern: {item['pattern']['label']} (score: {item['pattern']['score']:.3f})\n"
                    if item['season']['label']:
                        summary += f"  Season: {item['season']['label']} (score: {item['season']['score']:.3f})\n"
                    summary += "\n"
                
                return results, summary
                
            finally:
                sys.argv = old_argv
                os.unlink(temp_image_path)
                
    except Exception as e:
        return None, f"Error analyzing image: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    # Load models
    if not load_all_models():
        print("Warning: Some models failed to load. The app may not work correctly.")
    
    with gr.Blocks(title="Fashion Analysis Suite", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎽 Fashion Analysis Suite")
        gr.Markdown("Upload a fashion image to analyze clothing items for detection, color, pattern, and seasonality.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Fashion Image",
                    type="numpy",
                    height=400
                )
                
                with gr.Row():
                    detection_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.35,
                        step=0.05,
                        label="Detection Threshold"
                    )
                
                with gr.Row():
                    run_color = gr.Checkbox(label="Color Analysis", value=True)
                    run_pattern = gr.Checkbox(label="Pattern Analysis", value=True)
                    run_season = gr.Checkbox(label="Season Analysis", value=True)
                
                analyze_btn = gr.Button("🔍 Analyze Fashion", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                results_output = gr.JSON(
                    label="Analysis Results",
                    visible=True
                )
                
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Example images
        gr.Markdown("## 📸 Example Images")
        gr.Examples(
            examples=[
                ["fashion_out/det_00_crop.jpg"],
                ["fashion_out/det_01_crop.jpg"],
            ],
            inputs=image_input,
            label="Click to load example"
        )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_fashion_image,
            inputs=[image_input, detection_threshold, run_color, run_pattern, run_season],
            outputs=[results_output, summary_output]
        )
        
        # Instructions
        gr.Markdown("""
        ## 📋 How to Use
        
        1. **Upload an Image**: Choose a fashion image containing clothing items
        2. **Adjust Settings**: 
           - Detection Threshold: Lower values detect more items (may include false positives)
           - Check/uncheck analysis types as needed
        3. **Click Analyze**: The app will detect clothing items and analyze their properties
        4. **View Results**: See detailed JSON results and a human-readable summary
        
        ## 🔧 Features
        
        - **Object Detection**: Identifies and localizes clothing items using DeepFashion2
        - **Color Analysis**: Determines primary colors using Fashion-Product-baseColour model
        - **Pattern Recognition**: Classifies fabric patterns using Clothes_Pattern_Classification_v2
        - **Seasonality Detection**: Predicts suitable seasons using Fashion-Product-Season model
        
        ## 📊 Output Format
        
        Each detected item includes:
        - Detection label and confidence score
        - Bounding box coordinates
        - Color prediction with confidence
        - Pattern classification with confidence  
        - Season prediction with confidence
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
