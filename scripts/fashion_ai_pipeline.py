"""
Complete Fashion AI Pipeline
Integrates clothing detection, color analysis, data extraction, and recommendations.
"""

import os
import json
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Import our custom modules
from train_df2_maskcrnn import build_model, DF2_CLASSES
from color_detection import ColorDetector, analyze_clothing_colors
from data_extraction import ClothingDataExtractor, ClothingItem
from recommendation_engine import FashionRecommendationEngine, Recommendation, OutfitSuggestion


class FashionAIPipeline:
    """Complete fashion AI pipeline for outfit analysis and recommendations."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the fashion AI pipeline.
        
        Args:
            model_path: Path to trained clothing detection model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = model_path
        
        # Initialize components
        self.detection_model = None
        self.color_detector = ColorDetector(n_colors=5)
        self.data_extractor = ClothingDataExtractor()
        self.recommendation_engine = FashionRecommendationEngine()
        
        # Load the detection model
        self._load_detection_model()
    
    def _load_detection_model(self):
        """Load the trained clothing detection model."""
        try:
            # Load checkpoint and infer number of classes from head shape
            state = torch.load(self.model_path, map_location="cpu")
            sd = state.get("model", state)
            cls_w = sd.get("roi_heads.box_predictor.cls_score.weight")
            if cls_w is None:
                raise RuntimeError("Unexpected checkpoint format: missing roi_heads.box_predictor.cls_score.weight")
            num_classes_ckpt = int(cls_w.shape[0])

            self.detection_model = build_model(num_classes=num_classes_ckpt, pretrained=False)
            self.detection_model.load_state_dict(sd)
            self.detection_model.to(self.device).eval()
            print(f"✅ Detection model loaded from {self.model_path} (num_classes={num_classes_ckpt})")
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            raise
    
    def process_image(self, image_path: str, output_dir: str = "./fashion_analysis") -> Dict[str, Any]:
        """
        Process a single image through the complete fashion AI pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save analysis results
            
        Returns:
            Complete analysis results
        """
        print(f"🔄 Processing image: {image_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Clothing Detection
        print("🔍 Step 1: Detecting clothing items...")
        detections = self._detect_clothing(image_path)
        print(f"   Found {len(detections)} clothing items")
        
        # Step 2: Color Analysis
        print("🎨 Step 2: Analyzing colors...")
        color_analysis = self._analyze_colors(image_path, detections, output_dir)
        
        # Step 3: Data Extraction
        print("📊 Step 3: Extracting clothing data...")
        clothing_items = self._extract_clothing_data(image_path, detections, color_analysis)
        
        # Step 4: Generate Recommendations
        print("💡 Step 4: Generating recommendations...")
        recommendations = self._generate_recommendations(clothing_items)
        
        # Step 5: Generate Outfit Suggestions
        print("👗 Step 5: Creating outfit suggestions...")
        outfit_suggestions = self._generate_outfit_suggestions(clothing_items)
        
        # Step 6: Create Summary
        print("📋 Step 6: Creating analysis summary...")
        summary = self._create_analysis_summary(clothing_items, recommendations, outfit_suggestions)
        
        # Save results
        results = {
            'timestamp': timestamp,
            'image_path': image_path,
            'detections': detections,
            'color_analysis': color_analysis,
            'clothing_items': clothing_items,
            'recommendations': recommendations,
            'outfit_suggestions': outfit_suggestions,
            'summary': summary
        }
        
        # Export results
        self._export_results(results, output_dir, timestamp)
        
        print(f"✅ Analysis complete! Results saved to {output_dir}")
        return results
    
    def _detect_clothing(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect clothing items in the image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB and tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from torchvision.transforms import functional as F
        pil_image = F.to_pil_image(image_rgb)
        tensor = F.to_tensor(pil_image).to(self.device)
        
        # Run detection
        with torch.no_grad():
            outputs = self.detection_model([tensor])[0]
        
        # Process detections
        detections = []
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                bbox = boxes[i].tolist()
                label_id = int(labels[i])
                class_name = DF2_CLASSES[label_id - 1] if 0 < label_id <= len(DF2_CLASSES) else "unknown"
                
                detection = {
                    'bbox': bbox,
                    'score': float(scores[i]),
                    'class_id': label_id,
                    'class_name': class_name
                }
                detections.append(detection)
        
        return detections
    
    def _analyze_colors(self, image_path: str, detections: List[Dict], 
                       output_dir: str) -> List[Dict[str, Any]]:
        """Analyze colors for detected clothing items."""
        try:
            color_analysis = analyze_clothing_colors(image_path, detections, output_dir)
            return color_analysis
        except Exception as e:
            print(f"⚠️ Color analysis failed: {e}")
            return []
    
    def _extract_clothing_data(self, image_path: str, detections: List[Dict], 
                              color_analysis: List[Dict]) -> List[ClothingItem]:
        """Extract comprehensive clothing data."""
        try:
            clothing_items = self.data_extractor.extract_clothing_data(image_path, detections)
            return clothing_items
        except Exception as e:
            print(f"⚠️ Data extraction failed: {e}")
            return []
    
    def _generate_recommendations(self, clothing_items: List[ClothingItem]) -> List[Recommendation]:
        """Generate fashion recommendations."""
        try:
            recommendations = self.recommendation_engine.generate_recommendations(clothing_items)
            return recommendations
        except Exception as e:
            print(f"⚠️ Recommendation generation failed: {e}")
            return []
    
    def _generate_outfit_suggestions(self, clothing_items: List[ClothingItem]) -> List[OutfitSuggestion]:
        """Generate outfit suggestions."""
        try:
            suggestions = self.recommendation_engine.suggest_complete_outfits(clothing_items)
            return suggestions
        except Exception as e:
            print(f"⚠️ Outfit suggestion generation failed: {e}")
            return []
    
    def _create_analysis_summary(self, clothing_items: List[ClothingItem], 
                                recommendations: List[Recommendation],
                                outfit_suggestions: List[OutfitSuggestion]) -> Dict[str, Any]:
        """Create a summary of the analysis."""
        summary = {
            'total_items_detected': len(clothing_items),
            'categories_found': list(set(item.category.value for item in clothing_items)),
            'dominant_colors': self._get_dominant_colors(clothing_items),
            'overall_style': self._get_overall_style(clothing_items),
            'outfit_completeness': self._assess_completeness(clothing_items),
            'recommendations_count': len(recommendations),
            'outfit_suggestions_count': len(outfit_suggestions),
            'top_recommendations': [rec.title for rec in recommendations[:3]],
            'top_outfit_suggestions': [sug.outfit_name for sug in outfit_suggestions[:3]]
        }
        return summary
    
    def _get_dominant_colors(self, clothing_items: List[ClothingItem]) -> List[str]:
        """Get dominant colors across all items."""
        all_colors = []
        for item in clothing_items:
            all_colors.extend(item.color_names)
        
        # Count color frequency
        color_counts = {}
        for color in all_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Return top 3 colors
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        return [color for color, _ in sorted_colors[:3]]
    
    def _get_overall_style(self, clothing_items: List[ClothingItem]) -> str:
        """Get overall style of the outfit."""
        if not clothing_items:
            return "unknown"
        
        style_counts = {}
        for item in clothing_items:
            style = item.style_type.value
            style_counts[style] = style_counts.get(style, 0) + 1
        
        return max(style_counts.items(), key=lambda x: x[1])[0]
    
    def _assess_completeness(self, clothing_items: List[ClothingItem]) -> str:
        """Assess outfit completeness."""
        if not clothing_items:
            return "incomplete"
        
        categories = {item.category.value for item in clothing_items}
        
        has_top = any(cat in categories for cat in 
                     ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_dress', 'long_sleeved_dress'])
        has_bottom = any(cat in categories for cat in 
                        ['trousers', 'shorts', 'skirt'])
        has_outwear = any(cat in categories for cat in 
                         ['short_sleeved_outwear', 'long_sleeved_outwear'])
        
        if has_top and has_bottom and has_outwear:
            return "complete"
        elif has_top and has_bottom:
            return "basic"
        elif has_top or has_bottom:
            return "partial"
        else:
            return "incomplete"
    
    def _export_results(self, results: Dict[str, Any], output_dir: str, timestamp: str):
        """Export analysis results to files."""
        # Export main results
        results_path = os.path.join(output_dir, f"fashion_analysis_{timestamp}.json")
        
        # Convert non-serializable objects to dictionaries
        exportable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        # Export individual components
        if results['clothing_items']:
            clothing_data_path = os.path.join(output_dir, f"clothing_data_{timestamp}.json")
            self.data_extractor.export_clothing_data(results['clothing_items'], clothing_data_path)
        
        if results['recommendations']:
            recommendations_path = os.path.join(output_dir, f"recommendations_{timestamp}.json")
            self.recommendation_engine.export_recommendations(results['recommendations'], recommendations_path)
        
        if results['outfit_suggestions']:
            outfits_path = os.path.join(output_dir, f"outfit_suggestions_{timestamp}.json")
            self.recommendation_engine.export_outfit_suggestions(results['outfit_suggestions'], outfits_path)
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        import numpy as _np
        from types import MappingProxyType as _MappingProxyType
        import enum as _enum
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, _MappingProxyType):
            return self._make_serializable(dict(obj))
        elif isinstance(obj, _enum.Enum):
            return self._make_serializable(obj.value)
        elif isinstance(obj, (_np.integer,)):
            return int(obj)
        elif isinstance(obj, (_np.floating,)):
            return float(obj)
        elif isinstance(obj, (_np.bool_,)):
            return bool(obj)
        elif isinstance(obj, _np.ndarray):
            return self._make_serializable(obj.tolist())
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def create_visualization(self, image_path: str, results: Dict[str, Any], 
                           output_path: str = None) -> str:
        """Create a visualization of the analysis results."""
        if output_path is None:
            output_path = os.path.join(os.path.dirname(image_path), "fashion_analysis_visualization.jpg")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Draw bounding boxes and labels
        for detection in results['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            class_name = detection['class_name']
            score = detection['score']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, image)
        return output_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Fashion AI Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", default="./fashion_analysis", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FashionAIPipeline(args.model, args.device)
    
    # Process image
    results = pipeline.process_image(args.image, args.output)
    
    # Create visualization
    vis_path = pipeline.create_visualization(args.image, results)
    print(f"📊 Visualization saved to: {vis_path}")
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*50)
    print("FASHION ANALYSIS SUMMARY")
    print("="*50)
    print(f"Items detected: {summary['total_items_detected']}")
    print(f"Categories: {', '.join(summary['categories_found'])}")
    print(f"Dominant colors: {', '.join(summary['dominant_colors'])}")
    print(f"Overall style: {summary['overall_style']}")
    print(f"Completeness: {summary['outfit_completeness']}")
    print(f"Recommendations: {summary['recommendations_count']}")
    print(f"Outfit suggestions: {summary['outfit_suggestions_count']}")
    print("="*50)


if __name__ == "__main__":
    main()

