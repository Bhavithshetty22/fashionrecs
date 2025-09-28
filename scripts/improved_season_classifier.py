"""
Improved Season Classifier
=========================

This module provides an improved season classification that combines:
1. ML model predictions
2. Rule-based logic based on clothing types
3. Confidence thresholds and fallback mechanisms
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
from typing import Dict, List, Tuple, Optional

# Season classification model
SEASON_MODEL_NAME = "prithivMLmods/Fashion-Product-Season"

# Clothing type to season mapping rules
CLOTHING_SEASON_RULES = {
    # Winter clothing (high confidence)
    "long_sleeved_outwear": "Winter",
    "coat": "Winter", 
    "jacket": "Winter",
    "sweater": "Winter",
    "hoodie": "Winter",
    "cardigan": "Winter",
    "blazer": "Winter",
    "trench_coat": "Winter",
    "parka": "Winter",
    "down_jacket": "Winter",
    "wool_coat": "Winter",
    
    # Summer clothing (high confidence)
    "short_sleeved_shirt": "Summer",
    "tank_top": "Summer",
    "sleeveless": "Summer",
    "shorts": "Summer",
    "skirt": "Summer",
    "dress": "Summer",
    "swimwear": "Summer",
    "bikini": "Summer",
    "swimsuit": "Summer",
    "sundress": "Summer",
    "t_shirt": "Summer",
    "polo_shirt": "Summer",
    
    # Spring/Fall clothing (medium confidence)
    "long_sleeved_shirt": "Spring",
    "jeans": "Spring",
    "trousers": "Spring",
    "pants": "Spring",
    "chinos": "Spring",
    "khakis": "Spring",
    "denim": "Spring",
    
    # Ambiguous clothing (use ML model)
    "shirt": None,  # Let ML model decide
    "top": None,
    "bottom": None,
    "accessory": None,
}

# Season confidence thresholds
SEASON_CONFIDENCE_THRESHOLDS = {
    "high": 0.8,    # High confidence threshold
    "medium": 0.6,  # Medium confidence threshold
    "low": 0.4      # Low confidence threshold
}

class ImprovedSeasonClassifier:
    def __init__(self, model_name: str = SEASON_MODEL_NAME):
        """Initialize the improved season classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.id2label = {0: "Fall", 1: "Spring", 2: "Summer", 3: "Winter"}
        
    def predict_season_ml(self, image: Image.Image) -> Dict[str, float]:
        """Get ML model predictions for all seasons"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
        
        return {self.id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    
    def get_rule_based_season(self, clothing_type: str) -> Optional[str]:
        """Get season based on clothing type rules"""
        # Normalize clothing type for matching
        clothing_type_lower = clothing_type.lower()
        
        # Check exact matches first
        if clothing_type_lower in CLOTHING_SEASON_RULES:
            return CLOTHING_SEASON_RULES[clothing_type_lower]
        
        # Check partial matches (both directions)
        for rule_type, season in CLOTHING_SEASON_RULES.items():
            if rule_type in clothing_type_lower or clothing_type_lower in rule_type:
                return season
        
        # Check for key terms in clothing type
        winter_terms = ["outwear", "coat", "jacket", "sweater", "hoodie", "long_sleeved"]
        summer_terms = ["short_sleeved", "tank", "sleeveless", "shorts", "dress", "swim"]
        spring_terms = ["shirt", "pants", "trousers", "jeans"]
        
        if any(term in clothing_type_lower for term in winter_terms):
            return "Winter"
        elif any(term in clothing_type_lower for term in summer_terms):
            return "Summer"
        elif any(term in clothing_type_lower for term in spring_terms):
            return "Spring"
                
        return None
    
    def classify_season(self, image: Image.Image, clothing_type: str, 
                       ml_confidence_threshold: float = 0.6) -> Dict:
        """
        Improved season classification combining ML and rule-based approaches
        
        Args:
            image: PIL Image of the clothing item
            clothing_type: Detected clothing type (e.g., "long_sleeved_outwear")
            ml_confidence_threshold: Minimum confidence for ML predictions
            
        Returns:
            Dictionary with season prediction and confidence
        """
        # Get ML model predictions
        ml_predictions = self.predict_season_ml(image)
        ml_best_season = max(ml_predictions, key=ml_predictions.get)
        ml_confidence = ml_predictions[ml_best_season]
        
        # Get rule-based prediction
        rule_season = self.get_rule_based_season(clothing_type)
        
        # Decision logic
        if rule_season is not None:
            # Rule-based prediction available
            if ml_confidence >= ml_confidence_threshold:
                # High ML confidence - use ML prediction
                if ml_best_season == rule_season:
                    # Both agree - high confidence
                    final_season = ml_best_season
                    final_confidence = min(ml_confidence + 0.1, 1.0)  # Boost confidence
                    method = "ml_rule_agreement"
                else:
                    # Disagreement - prefer rule-based for clear clothing types
                    final_season = rule_season
                    final_confidence = 0.8  # High confidence for rule-based
                    method = "rule_based_override"
            else:
                # Low ML confidence - use rule-based
                final_season = rule_season
                final_confidence = 0.8
                method = "rule_based"
        else:
            # No rule available - use ML prediction
            final_season = ml_best_season
            final_confidence = ml_confidence
            method = "ml_only"
        
        return {
            "label": final_season,
            "score": round(final_confidence, 4),
            "method": method,
            "ml_predictions": ml_predictions,
            "rule_prediction": rule_season,
            "clothing_type": clothing_type
        }

def test_improved_classifier():
    """Test the improved classifier on the problematic image"""
    from pathlib import Path
    
    classifier = ImprovedSeasonClassifier()
    
    # Test on the crop image
    crop_path = Path("fashion_out/det_00_crop.jpg")
    if crop_path.exists():
        image = Image.open(crop_path).convert("RGB")
        clothing_type = "long_sleeved_outwear"  # From the detection results
        
        result = classifier.classify_season(image, clothing_type)
        
        print("Improved Season Classification Results:")
        print("=" * 50)
        print(f"Clothing Type: {clothing_type}")
        print(f"Final Prediction: {result['label']} (confidence: {result['score']:.3f})")
        print(f"Method: {result['method']}")
        print(f"Rule-based Prediction: {result['rule_prediction']}")
        print("\nML Model Predictions:")
        for season, conf in result['ml_predictions'].items():
            print(f"  {season}: {conf:.3f}")
    else:
        print("Crop file not found")

if __name__ == "__main__":
    test_improved_classifier()
