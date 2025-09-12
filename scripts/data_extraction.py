import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from color_detection import ColorDetector, analyze_clothing_colors


class ClothingCategory(Enum):
    """Clothing categories from DeepFashion2 dataset."""
    SHORT_SLEEVED_SHIRT = "short_sleeved_shirt"
    LONG_SLEEVED_SHIRT = "long_sleeved_shirt"
    SHORT_SLEEVED_OUTWEAR = "short_sleeved_outwear"
    LONG_SLEEVED_OUTWEAR = "long_sleeved_outwear"
    VEST = "vest"
    SLING = "sling"
    SHORTS = "shorts"
    TROUSERS = "trousers"
    SKIRT = "skirt"
    SHORT_SLEEVED_DRESS = "short_sleeved_dress"
    LONG_SLEEVED_DRESS = "long_sleeved_dress"
    VEST_DRESS = "vest_dress"
    SLING_DRESS = "sling_dress"


class StyleType(Enum):
    """Style categories for clothing."""
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTY = "sporty"
    ELEGANT = "elegant"
    VINTAGE = "vintage"
    MODERN = "modern"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"


@dataclass
class ClothingItem:
    """Data structure for a single clothing item."""
    # Detection data
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    category: ClothingCategory
    
    # Color data
    dominant_colors: List[Tuple[int, int, int]]
    color_names: List[str]
    color_percentages: List[float]
    color_harmony: str
    style_suggestion: str
    
    # Style analysis
    style_type: StyleType
    formality_level: int  # 1-5 scale
    season_suitability: List[str]  # ['spring', 'summer', 'fall', 'winter']
    
    # Additional attributes
    pattern_type: str  # 'solid', 'striped', 'patterned', 'textured'
    texture_analysis: str  # 'smooth', 'rough', 'shiny', 'matte'
    size_estimation: str  # 'small', 'medium', 'large'


class ClothingDataExtractor:
    """Extract and analyze clothing data from detection results."""
    
    def __init__(self):
        self.color_detector = ColorDetector(n_colors=5)
        self.category_mapping = {
            "short_sleeved_shirt": ClothingCategory.SHORT_SLEEVED_SHIRT,
            "long_sleeved_shirt": ClothingCategory.LONG_SLEEVED_SHIRT,
            "short_sleeved_outwear": ClothingCategory.SHORT_SLEEVED_OUTWEAR,
            "long_sleeved_outwear": ClothingCategory.LONG_SLEEVED_OUTWEAR,
            "vest": ClothingCategory.VEST,
            "sling": ClothingCategory.SLING,
            "shorts": ClothingCategory.SHORTS,
            "trousers": ClothingCategory.TROUSERS,
            "skirt": ClothingCategory.SKIRT,
            "short_sleeved_dress": ClothingCategory.SHORT_SLEEVED_DRESS,
            "long_sleeved_dress": ClothingCategory.LONG_SLEEVED_DRESS,
            "vest_dress": ClothingCategory.VEST_DRESS,
            "sling_dress": ClothingCategory.SLING_DRESS,
        }
        
        # Style rules based on clothing type and color
        self.style_rules = {
            ClothingCategory.SHORT_SLEEVED_SHIRT: {
                'casual': ['red', 'blue', 'green', 'yellow', 'orange'],
                'formal': ['white', 'navy', 'black', 'gray'],
                'sporty': ['red', 'blue', 'green', 'orange', 'yellow']
            },
            ClothingCategory.LONG_SLEEVED_SHIRT: {
                'formal': ['white', 'navy', 'black', 'gray', 'beige'],
                'casual': ['blue', 'green', 'red', 'purple'],
                'elegant': ['black', 'navy', 'maroon', 'gray']
            },
            ClothingCategory.TROUSERS: {
                'formal': ['black', 'navy', 'gray', 'beige'],
                'casual': ['blue', 'brown', 'gray', 'black'],
                'sporty': ['black', 'gray', 'blue', 'green']
            },
            ClothingCategory.SKIRT: {
                'elegant': ['black', 'navy', 'maroon', 'gray'],
                'casual': ['blue', 'green', 'red', 'purple'],
                'bohemian': ['brown', 'beige', 'olive', 'teal']
            },
            ClothingCategory.SHORT_SLEEVED_DRESS: {
                'casual': ['blue', 'green', 'red', 'yellow', 'orange'],
                'elegant': ['black', 'navy', 'maroon', 'gray'],
                'modern': ['white', 'black', 'gray', 'navy']
            },
            ClothingCategory.LONG_SLEEVED_DRESS: {
                'formal': ['black', 'navy', 'maroon', 'gray'],
                'elegant': ['black', 'navy', 'maroon', 'gray'],
                'vintage': ['brown', 'beige', 'olive', 'teal']
            }
        }
    
    def extract_clothing_data(self, image_path: str, detections: List[Dict]) -> List[ClothingItem]:
        """
        Extract comprehensive clothing data from detection results.
        
        Args:
            image_path: Path to input image
            detections: List of detection results from clothing detection model
            
        Returns:
            List of ClothingItem objects with extracted data
        """
        # Analyze colors for all detections
        color_analysis = analyze_clothing_colors(image_path, detections)
        
        clothing_items = []
        
        for i, detection in enumerate(detections):
            # Get corresponding color analysis
            color_data = color_analysis[i] if i < len(color_analysis) else None
            
            # Extract basic detection data
            bbox = detection['bbox']
            confidence = detection['score']
            category_name = detection['class_name']
            
            # Map category name to enum
            category = self.category_mapping.get(category_name, ClothingCategory.SHORT_SLEEVED_SHIRT)
            
            # Extract color data
            if color_data:
                dominant_colors = color_data['colors']['dominant_colors']
                color_names = color_data['colors']['color_names']
                color_percentages = color_data['colors']['percentages']
                color_harmony = color_data['harmony']['harmony_type']
                style_suggestion = color_data['harmony']['style_suggestion']
            else:
                dominant_colors = [(128, 128, 128)]  # Default gray
                color_names = ['gray']
                color_percentages = [100.0]
                color_harmony = 'unknown'
                style_suggestion = 'neutral'
            
            # Analyze style based on category and colors
            style_type = self._analyze_style(category, color_names)
            formality_level = self._analyze_formality(category, color_names, color_harmony)
            season_suitability = self._analyze_season(category, color_names)
            
            # Analyze pattern and texture (simplified)
            pattern_type = self._analyze_pattern(dominant_colors)
            texture_analysis = self._analyze_texture(category, color_names)
            size_estimation = self._estimate_size(bbox, category)
            
            # Create ClothingItem
            clothing_item = ClothingItem(
                bbox=bbox,
                confidence=confidence,
                category=category,
                dominant_colors=dominant_colors,
                color_names=color_names,
                color_percentages=color_percentages,
                color_harmony=color_harmony,
                style_suggestion=style_suggestion,
                style_type=style_type,
                formality_level=formality_level,
                season_suitability=season_suitability,
                pattern_type=pattern_type,
                texture_analysis=texture_analysis,
                size_estimation=size_estimation
            )
            
            clothing_items.append(clothing_item)
        
        return clothing_items
    
    def _analyze_style(self, category: ClothingCategory, color_names: List[str]) -> StyleType:
        """Analyze style type based on clothing category and colors."""
        if category not in self.style_rules:
            return StyleType.CASUAL
        
        # Get style rules for this category
        rules = self.style_rules[category]
        
        # Count matches for each style
        style_scores = {}
        for style, colors in rules.items():
            score = sum(1 for color in color_names if color in colors)
            style_scores[style] = score
        
        # Return style with highest score
        if style_scores:
            best_style = max(style_scores.items(), key=lambda x: x[1])[0]
            return StyleType(best_style)
        
        return StyleType.CASUAL
    
    def _analyze_formality(self, category: ClothingCategory, color_names: List[str], 
                          color_harmony: str) -> int:
        """Analyze formality level (1-5 scale)."""
        formality = 3  # Default neutral
        
        # Category-based formality
        formal_categories = [ClothingCategory.LONG_SLEEVED_SHIRT, ClothingCategory.TROUSERS, 
                           ClothingCategory.LONG_SLEEVED_DRESS]
        casual_categories = [ClothingCategory.SHORTS, ClothingCategory.SHORT_SLEEVED_SHIRT]
        
        if category in formal_categories:
            formality += 1
        elif category in casual_categories:
            formality -= 1
        
        # Color-based formality
        formal_colors = ['black', 'navy', 'white', 'gray', 'maroon']
        casual_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'pink']
        
        if any(color in formal_colors for color in color_names):
            formality += 1
        elif any(color in casual_colors for color in color_names):
            formality -= 1
        
        # Harmony-based formality
        if color_harmony == 'monochromatic':
            formality += 1
        elif color_harmony == 'complementary':
            formality -= 1
        
        return max(1, min(5, formality))
    
    def _analyze_season(self, category: ClothingCategory, color_names: List[str]) -> List[str]:
        """Analyze season suitability."""
        seasons = []
        
        # Category-based season analysis
        if category in [ClothingCategory.SHORT_SLEEVED_SHIRT, ClothingCategory.SHORTS, 
                       ClothingCategory.SHORT_SLEEVED_DRESS]:
            seasons.extend(['spring', 'summer'])
        elif category in [ClothingCategory.LONG_SLEEVED_OUTWEAR, ClothingCategory.LONG_SLEEVED_DRESS]:
            seasons.extend(['fall', 'winter'])
        else:
            seasons.extend(['spring', 'summer', 'fall'])
        
        # Color-based season analysis
        warm_colors = ['red', 'orange', 'yellow', 'pink']
        cool_colors = ['blue', 'green', 'purple', 'teal']
        neutral_colors = ['black', 'white', 'gray', 'brown', 'beige']
        
        if any(color in warm_colors for color in color_names):
            seasons.extend(['spring', 'summer'])
        elif any(color in cool_colors for color in color_names):
            seasons.extend(['fall', 'winter'])
        
        return list(set(seasons))  # Remove duplicates
    
    def _analyze_pattern(self, dominant_colors: List[Tuple[int, int, int]]) -> str:
        """Analyze pattern type based on color distribution."""
        if len(dominant_colors) == 1:
            return 'solid'
        elif len(dominant_colors) == 2:
            # Check if colors are complementary
            color1, color2 = dominant_colors
            if self._are_complementary(color1, color2):
                return 'striped'
            else:
                return 'two-tone'
        else:
            return 'patterned'
    
    def _analyze_texture(self, category: ClothingCategory, color_names: List[str]) -> str:
        """Analyze texture based on category and colors."""
        # Category-based texture
        if category in [ClothingCategory.SHORT_SLEEVED_SHIRT, ClothingCategory.LONG_SLEEVED_SHIRT]:
            return 'smooth'
        elif category in [ClothingCategory.TROUSERS, ClothingCategory.SHORTS]:
            return 'matte'
        elif category in [ClothingCategory.SKIRT, ClothingCategory.SHORT_SLEEVED_DRESS]:
            return 'smooth'
        
        # Color-based texture
        if 'black' in color_names or 'navy' in color_names:
            return 'smooth'
        elif 'brown' in color_names or 'beige' in color_names:
            return 'matte'
        
        return 'smooth'
    
    def _estimate_size(self, bbox: List[float], category: ClothingCategory) -> str:
        """Estimate clothing size based on bounding box."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Size estimation based on area (simplified)
        if area < 5000:
            return 'small'
        elif area < 15000:
            return 'medium'
        else:
            return 'large'
    
    def _are_complementary(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> bool:
        """Check if two colors are complementary."""
        # Simplified complementary check
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Check if colors are roughly opposite
        r_diff = abs(r1 - r2)
        g_diff = abs(g1 - g2)
        b_diff = abs(b1 - b2)
        
        return r_diff > 100 and g_diff > 100 and b_diff > 100
    
    def export_clothing_data(self, clothing_items: List[ClothingItem], 
                           output_path: str) -> None:
        """Export clothing data to JSON file."""
        import numpy as _np
        def _to_py(obj):
            if isinstance(obj, dict):
                return {k: _to_py(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_py(x) for x in obj]
            elif isinstance(obj, tuple):
                return [_to_py(x) for x in obj]
            elif isinstance(obj, set):
                return [_to_py(x) for x in obj]
            elif isinstance(obj, (_np.integer,)):
                return int(obj)
            elif isinstance(obj, (_np.floating,)):
                return float(obj)
            elif isinstance(obj, (_np.bool_,)):
                return bool(obj)
            elif isinstance(obj, _np.ndarray):
                return _to_py(obj.tolist())
            else:
                return obj

        data = []
        
        for item in clothing_items:
            item_data = {
                'bbox': item.bbox,
                'confidence': item.confidence,
                'category': item.category.value,
                'colors': {
                    'dominant_colors': item.dominant_colors,
                    'color_names': item.color_names,
                    'percentages': item.color_percentages
                },
                'style': {
                    'type': item.style_type.value,
                    'formality_level': item.formality_level,
                    'harmony': item.color_harmony,
                    'suggestion': item.style_suggestion
                },
                'season_suitability': item.season_suitability,
                'pattern_type': item.pattern_type,
                'texture_analysis': item.texture_analysis,
                'size_estimation': item.size_estimation
            }
            data.append(item_data)
        
        with open(output_path, 'w') as f:
            json.dump(_to_py(data), f, indent=2)
    
    def get_outfit_summary(self, clothing_items: List[ClothingItem]) -> Dict[str, Any]:
        """Get overall outfit summary and analysis."""
        if not clothing_items:
            return {'error': 'No clothing items detected'}
        
        # Count items by category
        category_counts = {}
        for item in clothing_items:
            cat = item.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Analyze overall style
        style_types = [item.style_type.value for item in clothing_items]
        dominant_style = max(set(style_types), key=style_types.count)
        
        # Analyze overall formality
        avg_formality = sum(item.formality_level for item in clothing_items) / len(clothing_items)
        
        # Analyze color harmony
        all_colors = []
        for item in clothing_items:
            all_colors.extend(item.dominant_colors)
        
        overall_harmony = self.color_detector.analyze_color_harmony(all_colors)
        
        # Season suitability
        all_seasons = set()
        for item in clothing_items:
            all_seasons.update(item.season_suitability)
        
        return {
            'total_items': len(clothing_items),
            'category_breakdown': category_counts,
            'dominant_style': dominant_style,
            'average_formality': round(avg_formality, 2),
            'overall_harmony': overall_harmony,
            'season_suitability': list(all_seasons),
            'outfit_completeness': self._assess_completeness(category_counts)
        }
    
    def _assess_completeness(self, category_counts: Dict[str, int]) -> str:
        """Assess how complete the outfit is."""
        has_top = any(cat in category_counts for cat in 
                     ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_dress', 'long_sleeved_dress'])
        has_bottom = any(cat in category_counts for cat in 
                        ['trousers', 'shorts', 'skirt'])
        has_outwear = any(cat in category_counts for cat in 
                         ['short_sleeved_outwear', 'long_sleeved_outwear'])
        
        if has_top and has_bottom and has_outwear:
            return 'complete'
        elif has_top and has_bottom:
            return 'basic'
        elif has_top or has_bottom:
            return 'partial'
        else:
            return 'incomplete'


if __name__ == "__main__":
    print("Clothing Data Extraction Module")
    print("This module extracts comprehensive clothing data from detection results.")
