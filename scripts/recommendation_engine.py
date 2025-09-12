import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from data_extraction import ClothingItem, ClothingCategory, StyleType


class RecommendationType(Enum):
    """Types of recommendations."""
    COMPLETE_OUTFIT = "complete_outfit"
    ADD_ACCESSORIES = "add_accessories"
    COLOR_SUGGESTIONS = "color_suggestions"
    STYLE_IMPROVEMENTS = "style_improvements"
    SEASONAL_ADJUSTMENTS = "seasonal_adjustments"


@dataclass
class Recommendation:
    """A single recommendation item."""
    type: RecommendationType
    title: str
    description: str
    confidence: float
    priority: int  # 1-5, higher is more important
    items: List[Dict[str, Any]]  # Suggested items or changes
    reasoning: str


@dataclass
class OutfitSuggestion:
    """Complete outfit suggestion."""
    outfit_name: str
    description: str
    items: List[Dict[str, Any]]
    style_type: StyleType
    formality_level: int
    season_suitability: List[str]
    color_harmony: str
    confidence: float
    reasoning: str


class FashionRecommendationEngine:
    """AI-powered fashion recommendation engine."""
    
    def __init__(self):
        self.color_harmony_rules = {
            'complementary': {
                'description': 'Bold and contrasting colors',
                'examples': [('red', 'green'), ('blue', 'orange'), ('purple', 'yellow')]
            },
            'analogous': {
                'description': 'Harmonious and soothing colors',
                'examples': [('blue', 'green'), ('red', 'orange'), ('purple', 'blue')]
            },
            'triadic': {
                'description': 'Vibrant and balanced colors',
                'examples': [('red', 'blue', 'yellow'), ('green', 'purple', 'orange')]
            },
            'monochromatic': {
                'description': 'Elegant and sophisticated colors',
                'examples': [('navy', 'light_blue'), ('black', 'gray'), ('maroon', 'pink')]
            }
        }
        
        self.style_guidelines = {
            StyleType.CASUAL: {
                'description': 'Relaxed and comfortable',
                'colors': ['blue', 'green', 'red', 'yellow', 'orange', 'brown'],
                'patterns': ['solid', 'striped', 'patterned'],
                'accessories': ['sneakers', 'casual_watch', 'backpack']
            },
            StyleType.FORMAL: {
                'description': 'Professional and polished',
                'colors': ['black', 'navy', 'white', 'gray', 'maroon'],
                'patterns': ['solid', 'subtle_patterns'],
                'accessories': ['dress_shoes', 'formal_watch', 'briefcase']
            },
            StyleType.ELEGANT: {
                'description': 'Sophisticated and refined',
                'colors': ['black', 'navy', 'maroon', 'gray', 'beige'],
                'patterns': ['solid', 'monochromatic'],
                'accessories': ['heels', 'elegant_watch', 'clutch']
            },
            StyleType.SPORTY: {
                'description': 'Active and dynamic',
                'colors': ['red', 'blue', 'green', 'orange', 'yellow', 'black'],
                'patterns': ['solid', 'striped', 'athletic_patterns'],
                'accessories': ['sneakers', 'sports_watch', 'gym_bag']
            }
        }
        
        self.outfit_templates = {
            'business_casual': {
                'items': ['long_sleeved_shirt', 'trousers', 'shoes'],
                'style': StyleType.CASUAL,
                'formality': 3,
                'description': 'Professional yet comfortable'
            },
            'formal_business': {
                'items': ['long_sleeved_shirt', 'trousers', 'shoes', 'blazer'],
                'style': StyleType.FORMAL,
                'formality': 4,
                'description': 'Professional and polished'
            },
            'casual_weekend': {
                'items': ['short_sleeved_shirt', 'shorts', 'sneakers'],
                'style': StyleType.CASUAL,
                'formality': 2,
                'description': 'Relaxed and comfortable'
            },
            'elegant_evening': {
                'items': ['long_sleeved_dress', 'heels', 'clutch'],
                'style': StyleType.ELEGANT,
                'formality': 5,
                'description': 'Sophisticated and refined'
            },
            'sporty_active': {
                'items': ['short_sleeved_shirt', 'shorts', 'sneakers'],
                'style': StyleType.SPORTY,
                'formality': 1,
                'description': 'Active and dynamic'
            }
        }
    
    def generate_recommendations(self, clothing_items: List[ClothingItem], 
                               user_preferences: Dict[str, Any] = None) -> List[Recommendation]:
        """
        Generate fashion recommendations based on detected clothing items.
        
        Args:
            clothing_items: List of detected clothing items
            user_preferences: Optional user preferences
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze current outfit
        outfit_analysis = self._analyze_current_outfit(clothing_items)
        
        # Generate different types of recommendations
        recommendations.extend(self._recommend_missing_items(outfit_analysis))
        recommendations.extend(self._recommend_color_improvements(outfit_analysis))
        recommendations.extend(self._recommend_style_improvements(outfit_analysis))
        recommendations.extend(self._recommend_accessories(outfit_analysis))
        recommendations.extend(self._recommend_seasonal_adjustments(outfit_analysis))
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return recommendations
    
    def suggest_complete_outfits(self, clothing_items: List[ClothingItem], 
                               style_preference: StyleType = None) -> List[OutfitSuggestion]:
        """
        Suggest complete outfits based on detected items and preferences.
        
        Args:
            clothing_items: List of detected clothing items
            style_preference: Preferred style type
            
        Returns:
            List of complete outfit suggestions
        """
        outfit_suggestions = []
        
        # Analyze current items
        current_categories = {item.category for item in clothing_items}
        current_colors = [color for item in clothing_items for color in item.dominant_colors]
        current_style = self._determine_dominant_style(clothing_items)
        
        # Generate outfit suggestions
        for template_name, template in self.outfit_templates.items():
            if style_preference and template['style'] != style_preference:
                continue
            
            # Check if we can build this outfit
            if self._can_build_outfit(template, current_categories):
                suggestion = self._create_outfit_suggestion(
                    template_name, template, clothing_items, current_colors
                )
                outfit_suggestions.append(suggestion)
        
        # Sort by confidence and relevance
        outfit_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return outfit_suggestions
    
    def _analyze_current_outfit(self, clothing_items: List[ClothingItem]) -> Dict[str, Any]:
        """Analyze the current outfit composition."""
        if not clothing_items:
            return {'error': 'No clothing items detected'}
        
        # Count items by category
        category_counts = {}
        for item in clothing_items:
            cat = item.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Analyze colors
        all_colors = [color for item in clothing_items for color in item.dominant_colors]
        color_harmony = self._analyze_color_harmony(all_colors)
        
        # Analyze style
        dominant_style = self._determine_dominant_style(clothing_items)
        avg_formality = sum(item.formality_level for item in clothing_items) / len(clothing_items)
        
        # Check completeness
        has_top = any(cat in category_counts for cat in 
                     ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_dress', 'long_sleeved_dress'])
        has_bottom = any(cat in category_counts for cat in 
                        ['trousers', 'shorts', 'skirt'])
        has_outwear = any(cat in category_counts for cat in 
                         ['short_sleeved_outwear', 'long_sleeved_outwear'])
        
        return {
            'total_items': len(clothing_items),
            'category_counts': category_counts,
            'dominant_style': dominant_style,
            'avg_formality': avg_formality,
            'color_harmony': color_harmony,
            'completeness': {
                'has_top': has_top,
                'has_bottom': has_bottom,
                'has_outwear': has_outwear,
                'completeness_score': sum([has_top, has_bottom, has_outwear]) / 3
            }
        }
    
    def _recommend_missing_items(self, outfit_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Recommend missing essential items."""
        recommendations = []
        completeness = outfit_analysis['completeness']
        
        if not completeness['has_top']:
            recommendations.append(Recommendation(
                type=RecommendationType.COMPLETE_OUTFIT,
                title="Add a Top",
                description="Your outfit is missing a top. Consider adding a shirt, blouse, or dress.",
                confidence=0.9,
                priority=5,
                items=[
                    {'type': 'short_sleeved_shirt', 'suggested_colors': ['white', 'blue', 'navy']},
                    {'type': 'long_sleeved_shirt', 'suggested_colors': ['white', 'navy', 'black']},
                    {'type': 'short_sleeved_dress', 'suggested_colors': ['black', 'navy', 'blue']}
                ],
                reasoning="A top is essential for a complete outfit."
            ))
        
        if not completeness['has_bottom']:
            recommendations.append(Recommendation(
                type=RecommendationType.COMPLETE_OUTFIT,
                title="Add Bottom Wear",
                description="Your outfit is missing bottom wear. Consider adding pants, shorts, or a skirt.",
                confidence=0.9,
                priority=5,
                items=[
                    {'type': 'trousers', 'suggested_colors': ['black', 'navy', 'gray']},
                    {'type': 'shorts', 'suggested_colors': ['blue', 'khaki', 'black']},
                    {'type': 'skirt', 'suggested_colors': ['black', 'navy', 'gray']}
                ],
                reasoning="Bottom wear is essential for a complete outfit."
            ))
        
        if not completeness['has_outwear'] and outfit_analysis['avg_formality'] > 3:
            recommendations.append(Recommendation(
                type=RecommendationType.COMPLETE_OUTFIT,
                title="Add Outerwear",
                description="Consider adding a jacket or blazer for a more polished look.",
                confidence=0.7,
                priority=3,
                items=[
                    {'type': 'long_sleeved_outwear', 'suggested_colors': ['black', 'navy', 'gray']},
                    {'type': 'short_sleeved_outwear', 'suggested_colors': ['black', 'navy', 'beige']}
                ],
                reasoning="Outerwear adds polish to formal outfits."
            ))
        
        return recommendations
    
    def _recommend_color_improvements(self, outfit_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Recommend color improvements."""
        recommendations = []
        color_harmony = outfit_analysis['color_harmony']
        
        if color_harmony == 'unknown' or color_harmony == 'mixed':
            recommendations.append(Recommendation(
                type=RecommendationType.COLOR_SUGGESTIONS,
                title="Improve Color Harmony",
                description="Your outfit could benefit from better color coordination.",
                confidence=0.8,
                priority=4,
                items=[
                    {'suggestion': 'Try complementary colors', 'examples': ['red + green', 'blue + orange']},
                    {'suggestion': 'Use analogous colors', 'examples': ['blue + green', 'red + orange']},
                    {'suggestion': 'Create monochromatic look', 'examples': ['navy + light blue', 'black + gray']}
                ],
                reasoning="Better color harmony creates a more cohesive look."
            ))
        
        return recommendations
    
    def _recommend_style_improvements(self, outfit_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Recommend style improvements."""
        recommendations = []
        dominant_style = outfit_analysis['dominant_style']
        avg_formality = outfit_analysis['avg_formality']
        
        # Check for style consistency
        if dominant_style == StyleType.CASUAL and avg_formality > 4:
            recommendations.append(Recommendation(
                type=RecommendationType.STYLE_IMPROVEMENTS,
                title="Style Consistency",
                description="Your outfit mixes casual and formal elements. Consider making it more consistent.",
                confidence=0.7,
                priority=3,
                items=[
                    {'suggestion': 'Add casual accessories', 'examples': ['sneakers', 'casual watch']},
                    {'suggestion': 'Use more casual colors', 'examples': ['blue', 'green', 'brown']}
                ],
                reasoning="Consistent style creates a more polished look."
            ))
        
        return recommendations
    
    def _recommend_accessories(self, outfit_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Recommend accessories."""
        recommendations = []
        
        # Always recommend accessories for complete outfits
        if outfit_analysis['completeness']['completeness_score'] > 0.7:
            recommendations.append(Recommendation(
                type=RecommendationType.ADD_ACCESSORIES,
                title="Add Accessories",
                description="Complete your look with appropriate accessories.",
                confidence=0.8,
                priority=2,
                items=[
                    {'type': 'shoes', 'suggested_styles': ['dress shoes', 'sneakers', 'heels']},
                    {'type': 'watch', 'suggested_styles': ['formal watch', 'casual watch', 'sports watch']},
                    {'type': 'bag', 'suggested_styles': ['briefcase', 'handbag', 'backpack']}
                ],
                reasoning="Accessories complete and enhance your outfit."
            ))
        
        return recommendations
    
    def _recommend_seasonal_adjustments(self, outfit_analysis: Dict[str, Any]) -> List[Recommendation]:
        """Recommend seasonal adjustments."""
        recommendations = []
        
        # This would typically use current season data
        # For now, provide general seasonal advice
        recommendations.append(Recommendation(
            type=RecommendationType.SEASONAL_ADJUSTMENTS,
            title="Seasonal Considerations",
            description="Consider the current season when choosing your outfit.",
            confidence=0.6,
            priority=1,
            items=[
                {'season': 'spring', 'suggestions': ['light colors', 'layered clothing', 'light jackets']},
                {'season': 'summer', 'suggestions': ['breathable fabrics', 'light colors', 'short sleeves']},
                {'season': 'fall', 'suggestions': ['warmer colors', 'layered clothing', 'medium weight fabrics']},
                {'season': 'winter', 'suggestions': ['dark colors', 'heavy fabrics', 'warm outerwear']}
            ],
            reasoning="Seasonal appropriateness enhances comfort and style."
        ))
        
        return recommendations
    
    def _create_outfit_suggestion(self, template_name: str, template: Dict[str, Any], 
                                current_items: List[ClothingItem], 
                                current_colors: List[Tuple[int, int, int]]) -> OutfitSuggestion:
        """Create a complete outfit suggestion."""
        # Generate items for this outfit
        suggested_items = []
        
        for item_type in template['items']:
            # Check if we already have this type
            existing_item = next((item for item in current_items 
                                if item.category.value == item_type), None)
            
            if existing_item:
                suggested_items.append({
                    'type': item_type,
                    'status': 'existing',
                    'item': existing_item,
                    'suggested_colors': existing_item.color_names
                })
            else:
                # Suggest new item
                suggested_colors = self._suggest_colors_for_item(item_type, current_colors)
                suggested_items.append({
                    'type': item_type,
                    'status': 'suggested',
                    'suggested_colors': suggested_colors,
                    'reasoning': f"Add {item_type} to complete the outfit"
                })
        
        # Calculate confidence based on existing items
        existing_count = sum(1 for item in suggested_items if item['status'] == 'existing')
        confidence = existing_count / len(suggested_items)
        
        return OutfitSuggestion(
            outfit_name=template_name.replace('_', ' ').title(),
            description=template['description'],
            items=suggested_items,
            style_type=template['style'],
            formality_level=template['formality'],
            season_suitability=['spring', 'summer', 'fall', 'winter'],  # Default
            color_harmony='balanced',
            confidence=confidence,
            reasoning=f"Based on {template_name} template with {existing_count}/{len(suggested_items)} existing items"
        )
    
    def _can_build_outfit(self, template: Dict[str, Any], 
                         current_categories: set) -> bool:
        """Check if we can build this outfit template."""
        required_items = template['items']
        available_items = {cat.value for cat in current_categories}
        
        # Check if we have at least some of the required items
        overlap = len(set(required_items) & available_items)
        return overlap > 0
    
    def _suggest_colors_for_item(self, item_type: str, current_colors: List[Tuple[int, int, int]]) -> List[str]:
        """Suggest colors for a specific item type."""
        # This is a simplified version - in practice, you'd have more sophisticated color matching
        base_colors = ['black', 'white', 'navy', 'gray', 'blue']
        
        # Add colors that complement current colors
        if current_colors:
            # Simple complementary color suggestion
            return base_colors + ['red', 'green', 'brown']
        
        return base_colors
    
    def _determine_dominant_style(self, clothing_items: List[ClothingItem]) -> StyleType:
        """Determine the dominant style of the outfit."""
        if not clothing_items:
            return StyleType.CASUAL
        
        style_counts = {}
        for item in clothing_items:
            style = item.style_type
            style_counts[style] = style_counts.get(style, 0) + 1
        
        return max(style_counts.items(), key=lambda x: x[1])[0]
    
    def _analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> str:
        """Analyze color harmony of the outfit."""
        if len(colors) < 2:
            return 'monochromatic'
        
        # Simplified color harmony analysis
        # In practice, you'd use more sophisticated color theory
        return 'balanced'  # Default to balanced
    
    def _to_py(self, obj):
        """Convert objects to JSON-serializable Python types."""
        import numpy as _np
        from dataclasses import asdict, is_dataclass
        from enum import Enum as _Enum
        if is_dataclass(obj):
            return self._to_py(asdict(obj))
        if isinstance(obj, dict):
            return {k: self._to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_py(x) for x in obj]
        if isinstance(obj, tuple):
            return [self._to_py(x) for x in obj]
        if isinstance(obj, set):
            return [self._to_py(x) for x in obj]
        if isinstance(obj, _Enum):
            return obj.value
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return self._to_py(obj.tolist())
        return obj

    def export_recommendations(self, recommendations: List[Recommendation], 
                             output_path: str) -> None:
        """Export recommendations to JSON file."""
        data = []
        for rec in recommendations:
            rec_data = {
                'type': rec.type.value,
                'title': rec.title,
                'description': rec.description,
                'confidence': rec.confidence,
                'priority': rec.priority,
                'items': rec.items,
                'reasoning': rec.reasoning
            }
            data.append(rec_data)
        
        with open(output_path, 'w') as f:
            json.dump(self._to_py(data), f, indent=2)
    
    def export_outfit_suggestions(self, suggestions: List[OutfitSuggestion], 
                                output_path: str) -> None:
        """Export outfit suggestions to JSON file."""
        data = []
        for suggestion in suggestions:
            suggestion_data = {
                'outfit_name': suggestion.outfit_name,
                'description': suggestion.description,
                'items': suggestion.items,
                'style_type': suggestion.style_type.value,
                'formality_level': suggestion.formality_level,
                'season_suitability': suggestion.season_suitability,
                'color_harmony': suggestion.color_harmony,
                'confidence': suggestion.confidence,
                'reasoning': suggestion.reasoning
            }
            data.append(suggestion_data)
        
        with open(output_path, 'w') as f:
            json.dump(self._to_py(data), f, indent=2)


if __name__ == "__main__":
    print("Fashion Recommendation Engine")
    print("This module provides AI-powered fashion recommendations.")
