import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib
# Force non-interactive backend for headless/CLI usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from typing import List, Tuple, Dict, Any
import torch
from PIL import Image
import torchvision.transforms as transforms


class ColorDetector:
    """
    Color detection system using K-means clustering for fashion items.
    Detects dominant colors, color harmony, and provides color analysis.
    """
    
    def __init__(self, n_colors: int = 5, random_state: int = 42):
        self.n_colors = n_colors
        self.random_state = random_state
        self.color_names = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'gray': (128, 128, 128),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'navy': (0, 0, 128),
            'beige': (245, 245, 220),
            'maroon': (128, 0, 0),
            'olive': (128, 128, 0),
            'teal': (0, 128, 128),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'lime': (0, 255, 0),
            'indigo': (75, 0, 130)
        }
    
    def extract_clothing_region(self, image: np.ndarray, bbox: List[float], mask: np.ndarray = None) -> np.ndarray:
        """
        Extract clothing region from image using bounding box and optional mask.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Optional mask for more precise extraction
            
        Returns:
            Extracted clothing region
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Extract region
        clothing_region = image[y1:y2, x1:x2]
        
        # Apply mask if provided
        if mask is not None:
            # Resize mask to match region size
            mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
            # Apply mask
            clothing_region = cv2.bitwise_and(clothing_region, clothing_region, mask=mask_resized)
        
        return clothing_region
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for color analysis.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (for faster processing)
        h, w = image_rgb.shape[:2]
        if h > 512 or w > 512:
            scale = min(512/h, 512/w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        return image_rgb
    
    def detect_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect dominant colors in image using K-means clustering.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Dictionary containing color analysis results
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove black/white pixels (background)
        # Filter out very dark and very light pixels
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 220)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < self.n_colors:
            # If not enough pixels, use all pixels
            filtered_pixels = pixels
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=self.random_state, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        color_counts = Counter(labels)
        
        # Calculate percentages
        total_pixels = len(labels)
        color_percentages = {i: count/total_pixels * 100 for i, count in color_counts.items()}
        
        # Sort colors by frequency
        sorted_colors = sorted(zip(colors, color_percentages.values()), 
                             key=lambda x: x[1], reverse=True)
        
        # Convert to RGB tuples
        dominant_colors = [tuple(color) for color, _ in sorted_colors]
        percentages = [pct for _, pct in sorted_colors]
        
        # Identify color names
        color_names = [self._get_closest_color_name(color) for color in dominant_colors]
        
        return {
            'dominant_colors': dominant_colors,
            'percentages': percentages,
            'color_names': color_names,
            'n_colors_found': len(dominant_colors)
        }
    
    def _get_closest_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Find the closest named color for an RGB value.
        
        Args:
            rgb_color: RGB color tuple
            
        Returns:
            Closest color name
        """
        r, g, b = rgb_color
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for name, (cr, cg, cb) in self.color_names.items():
            distance = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color
    
    def analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """
        Analyze color harmony and provide style suggestions.
        
        Args:
            colors: List of RGB color tuples
            
        Returns:
            Color harmony analysis
        """
        if len(colors) < 2:
            return {'harmony_type': 'single', 'style_suggestion': 'monochromatic'}
        
        # Convert to HSV for better color analysis
        hsv_colors = [self._rgb_to_hsv(color) for color in colors]
        hues = [hsv[0] for hsv in hsv_colors]
        
        # Analyze color relationships
        harmony_analysis = {
            'complementary': self._check_complementary(hues),
            'analogous': self._check_analogous(hues),
            'triadic': self._check_triadic(hues),
            'monochromatic': self._check_monochromatic(hsv_colors)
        }
        
        # Determine dominant harmony type
        dominant_harmony = max(harmony_analysis.items(), key=lambda x: x[1])
        
        # Style suggestions based on harmony
        style_suggestions = {
            'complementary': 'bold and contrasting',
            'analogous': 'harmonious and soothing',
            'triadic': 'vibrant and balanced',
            'monochromatic': 'elegant and sophisticated'
        }
        
        return {
            'harmony_type': dominant_harmony[0],
            'confidence': dominant_harmony[1],
            'style_suggestion': style_suggestions.get(dominant_harmony[0], 'mixed'),
            'harmony_analysis': harmony_analysis
        }
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB (0-255) to HSV with hue in degrees (0-360).

        Uses colorsys to avoid OpenCV uint8 range issues and returns
        (H_degrees, S_percent, V_percent).
        """
        import colorsys
        r, g, b = [float(x) / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)  # h in [0,1), s,v in [0,1]
        return (h * 360.0, s * 100.0, v * 100.0)
    
    def _check_complementary(self, hues: List[float]) -> float:
        """Check if colors are complementary (opposite on color wheel)."""
        if len(hues) < 2:
            return 0.0
        
        # Check for colors that are ~180 degrees apart
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = abs(h1 - h2)
                if min(diff, 360 - diff) < 30:  # Within 30 degrees of complementary
                    return 0.8
        return 0.0
    
    def _check_analogous(self, hues: List[float]) -> float:
        """Check if colors are analogous (adjacent on color wheel)."""
        if len(hues) < 2:
            return 0.0
        
        hues_sorted = sorted(hues)
        for i in range(len(hues_sorted) - 1):
            diff = hues_sorted[i+1] - hues_sorted[i]
            if diff > 60:  # Not close enough
                return 0.0
        return 0.9
    
    def _check_triadic(self, hues: List[float]) -> float:
        """Check if colors are triadic (120 degrees apart)."""
        if len(hues) < 3:
            return 0.0
        
        # Check for three colors roughly 120 degrees apart
        hues_sorted = sorted(hues)
        for i in range(len(hues_sorted) - 2):
            diff1 = hues_sorted[i+1] - hues_sorted[i]
            diff2 = hues_sorted[i+2] - hues_sorted[i+1]
            if 100 < diff1 < 140 and 100 < diff2 < 140:
                return 0.8
        return 0.0
    
    def _check_monochromatic(self, hsv_colors: List[Tuple[float, float, float]]) -> float:
        """Check if colors are monochromatic (same hue, different saturation/value)."""
        if len(hsv_colors) < 2:
            return 0.0
        
        hues = [hsv[0] for hsv in hsv_colors]
        hue_variance = np.var(hues)
        return 0.9 if hue_variance < 20 else 0.0
    
    def visualize_colors(self, colors: List[Tuple[int, int, int]], 
                        percentages: List[float], save_path: str = None) -> np.ndarray:
        """
        Create a visualization of detected colors.
        
        Args:
            colors: List of RGB color tuples
            percentages: List of color percentages
            save_path: Optional path to save visualization
            
        Returns:
            Color palette visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        # Attach Agg canvas explicitly for safe rasterization
        canvas = FigureCanvas(fig)
        
        # Create color bars
        for i, (color, pct) in enumerate(zip(colors, percentages)):
            ax.barh(0, pct, left=sum(percentages[:i]), color=[c/255 for c in color], 
                   label=f'{color} ({pct:.1f}%)')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Color Distribution (%)')
        ax.set_title('Detected Color Palette')
        ax.set_yticks([])
        
        # Save to disk if requested
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Robust conversion to numpy via PNG buffer (works on all backends)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        vis_array = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # Convert BGR (cv2) to RGB for consistency
        if vis_array is not None:
            vis_array = vis_array[:, :, ::-1]
        
        plt.close(fig)
        return vis_array


def analyze_clothing_colors(image_path: str, detections: List[Dict], 
                          output_dir: str = "./color_analysis") -> List[Dict]:
    """
    Analyze colors for all detected clothing items.
    
    Args:
        image_path: Path to input image
        detections: List of detection results from clothing detection model
        output_dir: Directory to save color analysis results
        
    Returns:
        List of color analysis results for each detection
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize color detector
    color_detector = ColorDetector(n_colors=5)
    
    results = []
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['score']
        clothing_type = detection['class_name']
        
        # Extract clothing region
        clothing_region = color_detector.extract_clothing_region(image, bbox)
        
        if clothing_region.size == 0:
            continue
        
        # Preprocess for color analysis
        processed_region = color_detector.preprocess_image(clothing_region)
        
        # Detect colors
        color_analysis = color_detector.detect_colors(processed_region)
        
        # Analyze color harmony
        harmony_analysis = color_detector.analyze_color_harmony(color_analysis['dominant_colors'])
        
        # Create visualization
        vis_path = os.path.join(output_dir, f"colors_{i}.png")
        color_visualization = color_detector.visualize_colors(
            color_analysis['dominant_colors'], 
            color_analysis['percentages'],
            vis_path
        )
        
        # Store results
        result = {
            'detection_id': i,
            'clothing_type': clothing_type,
            'confidence': confidence,
            'bbox': bbox,
            'colors': {
                'dominant_colors': color_analysis['dominant_colors'],
                'percentages': color_analysis['percentages'],
                'color_names': color_analysis['color_names']
            },
            'harmony': harmony_analysis,
            'visualization_path': vis_path
        }
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Color Detection Module for Fashion AI Pipeline")
    print("This module will be integrated with clothing detection results.")
