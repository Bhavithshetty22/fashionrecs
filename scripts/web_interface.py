"""
Web interface for the Fashion AI Pipeline
Simple Flask-based web application for outfit analysis and recommendations.
"""

import os
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

# Import our pipeline
from fashion_ai_pipeline import FashionAIPipeline

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, '..'))
_TEMPLATES_DIR = os.path.join(_PROJECT_ROOT, 'templates')

app = Flask(__name__, template_folder=_TEMPLATES_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global pipeline instance
pipeline = None

def init_pipeline(model_path: str):
    """Initialize the fashion AI pipeline."""
    global pipeline
    try:
        pipeline = FashionAIPipeline(model_path)
        return True
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process with pipeline
            if pipeline is None:
                return jsonify({'error': 'Pipeline not initialized'}), 500
            
            # Process image
            results = pipeline.process_image(filepath, app.config['OUTPUT_FOLDER'])
            
            # Create visualization
            vis_path = pipeline.create_visualization(filepath, results)
            
            # Convert results to JSON-serializable format
            serializable_results = make_serializable(results)
            
            return jsonify({
                'success': True,
                'results': serializable_results,
                'visualization': vis_path
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/visualization/<path:filename>')
def get_visualization(filename):
    """Serve visualization images."""
    try:
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    except Exception as e:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for image analysis."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Process image
            results = pipeline.process_image(tmp_path, app.config['OUTPUT_FOLDER'])
            
            # Create visualization
            vis_path = pipeline.create_visualization(tmp_path, results)
            
            # Convert visualization to base64
            with open(vis_path, 'rb') as f:
                vis_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Convert results to JSON-serializable format
            serializable_results = make_serializable(results)
            
            return jsonify({
                'success': True,
                'results': serializable_results,
                'visualization': f"data:image/jpeg;base64,{vis_data}"
            })
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get pipeline status."""
    return jsonify({
        'pipeline_initialized': pipeline is not None,
        'model_loaded': pipeline is not None and pipeline.detection_model is not None
    })

def allowed_file(filename):
    """Check if file type is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_serializable(obj):
    """Convert objects to JSON-serializable format (handles numpy, enums, mappingproxy, dataclasses)."""
    import numpy as _np
    from types import MappingProxyType as _MappingProxyType
    from dataclasses import asdict, is_dataclass
    from enum import Enum as _Enum

    if is_dataclass(obj):
        return make_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, _MappingProxyType):
        return make_serializable(dict(obj))
    if isinstance(obj, _Enum):
        return make_serializable(obj.value)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return make_serializable(obj.tolist())
    if hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    return obj

if __name__ == '__main__':
    # Initialize pipeline with model path
    model_path = "./checkpoints_df2/epoch_2_step_40000.pth"
    
    if init_pipeline(model_path):
        print("✅ Fashion AI Pipeline initialized successfully")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize pipeline")
        print("Please check your model path and try again")

