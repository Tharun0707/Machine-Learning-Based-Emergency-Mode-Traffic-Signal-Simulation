from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add error handling for imports
try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print("❌ PyTorch not found. Please install it first:")
    print("pip install torch torchvision")
    sys.exit(1)

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print("❌ OpenCV not found. Please install it:")
    print("pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
    import io
    import pandas as pd
    import tempfile
    import time
    print("✅ All dependencies imported successfully")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variable to store the model
model = None

def load_model():
    """Load the YOLO model with error handling"""
    global model
    
    # Look for model in models folder
    model_paths = [
        'models/best.pt',
        'best.pt',
        '../best.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Model file 'best.pt' not found!")
        print("Please place your 'best.pt' file in one of these locations:")
        for path in model_paths:
            print(f"  - {os.path.abspath(path)}")
        return False
    
    try:
        print(f"🔄 Loading model from {model_path}...")
        
        # Try loading with ultralytics first
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✅ Model loaded successfully with Ultralytics YOLO")
            print(f"📋 Model classes: {model.names}")
            return True
        except ImportError:
            print("⚠️ Ultralytics not available, trying torch.hub...")
        
        # Fallback to torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.eval()
        print("✅ Model loaded successfully with torch.hub")
        print(f"📋 Model classes: {model.names}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure your 'best.pt' file is a valid YOLO model")
        return False

def process_video_frames(file_data, filename):
    """Process all frames from a video file"""
    try:
        print(f"🎥 Processing video file: {filename}")
        
        # Save video data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name
        
        try:
            # Use OpenCV to read video
            cap = cv2.VideoCapture(temp_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            frames_data = []
            frame_count = 0
            
            # Process every 5th frame to reduce processing time
            frame_skip = 5
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    image = Image.fromarray(frame_rgb)
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps if fps > 0 else 0
                    
                    frames_data.append({
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'image': image
                    })
                
                frame_count += 1
            
            cap.release()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            print(f"✅ Extracted {len(frames_data)} frames for processing")
            return frames_data, duration
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
            
    except Exception as e:
        raise Exception(f"Error processing video file: {e}")

def process_image_file(file_data, filename):
    """Process a single image file"""
    try:
        print(f"📸 Processing image file: {filename}")
        image = Image.open(io.BytesIO(file_data))
        # Convert to RGB if needed (handles WEBP, PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return [{
            'frame_number': 0,
            'timestamp': 0,
            'image': image
        }], 0
    except Exception as e:
        raise Exception(f"Error processing image file: {e}")

def detect_emergency_vehicle(image):
    """
    Detect emergency vehicle in the given image
    Returns: (detected: bool, confidence: float, detections: list, message: str)
    """
    if model is None:
        return False, 0.0, [], "Model not loaded"
    
    try:
        # Run inference
        results = model(image)
        
        # Handle different result formats
        if hasattr(results, 'pandas'):
            # YOLOv5 format
            detections = results.pandas().xyxy[0]
        else:
            # Ultralytics YOLO format
            detections = results[0].boxes
            if detections is not None:
                # Convert to pandas-like format
                boxes = detections.xyxy.cpu().numpy()
                confs = detections.conf.cpu().numpy()
                classes = detections.cls.cpu().numpy()
                
                # Create a simple dataframe-like structure
                detections = pd.DataFrame({
                    'xmin': boxes[:, 0],
                    'ymin': boxes[:, 1],
                    'xmax': boxes[:, 2],
                    'ymax': boxes[:, 3],
                    'confidence': confs,
                    'class': classes,
                    'name': [model.names[int(c)] for c in classes]
                })
            else:
                detections = pd.DataFrame()
        
        print(f"📊 Found {len(detections)} total detections")
        
        # Print all detections for debugging
        if len(detections) > 0:
            print("🔍 All detected objects:")
            for idx, row in detections.iterrows():
                print(f"  - {row['name']}: {row['confidence']:.3f}")
        
        # Get all available class names from the model
        available_classes = list(model.names.values()) if hasattr(model, 'names') else []
        print(f"📋 Available model classes: {available_classes}")
        
        # Check for emergency vehicle detections with broader matching
        emergency_classes = ['ambulance', 'emergency', 'emergency_vehicle', 'medical', 'fire_truck', 'police']
        
        emergency_detections = pd.DataFrame()
        
        # First, try exact matching
        for class_name in emergency_classes:
            if 'name' in detections.columns:
                class_detections = detections[detections['name'].str.lower() == class_name.lower()]
                if len(class_detections) > 0:
                    emergency_detections = pd.concat([emergency_detections, class_detections])
                    print(f"✅ Found exact match for '{class_name}'")
        
        # If no exact matches, try partial matching
        if len(emergency_detections) == 0 and len(detections) > 0:
            print("🔍 No exact matches found, trying partial matching...")
            for idx, row in detections.iterrows():
                detected_name = str(row['name']).lower()
                for emergency_class in emergency_classes:
                    if emergency_class in detected_name or detected_name in emergency_class:
                        emergency_detections = pd.concat([emergency_detections, pd.DataFrame([row])])
                        print(f"✅ Found partial match: '{detected_name}' matches '{emergency_class}'")
                        break
        
        # If still no matches, check if any detection has high confidence (might be misclassified)
        if len(emergency_detections) == 0 and len(detections) > 0:
            print("🔍 No emergency vehicle matches found, checking high-confidence detections...")
            high_conf_detections = detections[detections['confidence'] > 0.5]
            if len(high_conf_detections) > 0:
                print("⚠️ High confidence detections found but no emergency vehicles:")
                for idx, row in high_conf_detections.iterrows():
                    print(f"  - {row['name']}: {row['confidence']:.3f}")
        
        if len(emergency_detections) > 0:
            max_confidence = emergency_detections['confidence'].max()
            detected_classes = emergency_detections['name'].unique().tolist()
            print(f"🚨 Emergency vehicle detected: {detected_classes} with max confidence: {max_confidence:.2f}")
            return True, float(max_confidence), emergency_detections.to_dict('records'), f"Emergency vehicle detected: {', '.join(detected_classes)}"
        else:
            print("✅ No emergency vehicle detected")
            return False, 0.0, detections.to_dict('records') if len(detections) > 0 else [], "No emergency vehicle detected"
            
    except Exception as e:
        error_msg = f"Detection error: {e}"
        print(f"❌ {error_msg}")
        return False, 0.0, [], error_msg

@app.route('/detect-continuous', methods=['POST'])
def detect_continuous():
    try:
        print("\n" + "="*50)
        print("🔍 New continuous detection request received")
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"📁 Processing file: {file.filename}")
        
        # Read file data
        file_data = file.read()
        
        # Check file extension
        file_ext = file.filename.lower().split('.')[-1]
        
        if file_ext in ['mp4', 'avi', 'mov', 'mkv']:
            # Process video
            frames_data, duration = process_video_frames(file_data, file.filename)
        else:
            # Process image
            frames_data, duration = process_image_file(file_data, file.filename)
        
        # Process all frames and detect emergency vehicles
        detection_results = []
        emergency_detected_frames = []
        all_detections = []
        
        print(f"🔍 Starting detection on {len(frames_data)} frames...")
        
        for i, frame_data in enumerate(frames_data):
            detected, confidence, detections, message = detect_emergency_vehicle(frame_data['image'])
            
            result = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'detected': detected,
                'confidence': confidence,
                'detections': detections,
                'message': message
            }
            
            detection_results.append(result)
            all_detections.extend(detections)
            
            if detected:
                emergency_detected_frames.append({
                    'timestamp': frame_data['timestamp'],
                    'confidence': confidence
                })
                print(f"🚨 Emergency vehicle detected at {frame_data['timestamp']:.2f}s (confidence: {confidence:.2f})")
        
        # Calculate overall statistics
        total_detections = len(emergency_detected_frames)
        max_confidence = max([f['confidence'] for f in emergency_detected_frames]) if emergency_detected_frames else 0
        
        overall_result = {
            'file_type': 'video' if file_ext in ['mp4', 'avi', 'mov', 'mkv'] else 'image',
            'duration': duration,
            'total_frames_processed': len(frames_data),
            'emergency_detected': total_detections > 0,
            'total_detections': total_detections,
            'max_confidence': max_confidence,
            'detection_frames': emergency_detected_frames,
            'detailed_results': detection_results,
            'all_detections': all_detections,  # Include all detections for debugging
            'message': f"Emergency vehicle detected in {total_detections} frames" if total_detections > 0 else "No emergency vehicles detected"
        }
        
        print(f"📤 Processing complete: {overall_result['message']}")
        return jsonify(overall_result)
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/detect', methods=['POST'])
def detect():
    try:
        print("\n" + "="*50)
        print("🔍 New detection request received")
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        print(f"📁 Processing file: {file.filename}")
        
        # Read file data
        file_data = file.read()
        
        # Check file extension
        file_ext = file.filename.lower().split('.')[-1]

        if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
            # Handle image files
            image = Image.open(io.BytesIO(file_data))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
            # Handle video files - extract first frame
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    raise Exception("Could not read video frame")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                os.unlink(temp_path)
                
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
        else:
            raise Exception(f"Unsupported file format: {file_ext}")
        
        print(f"🖼️ Processed media size: {image.size}")
        
        # Detect emergency vehicle
        detected, confidence, detections, message = detect_emergency_vehicle(image)
        
        result = {
            'detected': detected,
            'confidence': confidence,
            'detections': detections,
            'message': message,
            'total_detections': len(detections)
        }
        
        print(f"📤 Sending result: {message}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health():
    model_status = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'message': 'Model loaded successfully' if model_status else 'Model not loaded'
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        info = {
            'model_type': type(model).__name__,
            'classes': getattr(model, 'names', 'Unknown'),
            'model_loaded': True
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Emergency Vehicle Detection Server")
    print("=" * 50)
    
    if not load_model():
        print("❌ Failed to load model. Server will not start.")
        sys.exit(1)
    
    print("\n🌐 Server starting on http://localhost:5000")
    print("📋 Available endpoints:")
    print("  - POST /detect - Detect emergency vehicle in single frame")
    print("  - POST /detect-continuous - Process entire video continuously")
    print("  - GET /health - Check server health")
    print("  - GET /model-info - Get model information")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
