from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'models/best.pt'  # Place your best.pt file here
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Traffic light states
TRAFFIC_LIGHT_STATES = {
    'RED': 'red',
    'YELLOW': 'yellow',
    'GREEN': 'green'
}

class TrafficLightController:
    def __init__(self):
        self.current_state = TRAFFIC_LIGHT_STATES['RED']
        self.emergency_mode = False
        self.state_start_time = time.time()
        self.lock = threading.Lock()
        self.normal_cycle_thread = None
        self.emergency_thread = None
        self.start_normal_cycle()
    
    def start_normal_cycle(self):
        """Start the normal traffic light cycle"""
        if self.normal_cycle_thread and self.normal_cycle_thread.is_alive():
            return
        
        self.normal_cycle_thread = threading.Thread(target=self._normal_cycle_loop, daemon=True)
        self.normal_cycle_thread.start()
    
    def _normal_cycle_loop(self):
        """Normal traffic light cycle: RED(30s) -> YELLOW(3s) -> GREEN(30s) -> YELLOW(3s) -> RED..."""
        while True:
            if not self.emergency_mode:
                with self.lock:
                    # Red for 30 seconds
                    self.current_state = TRAFFIC_LIGHT_STATES['RED']
                    self.state_start_time = time.time()
                
                time.sleep(30)
                
                if not self.emergency_mode:
                    with self.lock:
                        # Yellow for 3 seconds (Red to Green transition)
                        self.current_state = TRAFFIC_LIGHT_STATES['YELLOW']
                        self.state_start_time = time.time()
                    
                    time.sleep(3)
                    
                    if not self.emergency_mode:
                        with self.lock:
                            # Green for 30 seconds
                            self.current_state = TRAFFIC_LIGHT_STATES['GREEN']
                            self.state_start_time = time.time()
                        
                        time.sleep(30)
                        
                        if not self.emergency_mode:
                            with self.lock:
                                # Yellow for 3 seconds (Green to Red transition)
                                self.current_state = TRAFFIC_LIGHT_STATES['YELLOW']
                                self.state_start_time = time.time()
                            
                            time.sleep(3)
            else:
                time.sleep(1)  # Check every second if emergency mode is still active
    
    def trigger_emergency(self):
        """Trigger emergency mode - change to green immediately for 15 seconds"""
        with self.lock:
            self.emergency_mode = True
            self.current_state = TRAFFIC_LIGHT_STATES['GREEN']
            self.state_start_time = time.time()
        
        # Start emergency timer
        if self.emergency_thread and self.emergency_thread.is_alive():
            return
        
        self.emergency_thread = threading.Thread(target=self._emergency_timer, daemon=True)
        self.emergency_thread.start()
    
    def _emergency_timer(self):
        """Emergency timer - keep green for 15 seconds then return to normal"""
        time.sleep(15)
        with self.lock:
            self.emergency_mode = False
            # The normal cycle will resume automatically
    
    def get_current_state(self):
        """Get current traffic light state and timing info"""
        with self.lock:
            elapsed_time = time.time() - self.state_start_time
            return {
                'state': self.current_state,
                'emergency_mode': self.emergency_mode,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }

# Initialize traffic light controller
traffic_controller = TrafficLightController()

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def detect_ambulance_in_image(image_path):
    """Detect ambulance in a single image"""
    if model is None:
        return False, None
    
    try:
        results = model(image_path)
        
        # Check if any detection is an ambulance
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id].lower()
                    confidence = float(box.conf.item())
                    
                    # For your 2-class model: 0=Ambulance, 1=Others
                    # Check if class_id is 0 (Ambulance) or class name contains 'ambulance'
                    if (class_id == 0 or 'ambulance' in class_name) and confidence > 0.5:
                        print(f"Ambulance detected with confidence: {confidence:.2f}")
                        return True, results
        
        return False, results
    except Exception as e:
        print(f"Error in detection: {e}")
        return False, None

def detect_ambulance_in_video(video_path):
    """Detect ambulance in video frames"""
    if model is None:
        return False, None
    
    try:
        # Process video with YOLO
        results = model(video_path)
        
        # Check each frame for ambulance
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id].lower()
                    confidence = float(box.conf.item())
                    
                    # For your 2-class model: 0=Ambulance, 1=Others
                    # Check if class_id is 0 (Ambulance) or class name contains 'ambulance'
                    if (class_id == 0 or 'ambulance' in class_name) and confidence > 0.5:
                        print(f"Ambulance detected in video with confidence: {confidence:.2f}")
                        return True, results
        
        return False, results
    except Exception as e:
        print(f"Error in video detection: {e}")
        return False, None

@app.route('/')
def index():
    return jsonify({
        "message": "Emergency Vehicle Detection API",
        "endpoints": {
            "/upload": "POST - Upload image/video for detection",
            "/traffic-status": "GET - Get current traffic light status",
            "/health": "GET - API health check"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/traffic-status')
def traffic_status():
    """Get current traffic light status"""
    status = traffic_controller.get_current_state()
    return jsonify(status)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    
    try:
        # Detect ambulance
        if is_video_file(filename):
            ambulance_detected, results = detect_ambulance_in_video(file_path)
        else:
            ambulance_detected, results = detect_ambulance_in_image(file_path)
        
        # Get detailed detection info
        detection_details = []
        total_detections = 0
        ambulance_count = 0
        
        if results:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_name = model.names[class_id]
                        confidence = float(box.conf.item())
                        
                        detection_details.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'is_ambulance': class_id == 0 or 'ambulance' in class_name.lower()
                        })
                        
                        total_detections += 1
                        if class_id == 0 or 'ambulance' in class_name.lower():
                            ambulance_count += 1
        
        # Get current traffic state before any changes
        current_traffic_state = traffic_controller.get_current_state()
        
        # Trigger emergency if ambulance detected
        if ambulance_detected:
            traffic_controller.trigger_emergency()
            action_taken = f"Emergency mode activated - {ambulance_count} ambulance(s) detected"
        else:
            action_taken = "No ambulance detected - Normal traffic cycle continues"
        
        # Get updated traffic state
        updated_traffic_state = traffic_controller.get_current_state()
        
        # Save results if available
        result_filename = None
        if results:
            try:
                result_filename = f"result_{unique_filename}"
                result_path = os.path.join(RESULTS_FOLDER, result_filename)
                
                if is_video_file(filename):
                    # For video, save the first frame with detections
                    if len(results) > 0:
                        results[0].save(result_path.replace('.mp4', '.jpg').replace('.avi', '.jpg').replace('.mov', '.jpg').replace('.mkv', '.jpg'))
                else:
                    # For image, save the result
                    results[0].save(result_path)
            except Exception as e:
                print(f"Error saving results: {e}")
        
        response = {
            'success': True,
            'filename': unique_filename,
            'ambulance_detected': ambulance_detected,
            'ambulance_count': ambulance_count,
            'total_detections': total_detections,
            'detection_details': detection_details,
            'action_taken': action_taken,
            'traffic_state_before': current_traffic_state,
            'traffic_state_after': updated_traffic_state,
            'result_filename': result_filename,
            'model_classes': model.names if model else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify(response)
    
    except Exception as e:
        # Clean up uploaded file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'success': False
        }), 500

@app.route('/results/<filename>')
def get_result_file(filename):
    """Serve result files"""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/reset-traffic', methods=['POST'])
def reset_traffic():
    """Reset traffic light to normal cycle (for testing purposes)"""
    global traffic_controller
    traffic_controller = TrafficLightController()
    return jsonify({
        'message': 'Traffic light reset to normal cycle',
        'current_state': traffic_controller.get_current_state()
    })

if __name__ == '__main__':
    print("Starting Emergency Vehicle Detection Server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("Place your best.pt file in the 'models' directory")
    print("Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)