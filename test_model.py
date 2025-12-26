#!/usr/bin/env python3
"""
Model Testing Script for Emergency Vehicle Detection
This script helps you test your YOLO model before integrating with the Flask backend
"""

import os
import cv2
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def test_model(model_path, test_image_path=None, confidence_threshold=0.5):
    """Test the YOLO model with a sample image"""
    
    print(f"Testing model: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load model
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        
        # Print model information
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
        # Verify it's a 2-class model
        if len(model.names) == 2:
            print("✅ Confirmed: 2-class model detected")
            class_0 = model.names[0].lower()
            class_1 = model.names[1].lower()
            print(f"Class 0: {model.names[0]}")
            print(f"Class 1: {model.names[1]}")
            
            # Check if ambulance is class 0
            if 'ambulance' in class_0:
                print("✅ Ambulance is correctly mapped to class 0")
            else:
                print("⚠️  Warning: Ambulance might not be class 0")
                
        else:
            print(f"⚠️  Warning: Expected 2 classes, found {len(model.names)}")
        
        # Test with image if provided
        if test_image_path:
            if not os.path.exists(test_image_path):
                print(f"❌ Test image not found: {test_image_path}")
                return False
            
            print(f"\nTesting with image: {test_image_path}")
            
            # Run inference
            results = model(test_image_path)
            
            # Analyze results
            ambulance_detected = False
            total_detections = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_name = model.names[class_id]
                        confidence = float(box.conf.item())
                        
                        print(f"Detection: {class_name} (ID: {class_id}) - Confidence: {confidence:.3f}")
                        
                        total_detections += 1
                        
                        # Check for ambulance
                        if (class_id == 0 or 'ambulance' in class_name.lower()) and confidence > confidence_threshold:
                            ambulance_detected = True
                            print(f"✅ AMBULANCE DETECTED! Confidence: {confidence:.3f}")
            
            print(f"\nSummary:")
            print(f"Total detections: {total_detections}")
            print(f"Ambulance detected: {ambulance_detected}")
            
            # Save result image
            if results:
                output_path = f"test_result_{Path(test_image_path).stem}.jpg"
                results[0].save(output_path)
                print(f"Result saved to: {output_path}")
            
            return ambulance_detected
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def create_test_structure():
    """Create the required directory structure"""
    directories = ['models', 'uploads', 'results', 'test_images']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model for ambulance detection')
    parser.add_argument('--model', '-m', default='models/best.pt', help='Path to YOLO model file')
    parser.add_argument('--image', '-i', help='Path to test image')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--setup', action='store_true', help='Create directory structure')
    
    args = parser.parse_args()
    
    if args.setup:
        print("Setting up directory structure...")
        create_test_structure()
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Place your best.pt file in the 'models' directory")
        print("2. Place test images in the 'test_images' directory")
        print("3. Run: python test_model.py --image test_images/your_image.jpg")
        return
    
    print("🚨 Emergency Vehicle Detection - Model Test")
    print("=" * 50)
    
    # Test the model
    success = test_model(args.model, args.image, args.confidence)
    
    if success:
        print("\n✅ Model test completed successfully!")
        print("\nYour model is ready for integration with the Flask backend.")
    else:
        print("\n❌ Model test failed!")
        print("\nPlease check:")
        print("1. Model file exists at the specified path")
        print("2. Model is trained for ambulance detection")
        print("3. Test image exists (if provided)")

if __name__ == "__main__":
    main()