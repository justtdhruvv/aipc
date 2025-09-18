#!/usr/bin/env python3
"""
Advanced AI Feature Extractor for Photo Similarity
==================================================
Uses state-of-the-art deep learning models to extract semantic features from images
for superior photo similarity detection and organization.

Features:
- CLIP embeddings for semantic understanding
- ResNet50 features for visual patterns
- Face recognition for people detection
- YOLO object detection for content analysis
- Combined feature vectors for comprehensive similarity
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class AIFeatureExtractor:
    """Advanced AI-powered feature extraction for images"""
    
    def __init__(self):
        """Initialize all AI models"""
        print("🤖 Initializing AI models...")
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # 1. CLIP model for semantic understanding
        try:
            print("📚 Loading CLIP model...")
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"❌ CLIP loading failed: {e}")
            self.clip_model = None
        
        # 2. ResNet50 for visual features
        try:
            print("🧠 Loading ResNet50 model...")
            self.resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet_model.eval()
            self.resnet_model = self.resnet_model.to(self.device)
            
            # Remove the final classification layer to get features
            self.resnet_features = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
            
            # Image preprocessing for ResNet
            self.resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ ResNet50 loaded successfully")
        except Exception as e:
            print(f"❌ ResNet50 loading failed: {e}")
            self.resnet_model = None
            
        # 3. YOLO for object detection
        try:
            print("🎯 Loading YOLO model...")
            self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
            print("✅ YOLO loaded successfully")
        except Exception as e:
            print(f"❌ YOLO loading failed: {e}")
            self.yolo_model = None
            
        print("🚀 AI Feature Extractor initialized!\n")
    
    def extract_clip_features(self, image_path):
        """Extract CLIP embeddings for semantic understanding"""
        try:
            if self.clip_model is None:
                return None
            
            # Load and encode image
            image = Image.open(image_path).convert('RGB')
            features = self.clip_model.encode([image])
            return features[0]  # Return the embedding vector
            
        except Exception as e:
            print(f"❌ CLIP feature extraction failed for {image_path}: {e}")
            return None
    
    def extract_resnet_features(self, image_path):
        """Extract ResNet50 deep features"""
        try:
            if self.resnet_model is None:
                return None
                
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.resnet_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet_features(input_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"❌ ResNet feature extraction failed for {image_path}: {e}")
            return None
    
    def extract_face_features(self, image_path):
        """Extract face encodings for people detection"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return {
                'face_count': len(face_locations),
                'face_encodings': face_encodings,
                'face_locations': face_locations
            }
            
        except Exception as e:
            print(f"❌ Face detection failed for {image_path}: {e}")
            return {'face_count': 0, 'face_encodings': [], 'face_locations': []}
    
    def extract_object_features(self, image_path):
        """Extract object detection features using YOLO"""
        try:
            if self.yolo_model is None:
                return None
                
            # Run YOLO detection
            results = self.yolo_model(image_path, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
            
            return {
                'object_count': len(objects),
                'objects': objects,
                'primary_objects': [obj['class'] for obj in objects if obj['confidence'] > 0.5]
            }
            
        except Exception as e:
            print(f"❌ Object detection failed for {image_path}: {e}")
            return {'object_count': 0, 'objects': [], 'primary_objects': []}
    
    def extract_comprehensive_features(self, image_path):
        """Extract all AI features from an image"""
        print(f"🔍 Extracting AI features from: {image_path}")
        
        features = {
            'image_path': image_path,
            'clip_features': None,
            'resnet_features': None,
            'face_features': None,
            'object_features': None
        }
        
        # Extract CLIP semantic features
        features['clip_features'] = self.extract_clip_features(image_path)
        
        # Extract ResNet visual features
        features['resnet_features'] = self.extract_resnet_features(image_path)
        
        # Extract face features
        features['face_features'] = self.extract_face_features(image_path)
        
        # Extract object features
        features['object_features'] = self.extract_object_features(image_path)
        
        return features
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        similarities = {}
        
        # CLIP semantic similarity
        if features1['clip_features'] is not None and features2['clip_features'] is not None:
            clip_sim = np.dot(features1['clip_features'], features2['clip_features']) / (
                np.linalg.norm(features1['clip_features']) * np.linalg.norm(features2['clip_features'])
            )
            similarities['clip_similarity'] = float(clip_sim)
        else:
            similarities['clip_similarity'] = 0.0
        
        # ResNet visual similarity
        if features1['resnet_features'] is not None and features2['resnet_features'] is not None:
            resnet_sim = np.dot(features1['resnet_features'], features2['resnet_features']) / (
                np.linalg.norm(features1['resnet_features']) * np.linalg.norm(features2['resnet_features'])
            )
            similarities['resnet_similarity'] = float(resnet_sim)
        else:
            similarities['resnet_similarity'] = 0.0
        
        # Face similarity
        face1 = features1['face_features']
        face2 = features2['face_features']
        if face1['face_count'] > 0 and face2['face_count'] > 0:
            # Compare face encodings
            max_face_similarity = 0.0
            for enc1 in face1['face_encodings']:
                for enc2 in face2['face_encodings']:
                    face_distance = face_recognition.face_distance([enc1], enc2)[0]
                    face_similarity = 1.0 - face_distance  # Convert distance to similarity
                    max_face_similarity = max(max_face_similarity, face_similarity)
            similarities['face_similarity'] = float(max_face_similarity)
        else:
            similarities['face_similarity'] = 0.0
        
        # Object similarity
        obj1 = features1['object_features']
        obj2 = features2['object_features']
        if obj1 and obj2:
            common_objects = set(obj1['primary_objects']) & set(obj2['primary_objects'])
            total_objects = set(obj1['primary_objects']) | set(obj2['primary_objects'])
            object_sim = len(common_objects) / len(total_objects) if total_objects else 0.0
            similarities['object_similarity'] = float(object_sim)
        else:
            similarities['object_similarity'] = 0.0
        
        # Comprehensive weighted similarity
        weights = {
            'clip': 0.4,      # Semantic understanding - highest weight
            'resnet': 0.3,    # Visual features - high weight
            'face': 0.2,      # Face similarity - medium weight
            'object': 0.1     # Object similarity - lowest weight
        }
        
        comprehensive_score = (
            weights['clip'] * similarities['clip_similarity'] +
            weights['resnet'] * similarities['resnet_similarity'] +
            weights['face'] * similarities['face_similarity'] +
            weights['object'] * similarities['object_similarity']
        )
        
        similarities['comprehensive_similarity'] = float(comprehensive_score)
        
        return similarities

if __name__ == "__main__":
    # Test the feature extractor
    extractor = AIFeatureExtractor()
    
    # Test on sample images
    import os
    photos_dir = "../photos_to_scan"  # Go up one level
    
    if os.path.exists(photos_dir):
        image_files = [f for f in os.listdir(photos_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) >= 2:
            image1 = os.path.join(photos_dir, image_files[0])
            image2 = os.path.join(photos_dir, image_files[1])
            
            print("🧪 Testing AI feature extraction...")
            features1 = extractor.extract_comprehensive_features(image1)
            features2 = extractor.extract_comprehensive_features(image2)
            
            similarities = extractor.calculate_feature_similarity(features1, features2)
            
            print(f"\n📊 Similarity Analysis:")
            print(f"🎯 CLIP Semantic: {similarities['clip_similarity']:.3f}")
            print(f"🧠 ResNet Visual: {similarities['resnet_similarity']:.3f}")
            print(f"👤 Face Similarity: {similarities['face_similarity']:.3f}")
            print(f"📦 Object Similarity: {similarities['object_similarity']:.3f}")
            print(f"🎯 Comprehensive Score: {similarities['comprehensive_similarity']:.3f}")
        else:
            print("❌ Need at least 2 images in photos_to_scan directory for testing")
    else:
        print("❌ photos_to_scan directory not found")