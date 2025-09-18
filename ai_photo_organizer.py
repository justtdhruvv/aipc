#!/usr/bin/env python3
"""
AI-Powered Intelligent Photo Organizer
=====================================
Next-generation photo organization using advanced AI and machine learning.

Features:
- CLIP semantic understanding for concept-based grouping
- ResNet50 visual feature matching for appearance similarity
- Face recognition for people-based organization
- YOLO object detection for content-based categorization
- Hybrid scoring combining AI with traditional perceptual hashing
- Smart thresholds and intelligent grouping algorithms

This is the most advanced photo organizer, using state-of-the-art AI models
for superior accuracy and semantic understanding.
"""

import os
import sys
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_feature_extractor import AIFeatureExtractor
import imagehash
from PIL import Image

class AIPhotoOrganizer:
    """Advanced AI-powered photo organization system"""
    
    def __init__(self):
        """Initialize the AI organizer"""
        self.base_dir = Path(__file__).parent
        self.photos_dir = self.base_dir / "photos_to_scan"
        self.organized_dir = self.base_dir / "organized_photos"
        self.db_path = self.base_dir / "ai_organizer.db"
        
        # Initialize AI feature extractor
        print("🤖 Initializing AI Photo Organizer...")
        self.ai_extractor = AIFeatureExtractor()
        
        # Create organized directory structure
        self.setup_organized_directories()
        
        # Initialize database
        self.init_database()
        
        print("✅ AI Photo Organizer ready!\n")
    
    def setup_organized_directories(self):
        """Create organized directory structure"""
        categories = [
            "1_FACES_PEOPLE",
            "2_LANDSCAPES_NATURE", 
            "3_OBJECTS_ITEMS",
            "4_AI_SIMILAR_SEMANTIC",
            "5_AI_SIMILAR_VISUAL",
            "6_TRADITIONAL_SIMILAR",
            "7_MIXED_GROUPS",
            "8_UNIQUE_AI"
        ]
        
        for category in categories:
            (self.organized_dir / category).mkdir(parents=True, exist_ok=True)
    
    def init_database(self):
        """Initialize AI organizer database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create AI features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_features (
                file_path TEXT PRIMARY KEY,
                file_size INTEGER,
                image_hash TEXT,
                clip_features TEXT,
                resnet_features TEXT,
                face_count INTEGER,
                face_encodings TEXT,
                detected_objects TEXT,
                primary_category TEXT,
                processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create similarity groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_similarity_groups (
                group_id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_type TEXT,
                similarity_score REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create group members table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_group_members (
                group_id INTEGER,
                file_path TEXT,
                FOREIGN KEY (group_id) REFERENCES ai_similarity_groups (group_id),
                FOREIGN KEY (file_path) REFERENCES ai_features (file_path)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_traditional_features(self, image_path):
        """Extract traditional perceptual hashes"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate various hashes
                p_hash = str(imagehash.phash(img))
                d_hash = str(imagehash.dhash(img))
                a_hash = str(imagehash.average_hash(img))
                w_hash = str(imagehash.whash(img))
                
                return {
                    'phash': p_hash,
                    'dhash': d_hash,
                    'ahash': a_hash,
                    'whash': w_hash
                }
        except Exception as e:
            print(f"❌ Traditional feature extraction failed for {image_path}: {e}")
            return None
    
    def process_all_images(self):
        """Extract AI and traditional features from all images"""
        if not self.photos_dir.exists():
            print(f"❌ Photos directory not found: {self.photos_dir}")
            return
        
        image_files = [f for f in self.photos_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_files:
            print("❌ No image files found in photos_to_scan directory")
            return
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, image_path in enumerate(image_files):
            print(f"📸 Processing ({i+1}/{len(image_files)}): {image_path.name}")
            
            # Check if already processed
            cursor.execute('SELECT file_path FROM ai_features WHERE file_path = ?', (str(image_path),))
            if cursor.fetchone():
                print(f"   ⏭️  Already processed, skipping...")
                continue
            
            # Extract AI features
            ai_features = self.ai_extractor.extract_comprehensive_features(str(image_path))
            
            # Extract traditional features
            traditional_features = self.extract_traditional_features(image_path)
            
            # Store in database
            self.store_image_features(cursor, image_path, ai_features, traditional_features)
        
        conn.commit()
        conn.close()
        
        print("✅ Feature extraction completed!")
    
    def store_image_features(self, cursor, image_path, ai_features, traditional_features):
        """Store extracted features in database"""
        try:
            # Get file info
            file_size = image_path.stat().st_size
            
            # Prepare AI features for storage
            clip_features = json.dumps(ai_features['clip_features'].tolist()) if ai_features['clip_features'] is not None else None
            resnet_features = json.dumps(ai_features['resnet_features'].tolist()) if ai_features['resnet_features'] is not None else None
            
            face_count = ai_features['face_features']['face_count']
            face_encodings = json.dumps([enc.tolist() for enc in ai_features['face_features']['face_encodings']]) if ai_features['face_features']['face_encodings'] else None
            
            detected_objects = json.dumps(ai_features['object_features']) if ai_features['object_features'] else None
            
            # Determine primary category
            primary_category = self.determine_primary_category(ai_features)
            
            # Traditional hash (use perceptual hash)
            image_hash = traditional_features['phash'] if traditional_features else None
            
            cursor.execute('''
                INSERT INTO ai_features 
                (file_path, file_size, image_hash, clip_features, resnet_features, 
                 face_count, face_encodings, detected_objects, primary_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(image_path), file_size, image_hash, clip_features, resnet_features,
                face_count, face_encodings, detected_objects, primary_category
            ))
            
        except Exception as e:
            print(f"❌ Database storage failed for {image_path}: {e}")
    
    def determine_primary_category(self, ai_features):
        """Determine primary category based on AI analysis"""
        face_count = ai_features['face_features']['face_count']
        objects = ai_features['object_features']['primary_objects'] if ai_features['object_features'] else []
        
        # Categorize based on content
        if face_count > 0:
            return "FACES_PEOPLE"
        elif any(obj in ['car', 'truck', 'bicycle', 'motorcycle', 'bus'] for obj in objects):
            return "VEHICLES"
        elif any(obj in ['tree', 'grass', 'mountain', 'sky', 'cloud'] for obj in objects):
            return "LANDSCAPES_NATURE"
        elif any(obj in ['food', 'cup', 'bottle', 'chair', 'table'] for obj in objects):
            return "OBJECTS_ITEMS"
        else:
            return "GENERAL"
    
    def find_ai_similar_groups(self):
        """Find similar images using improved hierarchical clustering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all processed images
        cursor.execute('''
            SELECT file_path, clip_features, resnet_features, face_encodings, 
                   detected_objects, primary_category
            FROM ai_features 
            WHERE clip_features IS NOT NULL OR resnet_features IS NOT NULL
        ''')
        
        images = cursor.fetchall()
        
        if len(images) < 2:
            print("❌ Need at least 2 processed images for similarity detection")
            conn.close()
            return
        
        print(f"🧮 Analyzing {len(images)} images using enhanced clustering...")
        
        # Build comprehensive similarity matrix
        print("📊 Building similarity matrix...")
        similarity_matrix = self.build_similarity_matrix(images)
        
        # Multi-pass clustering approach
        similar_groups = self.multi_pass_clustering(images, similarity_matrix)
        
        print(f"✅ Found {len(similar_groups)} similarity groups using advanced clustering")
        
        # Store and organize groups
        for group in similar_groups:
            self.store_similarity_group(group, conn)
        
        # Organize into directories
        self.organize_similar_images(similar_groups)
        
        conn.close()
    
    def build_similarity_matrix(self, images):
        """Build complete similarity matrix for all image pairs"""
        n = len(images)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self.calculate_ai_similarity_from_db(images[i], images[j])
                sim_score = similarity.get('comprehensive_similarity', 0.0)
                similarity_matrix[i][j] = sim_score
                similarity_matrix[j][i] = sim_score  # Symmetric matrix
        
        return similarity_matrix
    
    def multi_pass_clustering(self, images, similarity_matrix):
        """Enhanced multi-pass clustering to capture all similar images"""
        n = len(images)
        assigned_images = set()
        groups = []
        
        # Enhanced thresholds for different passes
        thresholds = {
            'pass1_high_quality': 0.75,    # Very confident matches
            'pass2_medium_quality': 0.60,  # Good matches
            'pass3_low_quality': 0.50,     # Potential matches
            'face_bonus': 0.55,            # Lower threshold for face matches
            'semantic_bonus': 0.65         # Lower threshold for semantic matches
        }
        
        print("🔍 Pass 1: High-confidence groups (≥75% similarity)...")
        groups.extend(self.clustering_pass(images, similarity_matrix, 
                     thresholds['pass1_high_quality'], assigned_images, "HIGH"))
        
        print("🔍 Pass 2: Medium-confidence groups (≥60% similarity)...")  
        groups.extend(self.clustering_pass(images, similarity_matrix,
                     thresholds['pass2_medium_quality'], assigned_images, "MEDIUM"))
        
        print("🔍 Pass 3: Potential similar groups (≥50% similarity)...")
        groups.extend(self.clustering_pass(images, similarity_matrix,
                     thresholds['pass3_low_quality'], assigned_images, "LOW"))
        
        # Special pass for face matches with lower threshold
        print("🔍 Pass 4: Face similarity groups...")
        groups.extend(self.face_similarity_pass(images, assigned_images))
        
        return groups
    
    def clustering_pass(self, images, similarity_matrix, threshold, assigned_images, quality_level):
        """Single clustering pass with given threshold"""
        n = len(images)
        groups = []
        
        for i in range(n):
            if i in assigned_images:
                continue
                
            # Find all images similar to image i above threshold
            similar_indices = []
            for j in range(n):
                if (j != i and j not in assigned_images and 
                    similarity_matrix[i][j] >= threshold):
                    similar_indices.append(j)
            
            # If we found similar images, create a group
            if similar_indices:
                group_indices = [i] + similar_indices
                group_images = [images[idx] for idx in group_indices]
                
                # Calculate average similarity for this group
                avg_sim = self.calculate_group_similarity_from_matrix(
                    group_indices, similarity_matrix)
                
                # Determine group type and create group
                group_type = self.classify_group_type(group_images)
                
                group = {
                    'images': group_images,
                    'type': group_type,
                    'similarity': avg_sim,
                    'size': len(group_images),
                    'quality': quality_level
                }
                
                groups.append(group)
                
                # Mark all images in this group as assigned
                for idx in group_indices:
                    assigned_images.add(idx)
                
                print(f"   📁 Found {group_type} group: {len(group_images)} photos, {avg_sim:.3f} similarity")
        
        return groups
    
    def face_similarity_pass(self, images, assigned_images):
        """Special pass for face similarity with adaptive thresholds"""
        groups = []
        n = len(images)
        
        for i in range(n):
            if i in assigned_images:
                continue
                
            # Only process images with faces
            face_encodings_i = images[i][3]  # face_encodings column
            if not face_encodings_i:
                continue
                
            try:
                face_list_i = json.loads(face_encodings_i) if face_encodings_i else []
                if not face_list_i:
                    continue
                    
                similar_faces = [i]
                
                for j in range(n):
                    if j == i or j in assigned_images:
                        continue
                        
                    face_encodings_j = images[j][3]
                    if not face_encodings_j:
                        continue
                        
                    face_list_j = json.loads(face_encodings_j) if face_encodings_j else []
                    if not face_list_j:
                        continue
                    
                    # Calculate face similarity
                    max_face_sim = 0.0
                    for face1 in face_list_i:
                        for face2 in face_list_j:
                            face1_arr = np.array(face1)
                            face2_arr = np.array(face2)
                            distance = np.linalg.norm(face1_arr - face2_arr)
                            similarity = 1.0 - min(distance, 1.0)
                            max_face_sim = max(max_face_sim, similarity)
                    
                    # Lower threshold for face matches
                    if max_face_sim >= 0.55:
                        similar_faces.append(j)
                
                if len(similar_faces) > 1:
                    group_images = [images[idx] for idx in similar_faces]
                    avg_face_sim = self.calculate_face_group_similarity(group_images)
                    
                    group = {
                        'images': group_images,
                        'type': 'FACES_PEOPLE',
                        'similarity': avg_face_sim,
                        'size': len(group_images),
                        'quality': 'FACE_MATCH'
                    }
                    
                    groups.append(group)
                    
                    for idx in similar_faces:
                        assigned_images.add(idx)
                    
                    print(f"   👥 Found face group: {len(group_images)} photos, {avg_face_sim:.3f} face similarity")
                    
            except Exception as e:
                print(f"❌ Face similarity error: {e}")
                continue
        
        return groups
    
    def calculate_group_similarity_from_matrix(self, group_indices, similarity_matrix):
        """Calculate average similarity for a group using similarity matrix"""
        if len(group_indices) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(group_indices)):
            for j in range(i+1, len(group_indices)):
                idx_i, idx_j = group_indices[i], group_indices[j]
                total_similarity += similarity_matrix[idx_i][idx_j]
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def calculate_face_group_similarity(self, group_images):
        """Calculate average face similarity for a group"""
        if len(group_images) < 2:
            return 0.0
            
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(group_images)):
            for j in range(i+1, len(group_images)):
                try:
                    face_enc_i = json.loads(group_images[i][3]) if group_images[i][3] else []
                    face_enc_j = json.loads(group_images[j][3]) if group_images[j][3] else []
                    
                    max_face_sim = 0.0
                    for face1 in face_enc_i:
                        for face2 in face_enc_j:
                            face1_arr = np.array(face1)
                            face2_arr = np.array(face2)
                            distance = np.linalg.norm(face1_arr - face2_arr)
                            similarity = 1.0 - min(distance, 1.0)
                            max_face_sim = max(max_face_sim, similarity)
                    
                    total_similarity += max_face_sim
                    comparisons += 1
                except:
                    continue
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def store_similarity_group(self, group, conn):
        """Store similarity group in database"""
        cursor = conn.cursor()
        
        # Insert group record
        cursor.execute('''
            INSERT INTO ai_similarity_groups (group_type, similarity_score)
            VALUES (?, ?)
        ''', (group['type'], group['similarity']))
        
        group_id = cursor.lastrowid
        
        # Insert group members if table exists
        try:
            for image_data in group['images']:
                cursor.execute('''
                    INSERT INTO ai_similarity_group_members (group_id, file_path)
                    VALUES (?, ?)
                ''', (group_id, image_data[0]))  # file_path is index 0
        except sqlite3.OperationalError:
            # Table doesn't exist, create it
            cursor.execute('''
                CREATE TABLE ai_similarity_group_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER,
                    file_path TEXT,
                    FOREIGN KEY (group_id) REFERENCES ai_similarity_groups (group_id)
                )
            ''')
            for image_data in group['images']:
                cursor.execute('''
                    INSERT INTO ai_similarity_group_members (group_id, file_path)
                    VALUES (?, ?)
                ''', (group_id, image_data[0]))
        
        conn.commit()
        return group_id
    
    def organize_similar_images(self, similar_groups):
        """Organize similar images into appropriate directories"""
        for group in similar_groups:
            group_type = group['type'] 
            similarity = group['similarity']
            quality = group.get('quality', 'MEDIUM')
            
            # Determine target directory based on type and quality
            if group_type == "FACES_PEOPLE":
                if similarity >= 0.8:
                    target_dir = self.organized_dir / "1_FACES_PEOPLE" / f"High_Similarity_{similarity:.2f}"
                else:
                    target_dir = self.organized_dir / "1_FACES_PEOPLE" / f"Group_{similarity:.2f}"
            elif group_type == "LANDSCAPES_NATURE":
                target_dir = self.organized_dir / "2_LANDSCAPES_NATURE" / f"Group_{similarity:.2f}"
            elif similarity >= 0.8:
                target_dir = self.organized_dir / "4_AI_SIMILAR_SEMANTIC" / f"High_Quality_{similarity:.2f}"
            elif similarity >= 0.6:
                target_dir = self.organized_dir / "5_AI_SIMILAR_VISUAL" / f"Medium_Quality_{similarity:.2f}"
            else:
                target_dir = self.organized_dir / "7_MIXED_GROUPS" / f"Low_Quality_{similarity:.2f}"
            
            # Create directory and copy images
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for image_data in group['images']:
                source_path = Path(image_data[0])
                if source_path.exists():
                    target_path = target_dir / source_path.name
                    if not target_path.exists():
                        shutil.copy2(source_path, target_path)
                        print(f"📁 Copied {source_path.name} to {target_dir.name}")
        
        print(f"🎯 Found {len(similar_groups)} AI similarity groups!")
        return similar_groups
    
    def calculate_ai_similarity_from_db(self, img1_data, img2_data):
        """Calculate AI similarity from database stored features with adaptive weighting"""
        try:
            # Parse stored features
            clip1 = json.loads(img1_data[1]) if img1_data[1] else None
            clip2 = json.loads(img2_data[1]) if img2_data[1] else None
            
            resnet1 = json.loads(img1_data[2]) if img1_data[2] else None
            resnet2 = json.loads(img2_data[2]) if img2_data[2] else None
            
            faces1 = json.loads(img1_data[3]) if img1_data[3] else []
            faces2 = json.loads(img2_data[3]) if img2_data[3] else []
            
            objects1 = json.loads(img1_data[4]) if img1_data[4] else []
            objects2 = json.loads(img2_data[4]) if img2_data[4] else []
            
            category1 = img1_data[5]
            category2 = img2_data[5]
            
            similarities = {}
            
            # CLIP semantic similarity
            if clip1 and clip2:
                clip1_arr = np.array(clip1)
                clip2_arr = np.array(clip2)
                clip_sim = np.dot(clip1_arr, clip2_arr) / (np.linalg.norm(clip1_arr) * np.linalg.norm(clip2_arr))
                similarities['clip_similarity'] = float(clip_sim)
            else:
                similarities['clip_similarity'] = 0.0
            
            # ResNet visual similarity
            if resnet1 and resnet2:
                resnet1_arr = np.array(resnet1)
                resnet2_arr = np.array(resnet2)
                resnet_sim = np.dot(resnet1_arr, resnet2_arr) / (np.linalg.norm(resnet1_arr) * np.linalg.norm(resnet2_arr))
                similarities['resnet_similarity'] = float(resnet_sim)
            else:
                similarities['resnet_similarity'] = 0.0
            
            # Face similarity
            if faces1 and faces2:
                max_face_sim = 0.0
                for face1 in faces1:
                    for face2 in faces2:
                        face1_arr = np.array(face1)
                        face2_arr = np.array(face2)
                        face_distance = np.linalg.norm(face1_arr - face2_arr)
                        face_sim = 1.0 - min(face_distance, 1.0)  # Convert to similarity
                        max_face_sim = max(max_face_sim, face_sim)
                similarities['face_similarity'] = float(max_face_sim)
            else:
                similarities['face_similarity'] = 0.0
            
            # Object similarity  
            if objects1 and objects2:
                common_objects = set(objects1) & set(objects2)
                total_objects = set(objects1) | set(objects2)
                object_sim = len(common_objects) / len(total_objects) if total_objects else 0.0
                similarities['object_similarity'] = float(object_sim)
            else:
                similarities['object_similarity'] = 0.0
            
            # Adaptive weighted similarity based on content
            similarities['comprehensive_similarity'] = self.calculate_adaptive_weighted_similarity(
                similarities, category1, category2, faces1, faces2, objects1, objects2
            )
            
            return similarities
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            return {'clip_similarity': 0.0, 'resnet_similarity': 0.0, 'face_similarity': 0.0, 'comprehensive_similarity': 0.0}
    
    def calculate_adaptive_weighted_similarity(self, similarities, cat1, cat2, faces1, faces2, obj1, obj2):
        """Calculate weighted similarity with adaptive weights based on content"""
        
        # Base weights
        weights = {
            'clip': 0.4,      # Semantic understanding
            'resnet': 0.3,    # Visual features  
            'face': 0.2,      # Face similarity
            'object': 0.1     # Object similarity
        }
        
        # Adaptive weighting based on content type
        if cat1 == "FACES_PEOPLE" and cat2 == "FACES_PEOPLE":
            # Both have faces - increase face weight
            weights = {'clip': 0.25, 'resnet': 0.25, 'face': 0.4, 'object': 0.1}
            
        elif cat1 == "LANDSCAPES_NATURE" and cat2 == "LANDSCAPES_NATURE":
            # Both landscapes - increase semantic and visual, decrease face
            weights = {'clip': 0.5, 'resnet': 0.35, 'face': 0.05, 'object': 0.1}
            
        elif cat1 == cat2 and cat1 in ["VEHICLES", "OBJECTS_ITEMS"]:
            # Same object category - increase object and semantic weight
            weights = {'clip': 0.4, 'resnet': 0.3, 'face': 0.1, 'object': 0.2}
            
        elif faces1 and faces2:
            # Both have faces but different categories - moderate face weight
            weights = {'clip': 0.35, 'resnet': 0.25, 'face': 0.3, 'object': 0.1}
            
        elif not faces1 and not faces2:
            # No faces in either - focus on visual and semantic
            weights = {'clip': 0.45, 'resnet': 0.4, 'face': 0.0, 'object': 0.15}
        
        # Calculate weighted score
        comprehensive_score = (
            weights['clip'] * similarities['clip_similarity'] +
            weights['resnet'] * similarities['resnet_similarity'] +
            weights['face'] * similarities['face_similarity'] +
            weights['object'] * similarities['object_similarity']
        )
        
        # Apply content-specific bonuses
        bonus = 0.0
        
        # Same category bonus (except for GENERAL)
        if cat1 == cat2 and cat1 != "GENERAL":
            bonus += 0.05
            
        # High semantic similarity bonus
        if similarities['clip_similarity'] > 0.8:
            bonus += 0.03
            
        # High face similarity bonus
        if similarities['face_similarity'] > 0.7:
            bonus += 0.04
            
        # Apply bonus but cap at 1.0
        final_score = min(comprehensive_score + bonus, 1.0)
        
        return float(final_score)
    
    def classify_group_type(self, group):
        """Classify the type of similarity group"""
        # Analyze primary categories
        categories = [img[5] for img in group]  # primary_category is index 5
        
        if all(cat == "FACES_PEOPLE" for cat in categories):
            return "FACES_PEOPLE"
        elif all(cat == "LANDSCAPES_NATURE" for cat in categories):
            return "LANDSCAPES_NATURE"
        elif len(set(categories)) == 1:
            return categories[0]
        else:
            return "MIXED_GROUPS"
    
    def calculate_group_average_similarity(self, group):
        """Calculate average similarity within a group"""
        if len(group) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                similarity = self.calculate_ai_similarity_from_db(group[i], group[j])
                total_similarity += similarity['comprehensive_similarity']
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def organize_ai_groups(self, similar_groups):
        """Organize photos based on AI similarity groups"""
        print(f"📁 Organizing {len(similar_groups)} AI similarity groups...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, group in enumerate(similar_groups, 1):
            group_type = group['type']
            similarity_score = group['similarity']
            
            # Create group in database
            cursor.execute('''
                INSERT INTO ai_similarity_groups (group_type, similarity_score)
                VALUES (?, ?)
            ''', (group_type, similarity_score))
            
            group_id = cursor.lastrowid
            
            # Determine destination folder
            if group_type == "FACES_PEOPLE":
                dest_folder = self.organized_dir / "1_FACES_PEOPLE" / f"faces_group_{i:03d}"
            elif group_type == "LANDSCAPES_NATURE":
                dest_folder = self.organized_dir / "2_LANDSCAPES_NATURE" / f"landscape_group_{i:03d}"
            elif similarity_score >= 0.80:
                dest_folder = self.organized_dir / "4_AI_SIMILAR_SEMANTIC" / f"semantic_group_{i:03d}"
            elif similarity_score >= 0.65:
                dest_folder = self.organized_dir / "5_AI_SIMILAR_VISUAL" / f"visual_group_{i:03d}"
            else:
                dest_folder = self.organized_dir / "7_MIXED_GROUPS" / f"mixed_group_{i:03d}"
            
            # Create destination folder
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"   📂 Group {i}: {group_type} ({len(group['images'])} photos, {similarity_score:.3f} similarity)")
            
            # Copy images to destination
            for img_data in group['images']:
                source_path = Path(img_data[0])
                dest_path = dest_folder / source_path.name
                
                try:
                    shutil.copy2(source_path, dest_path)
                    
                    # Add to group members
                    cursor.execute('''
                        INSERT INTO ai_group_members (group_id, file_path)
                        VALUES (?, ?)
                    ''', (group_id, str(source_path)))
                    
                    print(f"      📸 {source_path.name}")
                    
                except Exception as e:
                    print(f"      ❌ Failed to copy {source_path.name}: {e}")
        
        conn.commit()
        conn.close()
        
        print("✅ AI organization completed!")
    
    def run_ai_organization(self):
        """Run the complete AI-powered photo organization"""
        print("🚀 Starting AI-Powered Photo Organization")
        print("=" * 50)
        
        # Step 1: Process all images
        self.process_all_images()
        
        # Step 2: Find AI similarity groups
        similar_groups = self.find_ai_similar_groups()
        
        if not similar_groups:
            print("😕 No similar groups found with current thresholds")
            return
        
        # Step 3: Organize groups
        self.organize_ai_groups(similar_groups)
        
        # Step 4: Show summary
        self.show_ai_summary()
    
    def show_ai_summary(self):
        """Show AI organization summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('SELECT COUNT(*) FROM ai_features')
        total_processed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM ai_similarity_groups')
        total_groups = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM ai_group_members')
        organized_photos = cursor.fetchone()[0]
        
        cursor.execute('SELECT group_type, COUNT(*) FROM ai_similarity_groups GROUP BY group_type')
        group_types = cursor.fetchall()
        
        conn.close()
        
        print("\n🎯 AI ORGANIZATION SUMMARY")
        print("=" * 30)
        print(f"📸 Total images processed: {total_processed}")
        print(f"📁 AI groups created: {total_groups}")
        print(f"📸 Photos organized: {organized_photos}")
        print(f"📸 Unique photos: {total_processed - organized_photos}")
        
        print(f"\n📊 Group Types:")
        for group_type, count in group_types:
            print(f"   {group_type}: {count} groups")
        
        print(f"\n💡 Check organized_photos/ folder for results!")
        print(f"🧹 Run interactive_cleanup.py for manual review!")

if __name__ == "__main__":
    organizer = AIPhotoOrganizer()
    organizer.run_ai_organization()