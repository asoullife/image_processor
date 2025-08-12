"""
Similarity Finder Module for Adobe Stock Image Processor

This module implements comprehensive image similarity detection including:
- Perceptual hashing functions (pHash, dHash, aHash)
- Deep learning feature extraction using pre-trained CNN
- Clustering algorithm (DBSCAN) for grouping similar images
- Similarity scoring and threshold management
- Group recommendation logic for duplicate detection
"""

from backend.ml import runtime

probe = runtime.probe()
tf = probe.tf
cv2 = probe.cv2

try:
    import numpy as np
    from PIL import Image
    import imagehash
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Handle missing dependencies gracefully for testing
    np = None
    Image = None
    imagehash = None
    DBSCAN = None
    cosine_similarity = None
    StandardScaler = None

import os
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity analysis for a single image"""
    image_path: str
    phash: str
    dhash: str
    ahash: str
    features: Optional[Any]  # np.ndarray when available
    similarity_group: int
    similar_images: List[str]
    similarity_scores: Dict[str, float]
    is_duplicate: bool
    recommended_action: str  # 'keep', 'remove', 'review'


@dataclass
class SimilarityGroupResult:
    """Result of similarity grouping analysis"""
    total_images: int
    total_groups: int
    duplicate_groups: List[List[str]]
    similar_groups: List[List[str]]
    recommended_keeps: List[str]
    recommended_removes: List[str]
    statistics: Dict[str, Any]


class SimilarityFinder:
    """
    Comprehensive similarity detection and clustering system
    
    This class provides functionality to:
    - Compute perceptual hashes for images
    - Extract deep learning features using pre-trained CNN
    - Cluster similar images using DBSCAN
    - Provide recommendations for duplicate handling
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 hash_threshold: int = 5,
                 clustering_eps: float = 0.3,
                 min_samples: int = 2,
                 feature_model: str = 'resnet50'):
        """
        Initialize SimilarityFinder with configuration parameters
        
        Args:
            similarity_threshold: Threshold for feature similarity (0.0-1.0)
            hash_threshold: Hamming distance threshold for hash similarity
            clustering_eps: DBSCAN epsilon parameter for clustering
            min_samples: DBSCAN minimum samples parameter
            feature_model: Pre-trained model for feature extraction
        """
        self.similarity_threshold = similarity_threshold
        self.hash_threshold = hash_threshold
        self.clustering_eps = clustering_eps
        self.min_samples = min_samples
        self.feature_model_name = feature_model
        
        # Initialize feature extraction model
        self._feature_model = None
        self._scaler = StandardScaler() if StandardScaler else None
        
        # Cache for computed hashes and features
        self._hash_cache = {}
        self._feature_cache = {}
        
        logger.info(f"SimilarityFinder initialized with threshold={similarity_threshold}, "
                   f"hash_threshold={hash_threshold}, eps={clustering_eps}")
    
    def _load_feature_model(self):
        """Load pre-trained CNN model for feature extraction"""
        if self._feature_model is not None or tf is None:
            return
            
        try:
            if self.feature_model_name.lower() == 'resnet50':
                self._feature_model = tf.keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            elif self.feature_model_name.lower() == 'vgg16':
                self._feature_model = tf.keras.applications.VGG16(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            else:
                logger.warning(f"Unknown model {self.feature_model_name}, using ResNet50")
                self._feature_model = tf.keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            
            logger.info(f"Loaded feature extraction model: {self.feature_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load feature model: {e}")
            self._feature_model = None
    
    def compute_hash(self, image_path: str) -> Dict[str, str]:
        """
        Compute perceptual hashes for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pHash, dHash, and aHash values
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {'phash': '', 'dhash': '', 'ahash': ''}
        
        # Check cache first
        cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]
        
        try:
            if Image is None or imagehash is None:
                logger.error("PIL or imagehash not available")
                return {'phash': '', 'dhash': '', 'ahash': ''}
            
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute hashes
                phash = str(imagehash.phash(img))
                dhash = str(imagehash.dhash(img))
                ahash = str(imagehash.average_hash(img))
                
                result = {
                    'phash': phash,
                    'dhash': dhash,
                    'ahash': ahash
                }
                
                # Cache result
                self._hash_cache[cache_key] = result
                
                logger.debug(f"Computed hashes for {image_path}: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Error computing hashes for {image_path}: {e}")
            return {'phash': '', 'dhash': '', 'ahash': ''}
    
    def extract_features(self, image_path: str) -> Optional[Any]:
        """
        Extract deep learning features from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array, or None if extraction fails
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        
        # Check cache first
        cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        try:
            # Load feature model if not already loaded
            self._load_feature_model()
            
            if self._feature_model is None or tf is None:
                logger.warning("Feature model not available, skipping feature extraction")
                return None
            
            # Load and preprocess image
            if cv2 is None:
                logger.error("OpenCV not available")
                return None
                
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            if np is not None:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype('float32') / 255.0
            
            # Add batch dimension
            if np is not None:
                img = np.expand_dims(img, axis=0)
            else:
                # Fallback without numpy
                img = img.reshape((1,) + img.shape)
            
            # Extract features
            features = self._feature_model.predict(img, verbose=0)
            features = features.flatten()
            
            # Cache result
            self._feature_cache[cache_key] = features
            
            logger.debug(f"Extracted {len(features)} features for {image_path}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {image_path}: {e}")
            return None    

    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two perceptual hashes
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Similarity score (0.0 = identical, higher = more different)
        """
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return float('inf')
        
        try:
            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            return float(hamming_distance)
        except Exception as e:
            logger.error(f"Error calculating hash similarity: {e}")
            return float('inf')
    
    def calculate_feature_similarity(self, features1: Any, features2: Any) -> float:
        """
        Calculate cosine similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score (0.0-1.0, higher = more similar)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            if np is None:
                return 0.0
                
            if cosine_similarity is None:
                # Fallback to manual cosine similarity calculation
                dot_product = np.dot(features1, features2)
                norm1 = np.linalg.norm(features1)
                norm2 = np.linalg.norm(features2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
            else:
                # Use sklearn cosine similarity
                similarity = cosine_similarity([features1], [features2])[0][0]
                return float(similarity)
                
        except Exception as e:
            logger.error(f"Error calculating feature similarity: {e}")
            return 0.0
    
    def find_similar_groups(self, image_paths: List[str]) -> Dict[int, List[str]]:
        """
        Group similar images using clustering algorithm
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping group IDs to lists of image paths
        """
        if not image_paths:
            return {}
        
        logger.info(f"Finding similar groups for {len(image_paths)} images")
        
        try:
            # Extract features for all images
            features_list = []
            valid_paths = []
            
            for img_path in image_paths:
                features = self.extract_features(img_path)
                if features is not None:
                    features_list.append(features)
                    valid_paths.append(img_path)
                else:
                    logger.warning(f"Could not extract features for {img_path}")
            
            if len(features_list) < 2:
                logger.warning("Not enough valid images for clustering")
                return {0: valid_paths}
            
            # Convert to numpy array
            if np is None:
                logger.warning("NumPy not available, using hash-based grouping")
                return self._hash_based_grouping(valid_paths)
                
            features_array = np.array(features_list)
            
            # Normalize features if scaler is available
            if self._scaler is not None:
                features_array = self._scaler.fit_transform(features_array)
            
            # Perform clustering
            if DBSCAN is None:
                logger.warning("DBSCAN not available, using hash-based grouping")
                return self._hash_based_grouping(valid_paths)
            
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.min_samples,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(features_array)
            
            # Group images by cluster labels
            groups = {}
            for i, label in enumerate(cluster_labels):
                if label not in groups:
                    groups[label] = []
                groups[label].append(valid_paths[i])
            
            logger.info(f"Found {len(groups)} similarity groups")
            return groups
            
        except Exception as e:
            logger.error(f"Error in similarity grouping: {e}")
            # Fallback to hash-based grouping
            return self._hash_based_grouping(image_paths)
    
    def _hash_based_grouping(self, image_paths: List[str]) -> Dict[int, List[str]]:
        """
        Fallback grouping method using perceptual hashes
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping group IDs to lists of image paths
        """
        groups = {}
        group_id = 0
        processed = set()
        
        for i, img_path in enumerate(image_paths):
            if img_path in processed:
                continue
            
            # Compute hashes for current image
            hashes1 = self.compute_hash(img_path)
            if not any(hashes1.values()):
                continue
            
            # Start new group
            current_group = [img_path]
            processed.add(img_path)
            
            # Find similar images
            for j, other_path in enumerate(image_paths[i+1:], i+1):
                if other_path in processed:
                    continue
                
                hashes2 = self.compute_hash(other_path)
                if not any(hashes2.values()):
                    continue
                
                # Check if images are similar based on any hash
                is_similar = False
                for hash_type in ['phash', 'dhash', 'ahash']:
                    distance = self.calculate_hash_similarity(
                        hashes1[hash_type], 
                        hashes2[hash_type]
                    )
                    if distance <= self.hash_threshold:
                        is_similar = True
                        break
                
                if is_similar:
                    current_group.append(other_path)
                    processed.add(other_path)
            
            groups[group_id] = current_group
            group_id += 1
        
        return groups
    
    def analyze_similarity(self, image_paths: List[str]) -> SimilarityGroupResult:
        """
        Perform comprehensive similarity analysis on a set of images
        
        Args:
            image_paths: List of image file paths to analyze
            
        Returns:
            SimilarityGroupResult with analysis results and recommendations
        """
        logger.info(f"Starting similarity analysis for {len(image_paths)} images")
        
        # Find similar groups
        groups = self.find_similar_groups(image_paths)
        
        # Analyze groups and make recommendations
        duplicate_groups = []
        similar_groups = []
        recommended_keeps = []
        recommended_removes = []
        
        for group_id, group_images in groups.items():
            if not group_images:  # Skip empty groups
                continue
                
            if len(group_images) == 1:
                # Single image, always keep
                recommended_keeps.extend(group_images)
                continue
            
            # Analyze group for duplicates vs similar images
            group_analysis = self._analyze_group(group_images)
            
            if group_analysis['is_duplicate_group']:
                duplicate_groups.append(group_images)
                # Keep the first image, remove others
                recommended_keeps.append(group_images[0])
                recommended_removes.extend(group_images[1:])
            else:
                similar_groups.append(group_images)
                # For similar groups, keep the best quality image
                best_image = self._select_best_image(group_images)
                recommended_keeps.append(best_image)
                recommended_removes.extend([img for img in group_images if img != best_image])
        
        # Calculate statistics
        statistics = {
            'total_groups': len(groups),
            'single_image_groups': len([g for g in groups.values() if len(g) == 1]),
            'duplicate_groups': len(duplicate_groups),
            'similar_groups': len(similar_groups),
            'total_duplicates_found': sum(len(g) - 1 for g in duplicate_groups),
            'total_similar_found': sum(len(g) - 1 for g in similar_groups),
            'recommended_keeps': len(recommended_keeps),
            'recommended_removes': len(recommended_removes)
        }
        
        result = SimilarityGroupResult(
            total_images=len(image_paths),
            total_groups=len(groups),
            duplicate_groups=duplicate_groups,
            similar_groups=similar_groups,
            recommended_keeps=recommended_keeps,
            recommended_removes=recommended_removes,
            statistics=statistics
        )
        
        logger.info(f"Similarity analysis complete: {statistics}")
        return result
    
    def _analyze_group(self, group_images: List[str]) -> Dict[str, Any]:
        """
        Analyze a group of images to determine if they are duplicates or just similar
        
        Args:
            group_images: List of image paths in the group
            
        Returns:
            Dictionary with analysis results
        """
        if len(group_images) < 2:
            return {'is_duplicate_group': False, 'avg_similarity': 0.0}
        
        similarities = []
        hash_similarities = []
        
        # Compare all pairs in the group
        for i in range(len(group_images)):
            for j in range(i + 1, len(group_images)):
                img1, img2 = group_images[i], group_images[j]
                
                # Hash similarity
                hashes1 = self.compute_hash(img1)
                hashes2 = self.compute_hash(img2)
                
                min_hash_distance = float('inf')
                for hash_type in ['phash', 'dhash', 'ahash']:
                    distance = self.calculate_hash_similarity(
                        hashes1[hash_type], 
                        hashes2[hash_type]
                    )
                    min_hash_distance = min(min_hash_distance, distance)
                
                hash_similarities.append(min_hash_distance)
                
                # Feature similarity
                features1 = self.extract_features(img1)
                features2 = self.extract_features(img2)
                
                if features1 is not None and features2 is not None:
                    feature_sim = self.calculate_feature_similarity(features1, features2)
                    similarities.append(feature_sim)
        
        # Determine if group contains duplicates
        if np is not None and hash_similarities:
            avg_hash_distance = np.mean(hash_similarities)
        else:
            avg_hash_distance = sum(hash_similarities) / len(hash_similarities) if hash_similarities else float('inf')
            
        if np is not None and similarities:
            avg_feature_similarity = np.mean(similarities)
        else:
            avg_feature_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Consider as duplicates if hash distance is very low (< 3) or feature similarity is very high (> 0.95)
        is_duplicate_group = (avg_hash_distance < 3) or (avg_feature_similarity > 0.95)
        
        return {
            'is_duplicate_group': is_duplicate_group,
            'avg_hash_distance': avg_hash_distance,
            'avg_feature_similarity': avg_feature_similarity,
            'pair_count': len(similarities)
        }
    
    def _select_best_image(self, group_images: List[str]) -> str:
        """
        Select the best image from a group based on file size and quality metrics
        
        Args:
            group_images: List of image paths in the group
            
        Returns:
            Path to the best image in the group
        """
        if len(group_images) == 1:
            return group_images[0]
        
        best_image = group_images[0]
        best_score = 0
        
        for img_path in group_images:
            try:
                # Simple scoring based on file size (larger is often better quality)
                file_size = os.path.getsize(img_path)
                
                # Get image dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    pixel_count = width * height
                
                # Combined score (file size + pixel count)
                score = file_size + pixel_count * 0.001
                
                if score > best_score:
                    best_score = score
                    best_image = img_path
                    
            except Exception as e:
                logger.warning(f"Error evaluating image {img_path}: {e}")
                continue
        
        return best_image
    
    def get_image_similarity_result(self, image_path: str, all_images: List[str]) -> SimilarityResult:
        """
        Get detailed similarity result for a single image
        
        Args:
            image_path: Path to the image to analyze
            all_images: List of all images to compare against
            
        Returns:
            SimilarityResult with detailed analysis
        """
        # Compute hashes
        hashes = self.compute_hash(image_path)
        
        # Extract features
        features = self.extract_features(image_path)
        
        # Find similar images
        similar_images = []
        similarity_scores = {}
        
        for other_path in all_images:
            if other_path == image_path:
                continue
            
            # Hash similarity
            other_hashes = self.compute_hash(other_path)
            min_hash_distance = float('inf')
            
            for hash_type in ['phash', 'dhash', 'ahash']:
                distance = self.calculate_hash_similarity(
                    hashes[hash_type], 
                    other_hashes[hash_type]
                )
                min_hash_distance = min(min_hash_distance, distance)
            
            # Feature similarity
            other_features = self.extract_features(other_path)
            feature_similarity = 0.0
            
            if features is not None and other_features is not None:
                feature_similarity = self.calculate_feature_similarity(features, other_features)
            
            # Determine if similar
            is_similar = (min_hash_distance <= self.hash_threshold) or \
                        (feature_similarity >= self.similarity_threshold)
            
            if is_similar:
                similar_images.append(other_path)
                similarity_scores[other_path] = {
                    'hash_distance': min_hash_distance,
                    'feature_similarity': feature_similarity
                }
        
        # Determine if duplicate and recommendation
        is_duplicate = any(
            scores['hash_distance'] < 3 or scores['feature_similarity'] > 0.95
            for scores in similarity_scores.values()
        )
        
        if is_duplicate:
            recommended_action = 'remove'
        elif len(similar_images) > 0:
            recommended_action = 'review'
        else:
            recommended_action = 'keep'
        
        return SimilarityResult(
            image_path=image_path,
            phash=hashes['phash'],
            dhash=hashes['dhash'],
            ahash=hashes['ahash'],
            features=features,
            similarity_group=-1,  # Will be set by group analysis
            similar_images=similar_images,
            similarity_scores=similarity_scores,
            is_duplicate=is_duplicate,
            recommended_action=recommended_action
        )