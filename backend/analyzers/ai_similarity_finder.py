"""
AI-Enhanced Similarity Finder for Adobe Stock Image Processor

This module implements AI-enhanced similarity detection using:
- CLIP embeddings for semantic similarity
- Advanced perceptual hashing with AI validation
- Deep learning feature extraction with TensorFlow
- Intelligent clustering with DBSCAN optimization
- Performance mode optimization (Speed/Balanced/Smart)
"""

import os
import logging
import gc
import time
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path

from backend.ml import runtime

probe = runtime.probe()
tf = probe.tf
cv2 = probe.cv2
TF_AVAILABLE = tf is not None
CV2_AVAILABLE = cv2 is not None

try:
    import torch
    import clip
    import numpy as np
    from PIL import Image
    import imagehash
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
    CLIP_AVAILABLE = True
    PIL_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI/ML dependencies not available: {e}")
    torch = None
    clip = None
    np = None
    Image = None
    imagehash = None
    DBSCAN = None
    cosine_similarity = None
    StandardScaler = None
    TORCH_AVAILABLE = False
    CLIP_AVAILABLE = False
    PIL_AVAILABLE = False
    SKLEARN_AVAILABLE = False

from .ai_model_manager import AIModelManager
try:
    from .similarity_finder import SimilarityFinder, SimilarityResult, SimilarityGroupResult
except ImportError:
    # Fallback for testing without full backend structure
    from dataclasses import dataclass
    from typing import List, Dict, Any
    
    @dataclass
    class SimilarityResult:
        """Fallback SimilarityResult for testing"""
        similar_images: List[str]
        similarity_scores: Dict[str, float]
        hash_distances: Dict[str, int]
        feature_distances: Dict[str, float]
        recommended_removes: List[str]
    
    @dataclass
    class SimilarityGroupResult:
        """Fallback SimilarityGroupResult for testing"""
        total_groups: int
        duplicate_groups: Dict[int, List[str]]
        similar_groups: Dict[int, List[str]]
        recommended_removes: List[str]
        group_representatives: Dict[int, str]
    
    class SimilarityFinder:
        """Fallback SimilarityFinder for testing"""
        def __init__(self, similarity_threshold=0.85, hash_threshold=5, clustering_eps=0.3):
            self.similarity_threshold = similarity_threshold
            self.hash_threshold = hash_threshold
            self.clustering_eps = clustering_eps
            logger.info(f"SimilarityFinder initialized with threshold={similarity_threshold}, hash_threshold={hash_threshold}, eps={clustering_eps}")
        
        def analyze_similarity(self, image_paths: List[str]) -> SimilarityGroupResult:
            return SimilarityGroupResult(0, {}, {}, [], {})

logger = logging.getLogger(__name__)


@dataclass
class AISimilarityResult:
    """Enhanced similarity analysis result with AI features"""
    traditional_result: SimilarityResult
    clip_embedding: Optional[Any]  # CLIP feature vector
    ai_similarity_scores: Dict[str, float]
    semantic_similarity: float
    visual_similarity: float
    ai_confidence: float
    ai_reasoning: str
    ai_reasoning_thai: str
    similarity_category: str  # 'identical', 'near_duplicate', 'similar', 'unique'
    recommended_action: str  # 'keep', 'remove', 'review'
    processing_time: float
    model_used: str
    fallback_used: bool


@dataclass
class AIGroupResult:
    """Enhanced group analysis result with AI clustering"""
    traditional_result: SimilarityGroupResult
    ai_clusters: Dict[int, List[str]]
    semantic_groups: Dict[int, List[str]]
    visual_groups: Dict[int, List[str]]
    ai_recommendations: Dict[str, str]
    cluster_quality_scores: Dict[int, float]
    ai_reasoning: str
    ai_reasoning_thai: str
    processing_time: float
    model_used: str
    fallback_used: bool


class AISimilarityFinder:
    """
    AI-Enhanced Similarity Finder with CLIP embeddings and advanced clustering
    
    Features:
    - CLIP model for semantic similarity understanding
    - TensorFlow models for visual feature extraction
    - Advanced clustering with quality assessment
    - GPU acceleration optimized for RTX2060
    - Fallback to traditional methods when AI unavailable
    - Performance mode optimization (Speed/Balanced/Smart)
    - Thai language explanations for recommendations
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: Optional[AIModelManager] = None):
        """
        Initialize AI Similarity Finder
        
        Args:
            config: Configuration dictionary
            model_manager: Optional AI model manager instance
        """
        self.config = config
        self.similarity_config = config.get('similarity', {})
        
        # Initialize model manager
        if model_manager is not None:
            self.model_manager = model_manager
        else:
            self.model_manager = AIModelManager(config)
        
        # Initialize traditional finder as fallback
        self.traditional_finder = SimilarityFinder(
            similarity_threshold=self.similarity_config.get('feature_threshold', 0.85),
            hash_threshold=self.similarity_config.get('hash_threshold', 5),
            clustering_eps=self.similarity_config.get('clustering_eps', 0.3)
        )
        
        # AI-specific thresholds - configurable for different use cases
        self.ai_thresholds = {
            'clip_similarity_threshold': self.similarity_config.get('clip_threshold', 0.90),
            'visual_similarity_threshold': self.similarity_config.get('visual_threshold', 0.85),
            'identical_threshold': self.similarity_config.get('identical_threshold', 0.95),
            'near_duplicate_threshold': self.similarity_config.get('near_duplicate_threshold', 0.90),
            'similar_threshold': self.similarity_config.get('similar_threshold', 0.75),
            'min_ai_confidence': self.similarity_config.get('min_confidence', 0.7),
            'clustering_eps': self.similarity_config.get('clustering_eps', 0.3),
            'min_cluster_size': self.similarity_config.get('min_cluster_size', 2)
        }
        
        # Similarity use case presets
        self.use_case_presets = {
            'strict': {
                'clip_threshold': 0.95,
                'visual_threshold': 0.90,
                'identical_threshold': 0.98,
                'near_duplicate_threshold': 0.95,
                'similar_threshold': 0.85,
                'clustering_eps': 0.2
            },
            'balanced': {
                'clip_threshold': 0.90,
                'visual_threshold': 0.85,
                'identical_threshold': 0.95,
                'near_duplicate_threshold': 0.90,
                'similar_threshold': 0.75,
                'clustering_eps': 0.3
            },
            'lenient': {
                'clip_threshold': 0.85,
                'visual_threshold': 0.80,
                'identical_threshold': 0.90,
                'near_duplicate_threshold': 0.85,
                'similar_threshold': 0.70,
                'clustering_eps': 0.4
            }
        }
        
        # Performance mode settings
        self.performance_mode = "balanced"
        self.batch_size = 8  # CLIP is memory-intensive
        
        # CLIP model
        self.clip_model = None
        self.clip_preprocess = None
        
        # Thai language explanations
        self.thai_reasons = {
            'identical': 'ภาพเหมือนกันทุกประการ ควรเก็บเพียงภาพเดียว',
            'near_duplicate': 'ภาพเกือบเหมือนกัน อาจถูกปฏิเสธเป็น spam',
            'similar_content': 'เนื้อหาคล้ายกัน ควรเลือกภาพที่ดีที่สุด',
            'similar_composition': 'องค์ประกอบภาพคล้ายกัน อาจซ้ำซ้อน',
            'unique_content': 'เนื้อหาไม่ซ้ำกัน เหมาะสำหรับการขาย',
            'low_confidence': 'AI ไม่มั่นใจในการประเมิน ควรตรวจสอบด้วยตนเอง',
            'multiple_similar': 'พบภาพคล้ายกันหลายภาพ ควรเลือกเก็บเพียงภาพเดียว'
        }
        
        logger.info(f"AISimilarityFinder initialized - CLIP: {CLIP_AVAILABLE}, TF: {TF_AVAILABLE}")
    
    def _load_clip_model(self):
        """Load CLIP model for semantic similarity"""
        if self.clip_model is not None or not CLIP_AVAILABLE:
            return
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            logger.info(f"Loaded CLIP model on {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode for AI processing
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        self.performance_mode = mode
        self.model_manager.set_performance_mode(mode)
        
        # Update batch size and thresholds based on mode
        mode_settings = {
            'speed': {'batch_size': 16, 'clip_threshold': 0.85},
            'balanced': {'batch_size': 8, 'clip_threshold': 0.90},
            'smart': {'batch_size': 4, 'clip_threshold': 0.92}
        }
        
        settings = mode_settings.get(mode, mode_settings['balanced'])
        self.batch_size = settings['batch_size']
        self.ai_thresholds['clip_similarity_threshold'] = settings['clip_threshold']
        
        logger.info(f"Performance mode set to: {mode}, batch_size: {self.batch_size}")
    
    def set_similarity_use_case(self, use_case: str):
        """
        Set similarity detection use case with predefined thresholds
        
        Args:
            use_case: Use case preset ('strict', 'balanced', 'lenient')
        """
        if use_case in self.use_case_presets:
            preset = self.use_case_presets[use_case]
            self.ai_thresholds.update(preset)
            logger.info(f"Similarity use case set to: {use_case}")
        else:
            logger.warning(f"Unknown use case: {use_case}, using balanced preset")
            self.set_similarity_use_case('balanced')
    
    def configure_thresholds(self, **thresholds):
        """
        Configure custom similarity thresholds
        
        Args:
            **thresholds: Custom threshold values
        """
        for key, value in thresholds.items():
            if key in self.ai_thresholds:
                self.ai_thresholds[key] = value
                logger.info(f"Updated threshold {key} to {value}")
            else:
                logger.warning(f"Unknown threshold: {key}")
    
    def analyze_similarity(self, image_paths: List[str]) -> AIGroupResult:
        """
        Perform AI-enhanced similarity analysis on a set of images
        
        Args:
            image_paths: List of image file paths to analyze
            
        Returns:
            AIGroupResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Always perform traditional analysis first (fallback)
            traditional_result = self.traditional_finder.analyze_similarity(image_paths)
            
            # Attempt AI-enhanced analysis
            if CLIP_AVAILABLE and self._load_clip_model():
                ai_result = self._analyze_with_ai(image_paths, traditional_result)
                processing_time = time.time() - start_time
                ai_result.processing_time = processing_time
                return ai_result
            else:
                # Fallback to traditional analysis only
                logger.info("AI models not available, using traditional analysis")
                return self._create_fallback_result(traditional_result, time.time() - start_time)
                
        except Exception as e:
            logger.error(f"Error in AI similarity analysis: {e}")
            # Return fallback result on error
            traditional_result = self.traditional_finder.analyze_similarity(image_paths)
            return self._create_fallback_result(traditional_result, time.time() - start_time)
    
    def _analyze_with_ai(self, image_paths: List[str], traditional_result: SimilarityGroupResult) -> AIGroupResult:
        """
        Perform AI-enhanced analysis using CLIP embeddings
        
        Args:
            image_paths: List of image file paths
            traditional_result: Traditional analysis result
            
        Returns:
            AIGroupResult with AI enhancements
        """
        try:
            # Extract CLIP embeddings for all images
            clip_embeddings = self._extract_clip_embeddings(image_paths)
            
            # Perform semantic clustering
            semantic_groups = self._cluster_semantic_similarity(image_paths, clip_embeddings)
            
            # Perform visual clustering using traditional methods
            visual_groups = traditional_result.similar_groups
            
            # Combine AI and traditional clustering
            ai_clusters = self._combine_clustering_results(semantic_groups, visual_groups)
            
            # Generate AI recommendations
            ai_recommendations = self._generate_ai_recommendations(ai_clusters, clip_embeddings)
            
            # Calculate cluster quality scores
            cluster_quality_scores = self._calculate_cluster_quality(ai_clusters, clip_embeddings)
            
            # Generate AI reasoning
            reasoning_en, reasoning_th = self._generate_ai_reasoning(ai_clusters, ai_recommendations)
            
            result = AIGroupResult(
                traditional_result=traditional_result,
                ai_clusters=ai_clusters,
                semantic_groups=semantic_groups,
                visual_groups=visual_groups,
                ai_recommendations=ai_recommendations,
                cluster_quality_scores=cluster_quality_scores,
                ai_reasoning=reasoning_en,
                ai_reasoning_thai=reasoning_th,
                processing_time=0.0,  # Will be set by caller
                model_used='clip_vit_b32',
                fallback_used=False
            )
            
            # Cleanup memory
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI similarity analysis: {e}")
            return self._create_fallback_result(traditional_result, 0.0)
    
    def _extract_clip_embeddings(self, image_paths: List[str]) -> Dict[str, Any]:
        """Extract CLIP embeddings for all images"""
        embeddings = {}
        
        if not CLIP_AVAILABLE or self.clip_model is None:
            return embeddings
        
        try:
            for image_path in image_paths:
                if PIL_AVAILABLE and Image:
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.clip_preprocess(image).unsqueeze(0)
                    
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image_tensor)
                        embeddings[image_path] = embedding.cpu().numpy()
                        
        except Exception as e:
            logger.error(f"Error extracting CLIP embeddings: {e}")
        
        return embeddings
    
    def _cluster_semantic_similarity(self, image_paths: List[str], embeddings: Dict[str, Any]) -> Dict[int, List[str]]:
        """Cluster images based on semantic similarity with enhanced algorithm"""
        if not embeddings or not SKLEARN_AVAILABLE:
            return {}
        
        try:
            # Prepare embedding matrix
            embedding_matrix = []
            path_list = []
            
            for path, embedding in embeddings.items():
                embedding_matrix.append(embedding.flatten())
                path_list.append(path)
            
            if len(embedding_matrix) < 2:
                return {0: path_list}
            
            embedding_matrix = np.array(embedding_matrix)
            
            # Normalize embeddings for better cosine similarity
            from sklearn.preprocessing import normalize
            embedding_matrix = normalize(embedding_matrix, norm='l2')
            
            # Perform DBSCAN clustering with configurable parameters
            clustering = DBSCAN(
                eps=self.ai_thresholds['clustering_eps'],
                min_samples=self.ai_thresholds['min_cluster_size'],
                metric='cosine'
            )
            cluster_labels = clustering.fit_predict(embedding_matrix)
            
            # Group by clusters and calculate cluster quality
            clusters = {}
            cluster_qualities = {}
            
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(path_list[i])
            
            # Calculate cluster quality scores
            for cluster_id, cluster_paths in clusters.items():
                if len(cluster_paths) > 1:
                    # Calculate intra-cluster similarity
                    cluster_embeddings = [embeddings[path].flatten() for path in cluster_paths]
                    similarities = []
                    
                    for i in range(len(cluster_embeddings)):
                        for j in range(i + 1, len(cluster_embeddings)):
                            sim = self._calculate_cosine_similarity(
                                cluster_embeddings[i], cluster_embeddings[j]
                            )
                            similarities.append(sim)
                    
                    cluster_qualities[cluster_id] = np.mean(similarities) if similarities else 0.5
                else:
                    cluster_qualities[cluster_id] = 1.0
            
            # Filter out low-quality clusters
            min_quality = 0.7
            filtered_clusters = {
                cluster_id: paths for cluster_id, paths in clusters.items()
                if cluster_qualities.get(cluster_id, 0) >= min_quality or len(paths) == 1
            }
            
            logger.debug(f"Semantic clustering: {len(filtered_clusters)} clusters from {len(image_paths)} images")
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            return {}
    
    def _combine_clustering_results(self, semantic_groups: Dict[int, List[str]], 
                                   visual_groups: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """Combine semantic and visual clustering results"""
        # For now, prioritize semantic clustering
        if semantic_groups:
            return semantic_groups
        else:
            return visual_groups
    
    def _generate_ai_recommendations(self, clusters: Dict[int, List[str]], 
                                   embeddings: Dict[str, Any]) -> Dict[str, str]:
        """Generate AI-based recommendations for each image with detailed reasoning"""
        recommendations = {}
        
        for cluster_id, image_paths in clusters.items():
            if len(image_paths) > 1:
                # Multiple images in cluster - analyze similarity levels
                cluster_analysis = self._analyze_cluster_similarity(image_paths, embeddings)
                
                # Categorize similarity level
                avg_similarity = cluster_analysis['average_similarity']
                if avg_similarity >= self.ai_thresholds['identical_threshold']:
                    # Identical images - keep only the best one
                    best_image = self._select_best_image_in_cluster(image_paths, embeddings)
                    for image_path in image_paths:
                        if image_path == best_image:
                            recommendations[image_path] = 'keep_best'
                        else:
                            recommendations[image_path] = 'remove_identical'
                            
                elif avg_similarity >= self.ai_thresholds['near_duplicate_threshold']:
                    # Near duplicates - keep best, mark others for review
                    best_image = self._select_best_image_in_cluster(image_paths, embeddings)
                    for image_path in image_paths:
                        if image_path == best_image:
                            recommendations[image_path] = 'keep_best'
                        else:
                            recommendations[image_path] = 'review_duplicate'
                            
                elif avg_similarity >= self.ai_thresholds['similar_threshold']:
                    # Similar content - recommend review
                    for image_path in image_paths:
                        recommendations[image_path] = 'review_similar'
                        
                else:
                    # Low similarity - might be false positive
                    for image_path in image_paths:
                        recommendations[image_path] = 'keep_different'
            else:
                # Single image in cluster - unique content
                recommendations[image_paths[0]] = 'keep_unique'
        
        return recommendations
    
    def _analyze_cluster_similarity(self, image_paths: List[str], embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze similarity within a cluster"""
        similarities = []
        max_similarity = 0.0
        min_similarity = 1.0
        
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                path1, path2 = image_paths[i], image_paths[j]
                if path1 in embeddings and path2 in embeddings:
                    sim = self._calculate_cosine_similarity(
                        embeddings[path1], embeddings[path2]
                    )
                    similarities.append(sim)
                    max_similarity = max(max_similarity, sim)
                    min_similarity = min(min_similarity, sim)
        
        return {
            'average_similarity': np.mean(similarities) if similarities else 0.0,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'similarity_variance': np.var(similarities) if similarities else 0.0,
            'cluster_size': len(image_paths)
        }
    
    def _select_best_image_in_cluster(self, image_paths: List[str], embeddings: Dict[str, Any]) -> str:
        """Select the best image from a cluster based on multiple criteria"""
        if len(image_paths) == 1:
            return image_paths[0]
        
        try:
            # Score each image based on multiple criteria
            image_scores = {}
            
            for image_path in image_paths:
                score = 0.0
                
                # 1. File size (larger is often better quality)
                try:
                    file_size = os.path.getsize(image_path)
                    score += min(file_size / (1024 * 1024), 10) * 0.2  # Max 10MB, 20% weight
                except:
                    pass
                
                # 2. Image dimensions (higher resolution is better)
                try:
                    if PIL_AVAILABLE and Image:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            megapixels = (width * height) / 1000000
                            score += min(megapixels, 50) * 0.3  # Max 50MP, 30% weight
                except:
                    pass
                
                # 3. Filename quality (avoid obvious duplicates)
                filename = os.path.basename(image_path).lower()
                if 'copy' in filename or 'duplicate' in filename or '(1)' in filename:
                    score -= 2.0  # Penalty for obvious duplicates
                
                # 4. Embedding quality (distance from cluster center)
                if image_path in embeddings:
                    cluster_embeddings = [embeddings[path] for path in image_paths if path in embeddings]
                    if len(cluster_embeddings) > 1:
                        # Calculate distance from cluster centroid
                        centroid = np.mean(cluster_embeddings, axis=0)
                        distance = np.linalg.norm(embeddings[image_path] - centroid)
                        score += (1.0 - min(distance, 1.0)) * 0.5  # 50% weight, closer to center is better
                
                image_scores[image_path] = score
            
            # Return image with highest score
            best_image = max(image_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Selected best image: {os.path.basename(best_image)} (score: {image_scores[best_image]:.2f})")
            return best_image
            
        except Exception as e:
            logger.error(f"Error selecting best image: {e}")
            return image_paths[0]  # Fallback to first image
    
    def _calculate_cluster_quality(self, clusters: Dict[int, List[str]], 
                                 embeddings: Dict[str, Any]) -> Dict[int, float]:
        """Calculate quality scores for each cluster"""
        quality_scores = {}
        
        for cluster_id, image_paths in clusters.items():
            if len(image_paths) > 1:
                # Calculate intra-cluster similarity
                similarities = []
                for i, path1 in enumerate(image_paths):
                    for j, path2 in enumerate(image_paths[i+1:], i+1):
                        if path1 in embeddings and path2 in embeddings:
                            sim = self._calculate_cosine_similarity(
                                embeddings[path1], embeddings[path2]
                            )
                            similarities.append(sim)
                
                quality_scores[cluster_id] = np.mean(similarities) if similarities else 0.5
            else:
                quality_scores[cluster_id] = 1.0  # Single image clusters are perfect
        
        return quality_scores
    
    def _calculate_cosine_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if cosine_similarity is not None:
                sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
                return float(sim[0][0])
            else:
                # Fallback calculation
                dot_product = np.dot(embedding1.flatten(), embedding2.flatten())
                norm1 = np.linalg.norm(embedding1.flatten())
                norm2 = np.linalg.norm(embedding2.flatten())
                return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.5
    
    def _generate_ai_reasoning(self, clusters: Dict[int, List[str]], 
                             recommendations: Dict[str, str]) -> Tuple[str, str]:
        """Generate AI reasoning in English and Thai"""
        try:
            total_images = sum(len(paths) for paths in clusters.values())
            total_clusters = len(clusters)
            remove_count = sum(1 for action in recommendations.values() if action == 'remove')
            
            # English reasoning
            reasoning_en = f"AI analyzed {total_images} images and found {total_clusters} similarity clusters. "
            reasoning_en += f"Recommended removing {remove_count} similar images to avoid spam detection."
            
            # Thai reasoning
            reasoning_th = f"AI วิเคราะห์ภาพ {total_images} ภาพ และพบกลุ่มภาพคล้ายกัน {total_clusters} กลุ่ม "
            reasoning_th += f"แนะนำให้ลบภาพที่คล้ายกัน {remove_count} ภาพ เพื่อหลีกเลี่ยงการถูกตรวจจับเป็น spam"
            
            return reasoning_en, reasoning_th
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            return "AI analysis completed", "การวิเคราะห์ด้วย AI เสร็จสิ้น"
    
    def _create_fallback_result(self, traditional_result: SimilarityGroupResult, 
                              processing_time: float) -> AIGroupResult:
        """Create fallback result when AI analysis is not available"""
        
        # Generate simple reasoning
        reasoning_en = "Traditional similarity analysis completed"
        reasoning_th = "การวิเคราะห์ความคล้ายกันแบบดั้งเดิมเสร็จสิ้น"
        
        return AIGroupResult(
            traditional_result=traditional_result,
            ai_clusters={},
            semantic_groups={},
            visual_groups=traditional_result.similar_groups,
            ai_recommendations={},
            cluster_quality_scores={},
            ai_reasoning=reasoning_en,
            ai_reasoning_thai=reasoning_th,
            processing_time=processing_time,
            model_used='traditional_fallback',
            fallback_used=True
        )
    
    def _cleanup_memory(self):
        """Cleanup memory after AI processing"""
        try:
            # Python garbage collection
            gc.collect()
            
            # PyTorch memory cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status for AI similarity analysis
        
        Returns:
            Status information dictionary
        """
        status = {
            'clip_available': CLIP_AVAILABLE,
            'tensorflow_available': TF_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'opencv_available': CV2_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'performance_mode': self.performance_mode,
            'batch_size': self.batch_size,
            'clip_model_loaded': self.clip_model is not None,
            'model_manager_status': self.model_manager.get_system_status(),
            'fallback_available': self.traditional_finder is not None
        }
        
        return status
    
    def _analyze_with_ai(self, image_paths: List[str], 
                        traditional_result: SimilarityGroupResult) -> AIGroupResult:
        """
        Perform AI-enhanced similarity analysis using CLIP and TensorFlow
        
        Args:
            image_paths: List of image file paths
            traditional_result: Traditional analysis result
            
        Returns:
            AIGroupResult with AI enhancements
        """
        try:
            # Extract CLIP embeddings for all images
            clip_embeddings = self._extract_clip_embeddings_batch(image_paths)
            
            # Extract visual features using TensorFlow
            visual_features = self._extract_visual_features_batch(image_paths)
            
            # Perform AI-based clustering
            ai_clusters = self._perform_ai_clustering(image_paths, clip_embeddings, visual_features)
            
            # Separate semantic and visual grouping
            semantic_groups = self._cluster_by_semantic_similarity(image_paths, clip_embeddings)
            visual_groups = self._cluster_by_visual_similarity(image_paths, visual_features)
            
            # Generate AI recommendations
            ai_recommendations = self._generate_ai_recommendations(
                image_paths, ai_clusters, clip_embeddings, visual_features
            )
            
            # Calculate cluster quality scores
            cluster_quality_scores = self._calculate_cluster_quality(ai_clusters, clip_embeddings)
            
            # Generate AI reasoning
            reasoning_en, reasoning_th = self._generate_group_reasoning(
                ai_clusters, semantic_groups, visual_groups, cluster_quality_scores
            )
            
            result = AIGroupResult(
                traditional_result=traditional_result,
                ai_clusters=ai_clusters,
                semantic_groups=semantic_groups,
                visual_groups=visual_groups,
                ai_recommendations=ai_recommendations,
                cluster_quality_scores=cluster_quality_scores,
                ai_reasoning=reasoning_en,
                ai_reasoning_thai=reasoning_th,
                processing_time=0.0,  # Will be set by caller
                model_used='clip_vit_b32',
                fallback_used=False
            )
            
            # Cleanup memory
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI similarity analysis: {e}")
            return self._create_fallback_group_result(traditional_result, 0.0)
    
    def _extract_clip_embeddings_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Extract CLIP embeddings for all images in batches
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to CLIP embeddings
        """
        embeddings = {}
        
        if not CLIP_AVAILABLE or self.clip_model is None:
            return embeddings
        
        try:
            # Process in batches to manage memory
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_embeddings = self._extract_clip_batch(batch_paths)
                embeddings.update(batch_embeddings)
                
                # Memory cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.debug(f"Extracted CLIP embeddings for {len(embeddings)} images")
            
        except Exception as e:
            logger.error(f"Error extracting CLIP embeddings: {e}")
        
        return embeddings
    
    def _extract_clip_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Extract CLIP embeddings for a batch of images
        
        Args:
            image_paths: List of image file paths in the batch
            
        Returns:
            Dictionary mapping image paths to CLIP embeddings
        """
        batch_embeddings = {}
        
        try:
            # Load and preprocess images
            images = []
            valid_paths = []
            
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.clip_preprocess(image).unsqueeze(0)
                    images.append(image_tensor)
                    valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")
                    continue
            
            if not images:
                return batch_embeddings
            
            # Stack images into batch tensor
            batch_tensor = torch.cat(images, dim=0)
            
            # Move to device
            device = next(self.clip_model.parameters()).device
            batch_tensor = batch_tensor.to(device)
            
            # Extract features
            with torch.no_grad():
                features = self.clip_model.encode_image(batch_tensor)
                features = features.cpu().numpy()
            
            # Map features to image paths
            for i, img_path in enumerate(valid_paths):
                batch_embeddings[img_path] = features[i]
            
        except Exception as e:
            logger.error(f"Error in CLIP batch processing: {e}")
        
        return batch_embeddings
    
    def _extract_visual_features_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Extract visual features using TensorFlow models
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to visual features
        """
        features = {}
        
        try:
            # Get TensorFlow model for feature extraction
            feature_model = self.model_manager.get_model('resnet50')
            if feature_model is None:
                logger.warning("ResNet50 not available for visual features")
                return features
            
            # Process in batches
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_features = self._extract_visual_batch(batch_paths, feature_model)
                features.update(batch_features)
            
            logger.debug(f"Extracted visual features for {len(features)} images")
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
        
        return features
    
    def _extract_visual_batch(self, image_paths: List[str], model: Any) -> Dict[str, Any]:
        """
        Extract visual features for a batch of images using TensorFlow
        
        Args:
            image_paths: List of image file paths in the batch
            model: TensorFlow model for feature extraction
            
        Returns:
            Dictionary mapping image paths to visual features
        """
        batch_features = {}
        
        try:
            # Load and preprocess images
            images = []
            valid_paths = []
            
            for img_path in image_paths:
                try:
                    if cv2 is None or np is None:
                        continue
                        
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB and resize
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    image = image.astype(np.float32) / 255.0
                    
                    images.append(image)
                    valid_paths.append(img_path)
                    
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")
                    continue
            
            if not images:
                return batch_features
            
            # Stack into batch tensor
            batch_tensor = np.array(images)
            
            # Extract features
            features = model.predict(batch_tensor, verbose=0)
            
            # Map features to image paths
            for i, img_path in enumerate(valid_paths):
                batch_features[img_path] = features[i].flatten()
            
        except Exception as e:
            logger.error(f"Error in visual feature batch processing: {e}")
        
        return batch_features
    
    def _perform_ai_clustering(self, image_paths: List[str], 
                              clip_embeddings: Dict[str, Any],
                              visual_features: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Perform AI-based clustering combining CLIP and visual features
        
        Args:
            image_paths: List of image file paths
            clip_embeddings: CLIP embeddings for images
            visual_features: Visual features for images
            
        Returns:
            Dictionary mapping cluster IDs to lists of image paths
        """
        clusters = {}
        
        try:
            if not SKLEARN_AVAILABLE or not clip_embeddings:
                return {0: image_paths}  # Single cluster fallback
            
            # Combine CLIP and visual features
            combined_features = []
            valid_paths = []
            
            for img_path in image_paths:
                if img_path in clip_embeddings and img_path in visual_features:
                    clip_feat = clip_embeddings[img_path]
                    visual_feat = visual_features[img_path]
                    
                    # Normalize features
                    clip_norm = clip_feat / (np.linalg.norm(clip_feat) + 1e-8)
                    visual_norm = visual_feat / (np.linalg.norm(visual_feat) + 1e-8)
                    
                    # Combine with weights (CLIP gets higher weight for semantic similarity)
                    combined = np.concatenate([clip_norm * 0.7, visual_norm * 0.3])
                    combined_features.append(combined)
                    valid_paths.append(img_path)
            
            if len(combined_features) < 2:
                return {0: valid_paths}
            
            # Perform DBSCAN clustering
            features_array = np.array(combined_features)
            
            # Adaptive eps based on performance mode
            eps_values = {
                'speed': 0.4,
                'balanced': 0.3,
                'smart': 0.25
            }
            eps = eps_values.get(self.performance_mode, 0.3)
            
            clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(features_array)
            
            # Group images by cluster labels
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_paths[i])
            
            logger.debug(f"AI clustering found {len(clusters)} clusters")
            
        except Exception as e:
            logger.error(f"Error in AI clustering: {e}")
            clusters = {0: image_paths}  # Fallback to single cluster
        
        return clusters
    
    def _cluster_by_semantic_similarity(self, image_paths: List[str], 
                                       clip_embeddings: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Cluster images by semantic similarity using CLIP embeddings
        
        Args:
            image_paths: List of image file paths
            clip_embeddings: CLIP embeddings for images
            
        Returns:
            Dictionary mapping cluster IDs to lists of image paths
        """
        clusters = {}
        
        try:
            if not clip_embeddings or not SKLEARN_AVAILABLE:
                return {0: image_paths}
            
            # Extract valid embeddings
            embeddings = []
            valid_paths = []
            
            for img_path in image_paths:
                if img_path in clip_embeddings:
                    embeddings.append(clip_embeddings[img_path])
                    valid_paths.append(img_path)
            
            if len(embeddings) < 2:
                return {0: valid_paths}
            
            # Perform clustering with tighter threshold for semantic similarity
            embeddings_array = np.array(embeddings)
            clustering = DBSCAN(eps=0.2, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings_array)
            
            # Group by cluster labels
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_paths[i])
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            clusters = {0: image_paths}
        
        return clusters
    
    def _cluster_by_visual_similarity(self, image_paths: List[str], 
                                     visual_features: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Cluster images by visual similarity using TensorFlow features
        
        Args:
            image_paths: List of image file paths
            visual_features: Visual features for images
            
        Returns:
            Dictionary mapping cluster IDs to lists of image paths
        """
        clusters = {}
        
        try:
            if not visual_features or not SKLEARN_AVAILABLE:
                return {0: image_paths}
            
            # Extract valid features
            features = []
            valid_paths = []
            
            for img_path in image_paths:
                if img_path in visual_features:
                    features.append(visual_features[img_path])
                    valid_paths.append(img_path)
            
            if len(features) < 2:
                return {0: valid_paths}
            
            # Normalize features
            features_array = np.array(features)
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
            cluster_labels = clustering.fit_predict(features_normalized)
            
            # Group by cluster labels
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_paths[i])
            
        except Exception as e:
            logger.error(f"Error in visual clustering: {e}")
            clusters = {0: image_paths}
        
        return clusters
    
    def _generate_ai_recommendations(self, image_paths: List[str], 
                                   ai_clusters: Dict[int, List[str]],
                                   clip_embeddings: Dict[str, Any],
                                   visual_features: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate AI-based recommendations for each image
        
        Args:
            image_paths: List of image file paths
            ai_clusters: AI clustering results
            clip_embeddings: CLIP embeddings
            visual_features: Visual features
            
        Returns:
            Dictionary mapping image paths to recommendations
        """
        recommendations = {}
        
        try:
            for cluster_id, cluster_images in ai_clusters.items():
                if len(cluster_images) == 1:
                    # Single image in cluster - keep
                    recommendations[cluster_images[0]] = 'keep'
                else:
                    # Multiple images in cluster - analyze similarity
                    cluster_recs = self._analyze_cluster_for_recommendations(
                        cluster_images, clip_embeddings, visual_features
                    )
                    recommendations.update(cluster_recs)
            
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            # Fallback: keep all images
            for img_path in image_paths:
                recommendations[img_path] = 'keep'
        
        return recommendations
    
    def _analyze_cluster_for_recommendations(self, cluster_images: List[str],
                                           clip_embeddings: Dict[str, Any],
                                           visual_features: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze a cluster of similar images and make recommendations
        
        Args:
            cluster_images: List of image paths in the cluster
            clip_embeddings: CLIP embeddings
            visual_features: Visual features
            
        Returns:
            Dictionary mapping image paths to recommendations
        """
        recommendations = {}
        
        try:
            if len(cluster_images) <= 1:
                for img in cluster_images:
                    recommendations[img] = 'keep'
                return recommendations
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(cluster_images)):
                for j in range(i + 1, len(cluster_images)):
                    img1, img2 = cluster_images[i], cluster_images[j]
                    
                    # CLIP similarity
                    clip_sim = 0.0
                    if img1 in clip_embeddings and img2 in clip_embeddings:
                        clip_sim = cosine_similarity(
                            [clip_embeddings[img1]], 
                            [clip_embeddings[img2]]
                        )[0][0]
                    
                    # Visual similarity
                    visual_sim = 0.0
                    if img1 in visual_features and img2 in visual_features:
                        visual_sim = cosine_similarity(
                            [visual_features[img1]], 
                            [visual_features[img2]]
                        )[0][0]
                    
                    # Combined similarity
                    combined_sim = (clip_sim * 0.6) + (visual_sim * 0.4)
                    similarities.append((img1, img2, combined_sim))
            
            # Find the best image to keep (highest average similarity to others)
            image_scores = {}
            for img in cluster_images:
                scores = []
                for img1, img2, sim in similarities:
                    if img1 == img:
                        scores.append(sim)
                    elif img2 == img:
                        scores.append(sim)
                
                if scores:
                    image_scores[img] = np.mean(scores)
                else:
                    image_scores[img] = 0.0
            
            # Sort by score and file size (as tiebreaker)
            sorted_images = sorted(cluster_images, key=lambda x: (
                image_scores.get(x, 0.0),
                os.path.getsize(x) if os.path.exists(x) else 0
            ), reverse=True)
            
            # Make recommendations based on similarity levels
            max_similarity = max([sim for _, _, sim in similarities]) if similarities else 0.0
            
            if max_similarity > self.ai_thresholds['identical_threshold']:
                # Very high similarity - keep only the best one
                recommendations[sorted_images[0]] = 'keep'
                for img in sorted_images[1:]:
                    recommendations[img] = 'remove'
            elif max_similarity > self.ai_thresholds['near_duplicate_threshold']:
                # Near duplicates - keep best, review others
                recommendations[sorted_images[0]] = 'keep'
                for img in sorted_images[1:]:
                    recommendations[img] = 'review'
            else:
                # Similar but not duplicates - review all
                for img in cluster_images:
                    recommendations[img] = 'review'
            
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            # Fallback: keep all
            for img in cluster_images:
                recommendations[img] = 'keep'
        
        return recommendations
    
    def _calculate_cluster_quality(self, ai_clusters: Dict[int, List[str]], 
                                  clip_embeddings: Dict[str, Any]) -> Dict[int, float]:
        """
        Calculate quality scores for each cluster
        
        Args:
            ai_clusters: AI clustering results
            clip_embeddings: CLIP embeddings
            
        Returns:
            Dictionary mapping cluster IDs to quality scores
        """
        quality_scores = {}
        
        try:
            for cluster_id, cluster_images in ai_clusters.items():
                if len(cluster_images) <= 1:
                    quality_scores[cluster_id] = 1.0
                    continue
                
                # Calculate intra-cluster similarity
                similarities = []
                for i in range(len(cluster_images)):
                    for j in range(i + 1, len(cluster_images)):
                        img1, img2 = cluster_images[i], cluster_images[j]
                        
                        if img1 in clip_embeddings and img2 in clip_embeddings:
                            sim = cosine_similarity(
                                [clip_embeddings[img1]], 
                                [clip_embeddings[img2]]
                            )[0][0]
                            similarities.append(sim)
                
                # Quality is average intra-cluster similarity
                if similarities:
                    quality_scores[cluster_id] = float(np.mean(similarities))
                else:
                    quality_scores[cluster_id] = 0.5
            
        except Exception as e:
            logger.error(f"Error calculating cluster quality: {e}")
            # Default quality scores
            for cluster_id in ai_clusters.keys():
                quality_scores[cluster_id] = 0.5
        
        return quality_scores
    
    def _generate_group_reasoning(self, ai_clusters: Dict[int, List[str]],
                                 semantic_groups: Dict[int, List[str]],
                                 visual_groups: Dict[int, List[str]],
                                 cluster_quality_scores: Dict[int, float]) -> Tuple[str, str]:
        """
        Generate reasoning for group analysis results
        
        Args:
            ai_clusters: AI clustering results
            semantic_groups: Semantic clustering results
            visual_groups: Visual clustering results
            cluster_quality_scores: Cluster quality scores
            
        Returns:
            Tuple of (English reasoning, Thai reasoning)
        """
        try:
            # Count clusters and images
            total_clusters = len(ai_clusters)
            single_image_clusters = len([c for c in ai_clusters.values() if len(c) == 1])
            multi_image_clusters = total_clusters - single_image_clusters
            
            # Calculate average cluster quality
            avg_quality = np.mean(list(cluster_quality_scores.values())) if cluster_quality_scores else 0.5
            
            # English reasoning
            reasoning_parts_en = [
                f"AI analysis found {total_clusters} similarity clusters",
                f"{single_image_clusters} unique images, {multi_image_clusters} similarity groups",
                f"Average cluster quality: {avg_quality:.2f}"
            ]
            
            if multi_image_clusters > 0:
                reasoning_parts_en.append(f"Detected potential duplicates in {multi_image_clusters} groups")
            
            reasoning_en = ". ".join(reasoning_parts_en) + "."
            
            # Thai reasoning
            reasoning_parts_th = [
                f"การวิเคราะห์ด้วย AI พบกลุ่มความคล้ายกัน {total_clusters} กลุ่ม",
                f"ภาพไม่ซ้ำ {single_image_clusters} ภาพ กลุ่มคล้ายกัน {multi_image_clusters} กลุ่ม",
                f"คุณภาพการจัดกลุ่มเฉลี่ย: {avg_quality:.2f}"
            ]
            
            if multi_image_clusters > 0:
                reasoning_parts_th.append(f"ตรวจพบภาพที่อาจซ้ำกันใน {multi_image_clusters} กลุ่ม")
            
            reasoning_th = " ".join(reasoning_parts_th)
            
            return reasoning_en, reasoning_th
            
        except Exception as e:
            logger.error(f"Error generating group reasoning: {e}")
            return "AI similarity analysis completed", "การวิเคราะห์ความคล้ายกันด้วย AI เสร็จสิ้น"
    
    def _create_fallback_group_result(self, traditional_result: SimilarityGroupResult, 
                                    processing_time: float) -> AIGroupResult:
        """
        Create fallback result when AI analysis is not available
        
        Args:
            traditional_result: Traditional analysis result
            processing_time: Processing time in seconds
            
        Returns:
            AIGroupResult using traditional analysis
        """
        # Convert traditional groups to AI format
        ai_clusters = {}
        for i, group in enumerate(traditional_result.duplicate_groups + traditional_result.similar_groups):
            ai_clusters[i] = group
        
        # Generate simple recommendations
        ai_recommendations = {}
        for img in traditional_result.recommended_keeps:
            ai_recommendations[img] = 'keep'
        for img in traditional_result.recommended_removes:
            ai_recommendations[img] = 'remove'
        
        # Default quality scores
        cluster_quality_scores = {i: 0.7 for i in ai_clusters.keys()}
        
        return AIGroupResult(
            traditional_result=traditional_result,
            ai_clusters=ai_clusters,
            semantic_groups=ai_clusters,  # Same as AI clusters for fallback
            visual_groups=ai_clusters,    # Same as AI clusters for fallback
            ai_recommendations=ai_recommendations,
            cluster_quality_scores=cluster_quality_scores,
            ai_reasoning="Traditional similarity analysis completed",
            ai_reasoning_thai="การวิเคราะห์ความคล้ายกันแบบดั้งเดิมเสร็จสิ้น",
            processing_time=processing_time,
            model_used='traditional_fallback',
            fallback_used=True
        )
    
    def _cleanup_memory(self):
        """Cleanup memory after AI processing"""
        try:
            # Python garbage collection
            gc.collect()
            
            # PyTorch memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # TensorFlow memory cleanup
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status for AI similarity analysis
        
        Returns:
            Status information dictionary
        """
        status = {
            'clip_available': CLIP_AVAILABLE,
            'tensorflow_available': TF_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'opencv_available': CV2_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'performance_mode': self.performance_mode,
            'batch_size': self.batch_size,
            'clip_model_loaded': self.clip_model is not None,
            'model_manager_status': self.model_manager.get_system_status(),
            'fallback_available': self.traditional_finder is not None
        }
        
        return status