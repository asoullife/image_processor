"""
Unit tests for SimilarityFinder module

Tests cover:
- Perceptual hash computation (pHash, dHash, aHash)
- Deep learning feature extraction
- Similarity scoring and threshold management
- Clustering algorithm (DBSCAN) functionality
- Group recommendation logic
- Edge cases and error handling
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.analyzers.similarity_finder import SimilarityFinder, SimilarityResult, SimilarityGroupResult


class TestSimilarityFinder(unittest.TestCase):
    """Test cases for SimilarityFinder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.similarity_finder = SimilarityFinder(
            similarity_threshold=0.85,
            hash_threshold=5,
            clustering_eps=0.3,
            min_samples=2
        )
        
        # Create test images
        self.test_images = []
        self._create_test_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_images(self):
        """Create test images for similarity testing"""
        try:
            from PIL import Image
            import numpy as np
            
            # Create identical images
            img1 = Image.new('RGB', (100, 100), color='red')
            img1_path = os.path.join(self.test_dir, 'identical1.jpg')
            img1.save(img1_path)
            self.test_images.append(img1_path)
            
            img2 = Image.new('RGB', (100, 100), color='red')
            img2_path = os.path.join(self.test_dir, 'identical2.jpg')
            img2.save(img2_path)
            self.test_images.append(img2_path)
            
            # Create similar images (same color, different size)
            img3 = Image.new('RGB', (150, 150), color='red')
            img3_path = os.path.join(self.test_dir, 'similar1.jpg')
            img3.save(img3_path)
            self.test_images.append(img3_path)
            
            # Create different image
            img4 = Image.new('RGB', (100, 100), color='blue')
            img4_path = os.path.join(self.test_dir, 'different1.jpg')
            img4.save(img4_path)
            self.test_images.append(img4_path)
            
            # Create gradient images for more realistic testing
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(100):
                arr[i, :, 0] = i * 2  # Red gradient
            img5 = Image.fromarray(arr)
            img5_path = os.path.join(self.test_dir, 'gradient1.jpg')
            img5.save(img5_path)
            self.test_images.append(img5_path)
            
            # Similar gradient
            arr2 = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(100):
                arr2[i, :, 0] = i * 2 + 10  # Slightly different gradient
            img6 = Image.fromarray(arr2)
            img6_path = os.path.join(self.test_dir, 'gradient2.jpg')
            img6.save(img6_path)
            self.test_images.append(img6_path)
            
        except ImportError:
            # If PIL is not available, create dummy files
            for i in range(6):
                dummy_path = os.path.join(self.test_dir, f'dummy{i}.jpg')
                with open(dummy_path, 'wb') as f:
                    f.write(b'dummy image data')
                self.test_images.append(dummy_path)
    
    def test_initialization(self):
        """Test SimilarityFinder initialization"""
        finder = SimilarityFinder(
            similarity_threshold=0.9,
            hash_threshold=3,
            clustering_eps=0.2,
            min_samples=3
        )
        
        self.assertEqual(finder.similarity_threshold, 0.9)
        self.assertEqual(finder.hash_threshold, 3)
        self.assertEqual(finder.clustering_eps, 0.2)
        self.assertEqual(finder.min_samples, 3)
        self.assertIsNone(finder._feature_model)
        self.assertEqual(len(finder._hash_cache), 0)
        self.assertEqual(len(finder._feature_cache), 0)
    
    @patch('analyzers.similarity_finder.Image')
    @patch('analyzers.similarity_finder.imagehash')
    def test_compute_hash_success(self, mock_imagehash, mock_image):
        """Test successful hash computation"""
        # Mock PIL Image and imagehash
        mock_img = Mock()
        mock_image.open.return_value.__enter__.return_value = mock_img
        mock_img.mode = 'RGB'
        
        mock_imagehash.phash.return_value = 'phash123'
        mock_imagehash.dhash.return_value = 'dhash456'
        mock_imagehash.average_hash.return_value = 'ahash789'
        
        result = self.similarity_finder.compute_hash(self.test_images[0])
        
        self.assertEqual(result['phash'], 'phash123')
        self.assertEqual(result['dhash'], 'dhash456')
        self.assertEqual(result['ahash'], 'ahash789')
        
        # Test caching
        result2 = self.similarity_finder.compute_hash(self.test_images[0])
        self.assertEqual(result, result2)
        mock_image.open.assert_called_once()  # Should only be called once due to caching
    
    def test_compute_hash_nonexistent_file(self):
        """Test hash computation with nonexistent file"""
        result = self.similarity_finder.compute_hash('nonexistent.jpg')
        
        self.assertEqual(result['phash'], '')
        self.assertEqual(result['dhash'], '')
        self.assertEqual(result['ahash'], '')
    
    @patch('analyzers.similarity_finder.Image')
    @patch('analyzers.similarity_finder.imagehash')
    def test_compute_hash_conversion(self, mock_imagehash, mock_image):
        """Test hash computation with image mode conversion"""
        # Mock PIL Image with RGBA mode
        mock_img = Mock()
        mock_img.mode = 'RGBA'
        mock_converted_img = Mock()
        mock_img.convert.return_value = mock_converted_img
        mock_image.open.return_value.__enter__.return_value = mock_img
        
        mock_imagehash.phash.return_value = 'phash123'
        mock_imagehash.dhash.return_value = 'dhash456'
        mock_imagehash.average_hash.return_value = 'ahash789'
        
        result = self.similarity_finder.compute_hash(self.test_images[0])
        
        mock_img.convert.assert_called_once_with('RGB')
        self.assertEqual(result['phash'], 'phash123')
    
    @patch('analyzers.similarity_finder.tf')
    @patch('analyzers.similarity_finder.cv2')
    def test_extract_features_success(self, mock_cv2, mock_tf):
        """Test successful feature extraction"""
        # Mock TensorFlow model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[1, 2, 3, 4, 5]])
        mock_tf.keras.applications.ResNet50.return_value = mock_model
        
        # Mock OpenCV
        mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_img
        mock_cv2.cvtColor.return_value = mock_img
        mock_cv2.resize.return_value = mock_img
        mock_cv2.COLOR_BGR2RGB = 4  # Mock constant
        
        result = self.similarity_finder.extract_features(self.test_images[0])
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))
    
    def test_extract_features_nonexistent_file(self):
        """Test feature extraction with nonexistent file"""
        result = self.similarity_finder.extract_features('nonexistent.jpg')
        self.assertIsNone(result)
    
    @patch('analyzers.similarity_finder.cv2')
    def test_extract_features_no_tensorflow(self, mock_cv2):
        """Test feature extraction when TensorFlow is not available"""
        with patch('analyzers.similarity_finder.tf', None):
            finder = SimilarityFinder()
            result = finder.extract_features(self.test_images[0])
            self.assertIsNone(result)
    
    def test_calculate_hash_similarity(self):
        """Test hash similarity calculation"""
        # Identical hashes
        similarity = self.similarity_finder.calculate_hash_similarity('abcd', 'abcd')
        self.assertEqual(similarity, 0.0)
        
        # Different hashes
        similarity = self.similarity_finder.calculate_hash_similarity('abcd', 'abce')
        self.assertEqual(similarity, 1.0)
        
        # Completely different hashes
        similarity = self.similarity_finder.calculate_hash_similarity('abcd', 'efgh')
        self.assertEqual(similarity, 4.0)
        
        # Invalid inputs
        similarity = self.similarity_finder.calculate_hash_similarity('', 'abcd')
        self.assertEqual(similarity, float('inf'))
        
        similarity = self.similarity_finder.calculate_hash_similarity('abc', 'abcd')
        self.assertEqual(similarity, float('inf'))
    
    def test_calculate_feature_similarity(self):
        """Test feature similarity calculation"""
        # Identical features
        features1 = np.array([1, 2, 3, 4, 5])
        features2 = np.array([1, 2, 3, 4, 5])
        similarity = self.similarity_finder.calculate_feature_similarity(features1, features2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Orthogonal features
        features1 = np.array([1, 0, 0])
        features2 = np.array([0, 1, 0])
        similarity = self.similarity_finder.calculate_feature_similarity(features1, features2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        # None inputs
        similarity = self.similarity_finder.calculate_feature_similarity(None, features1)
        self.assertEqual(similarity, 0.0)
        
        similarity = self.similarity_finder.calculate_feature_similarity(features1, None)
        self.assertEqual(similarity, 0.0)
    
    @patch('analyzers.similarity_finder.DBSCAN')
    def test_find_similar_groups_with_clustering(self, mock_dbscan):
        """Test similarity grouping with DBSCAN clustering"""
        # Mock DBSCAN
        mock_clustering = Mock()
        mock_clustering.fit_predict.return_value = np.array([0, 0, 1, -1])  # Two groups + noise
        mock_dbscan.return_value = mock_clustering
        
        # Mock feature extraction
        with patch.object(self.similarity_finder, 'extract_features') as mock_extract:
            mock_extract.side_effect = [
                np.array([1, 2, 3]),
                np.array([1.1, 2.1, 3.1]),
                np.array([5, 6, 7]),
                np.array([10, 11, 12])
            ]
            
            groups = self.similarity_finder.find_similar_groups(self.test_images[:4])
            
            self.assertEqual(len(groups), 3)  # Two groups + noise group
            self.assertIn(0, groups)
            self.assertIn(1, groups)
            self.assertIn(-1, groups)
    
    def test_find_similar_groups_fallback(self):
        """Test similarity grouping fallback to hash-based method"""
        with patch('analyzers.similarity_finder.DBSCAN', None):
            with patch.object(self.similarity_finder, 'compute_hash') as mock_hash:
                mock_hash.side_effect = [
                    {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},
                    {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},  # Identical
                    {'phash': 'xyz', 'dhash': 'uvw', 'ahash': 'rst'},  # Different
                ]
                
                groups = self.similarity_finder.find_similar_groups(self.test_images[:3])
                
                # Should have 2 groups: one with similar images, one with different
                self.assertEqual(len(groups), 2)
    
    def test_analyze_similarity(self):
        """Test comprehensive similarity analysis"""
        with patch.object(self.similarity_finder, 'find_similar_groups') as mock_groups:
            mock_groups.return_value = {
                0: [self.test_images[0]],  # Single image
                1: [self.test_images[1], self.test_images[2]],  # Group of 2
                2: [self.test_images[3], self.test_images[4], self.test_images[5]]  # Group of 3
            }
            
            with patch.object(self.similarity_finder, '_analyze_group') as mock_analyze:
                mock_analyze.side_effect = [
                    {'is_duplicate_group': True},  # First group is duplicates
                    {'is_duplicate_group': False}  # Second group is similar
                ]
                
                with patch.object(self.similarity_finder, '_select_best_image') as mock_select:
                    mock_select.side_effect = [
                        self.test_images[1],  # Best from duplicate group
                        self.test_images[3]   # Best from similar group
                    ]
                    
                    result = self.similarity_finder.analyze_similarity(self.test_images)
                    
                    self.assertIsInstance(result, SimilarityGroupResult)
                    self.assertEqual(result.total_images, len(self.test_images))
                    self.assertEqual(result.total_groups, 3)
                    self.assertEqual(len(result.duplicate_groups), 1)
                    self.assertEqual(len(result.similar_groups), 1)
    
    def test_analyze_group_duplicates(self):
        """Test group analysis for duplicate detection"""
        test_group = self.test_images[:2]
        
        with patch.object(self.similarity_finder, 'compute_hash') as mock_hash:
            mock_hash.side_effect = [
                {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},
                {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'}  # Identical hashes
            ]
            
            with patch.object(self.similarity_finder, 'extract_features') as mock_features:
                mock_features.side_effect = [
                    np.array([1, 2, 3]),
                    np.array([1, 2, 3])  # Identical features
                ]
                
                result = self.similarity_finder._analyze_group(test_group)
                
                self.assertTrue(result['is_duplicate_group'])
                self.assertEqual(result['avg_hash_distance'], 0.0)
                self.assertAlmostEqual(result['avg_feature_similarity'], 1.0)
    
    def test_analyze_group_similar(self):
        """Test group analysis for similar (non-duplicate) images"""
        test_group = self.test_images[:2]
        
        with patch.object(self.similarity_finder, 'compute_hash') as mock_hash:
            mock_hash.side_effect = [
                {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},
                {'phash': 'abd', 'dhash': 'deg', 'ahash': 'ghj'}  # Similar but not identical
            ]
            
            with patch.object(self.similarity_finder, 'extract_features') as mock_features:
                mock_features.side_effect = [
                    np.array([1, 2, 3]),
                    np.array([1.2, 2.1, 3.1])  # Similar features
                ]
                
                result = self.similarity_finder._analyze_group(test_group)
                
                self.assertFalse(result['is_duplicate_group'])
                self.assertGreater(result['avg_hash_distance'], 3)
                self.assertLess(result['avg_feature_similarity'], 0.95)
    
    @patch('analyzers.similarity_finder.Image')
    def test_select_best_image(self, mock_image):
        """Test best image selection from group"""
        # Mock image sizes and file sizes
        mock_img1 = Mock()
        mock_img1.size = (100, 100)
        mock_img2 = Mock()
        mock_img2.size = (200, 200)  # Larger image
        
        mock_image.open.side_effect = [
            Mock(__enter__=Mock(return_value=mock_img1)),
            Mock(__enter__=Mock(return_value=mock_img2))
        ]
        
        with patch('os.path.getsize') as mock_size:
            mock_size.side_effect = [1000, 2000]  # Second file is larger
            
            best = self.similarity_finder._select_best_image(self.test_images[:2])
            
            self.assertEqual(best, self.test_images[1])  # Should select larger image
    
    def test_get_image_similarity_result(self):
        """Test getting detailed similarity result for single image"""
        with patch.object(self.similarity_finder, 'compute_hash') as mock_hash:
            mock_hash.side_effect = [
                {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},
                {'phash': 'abc', 'dhash': 'def', 'ahash': 'ghi'},  # Identical
                {'phash': 'xyz', 'dhash': 'uvw', 'ahash': 'rst'}   # Different
            ]
            
            with patch.object(self.similarity_finder, 'extract_features') as mock_features:
                mock_features.side_effect = [
                    np.array([1, 2, 3]),
                    np.array([1, 2, 3]),      # Identical
                    np.array([5, 6, 7])       # Different
                ]
                
                result = self.similarity_finder.get_image_similarity_result(
                    self.test_images[0], 
                    self.test_images[:3]
                )
                
                self.assertIsInstance(result, SimilarityResult)
                self.assertEqual(result.image_path, self.test_images[0])
                self.assertEqual(result.phash, 'abc')
                self.assertEqual(len(result.similar_images), 1)  # Should find one similar
                self.assertTrue(result.is_duplicate)
                self.assertEqual(result.recommended_action, 'remove')
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Empty image list
        groups = self.similarity_finder.find_similar_groups([])
        self.assertEqual(groups, {})
        
        # Single image
        groups = self.similarity_finder.find_similar_groups([self.test_images[0]])
        self.assertEqual(len(groups), 1)
        
        # Analysis with single image group
        result = self.similarity_finder._analyze_group([self.test_images[0]])
        self.assertFalse(result['is_duplicate_group'])
        
        # Best image selection with single image
        best = self.similarity_finder._select_best_image([self.test_images[0]])
        self.assertEqual(best, self.test_images[0])


class TestSimilarityFinderIntegration(unittest.TestCase):
    """Integration tests for SimilarityFinder with real image processing"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.similarity_finder = SimilarityFinder()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_real_image_processing(self):
        """Test with real image processing if dependencies are available"""
        try:
            from PIL import Image
            import numpy as np
            
            # Create test images with known patterns
            img1 = Image.new('RGB', (64, 64), color='red')
            img1_path = os.path.join(self.test_dir, 'red1.jpg')
            img1.save(img1_path)
            
            img2 = Image.new('RGB', (64, 64), color='red')
            img2_path = os.path.join(self.test_dir, 'red2.jpg')
            img2.save(img2_path)
            
            img3 = Image.new('RGB', (64, 64), color='blue')
            img3_path = os.path.join(self.test_dir, 'blue1.jpg')
            img3.save(img3_path)
            
            test_images = [img1_path, img2_path, img3_path]
            
            # Test hash computation
            hashes1 = self.similarity_finder.compute_hash(img1_path)
            hashes2 = self.similarity_finder.compute_hash(img2_path)
            hashes3 = self.similarity_finder.compute_hash(img3_path)
            
            # Red images should have identical or very similar hashes
            self.assertIsNotNone(hashes1['phash'])
            self.assertIsNotNone(hashes2['phash'])
            self.assertIsNotNone(hashes3['phash'])
            
            # Test similarity analysis
            result = self.similarity_finder.analyze_similarity(test_images)
            
            self.assertIsInstance(result, SimilarityGroupResult)
            self.assertEqual(result.total_images, 3)
            self.assertGreaterEqual(result.total_groups, 1)
            
        except ImportError:
            self.skipTest("PIL not available for integration testing")


if __name__ == '__main__':
    unittest.main()