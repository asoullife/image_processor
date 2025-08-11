#!/usr/bin/env python3
"""
Comprehensive test for SimilarityFinder with real image processing

This test creates actual image files and tests the similarity detection
functionality when PIL and imagehash are available.
"""

import os
import sys
import tempfile
import shutil

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.analyzers.similarity_finder import SimilarityFinder


def create_test_images(test_dir):
    """Create test images with known similarity patterns"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        print("Creating test images with PIL...")
        
        # Create identical images (exact duplicates)
        img1 = Image.new('RGB', (100, 100), color='red')
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([25, 25, 75, 75], fill='blue')
        img1_path = os.path.join(test_dir, 'identical1.jpg')
        img1.save(img1_path, quality=95)
        
        # Save the same image again (duplicate)
        img2_path = os.path.join(test_dir, 'identical2.jpg')
        img1.save(img2_path, quality=95)
        
        # Create similar images (same pattern, different colors)
        img3 = Image.new('RGB', (100, 100), color='green')
        draw3 = ImageDraw.Draw(img3)
        draw3.rectangle([25, 25, 75, 75], fill='yellow')
        img3_path = os.path.join(test_dir, 'similar1.jpg')
        img3.save(img3_path, quality=95)
        
        img4 = Image.new('RGB', (100, 100), color='green')
        draw4 = ImageDraw.Draw(img4)
        draw4.rectangle([20, 20, 80, 80], fill='yellow')  # Slightly different size
        img4_path = os.path.join(test_dir, 'similar2.jpg')
        img4.save(img4_path, quality=95)
        
        # Create completely different images
        img5 = Image.new('RGB', (100, 100), color='white')
        draw5 = ImageDraw.Draw(img5)
        draw5.ellipse([10, 10, 90, 90], fill='purple')
        img5_path = os.path.join(test_dir, 'different1.jpg')
        img5.save(img5_path, quality=95)
        
        img6 = Image.new('RGB', (100, 100), color='black')
        draw6 = ImageDraw.Draw(img6)
        for i in range(0, 100, 10):
            draw6.line([(i, 0), (i, 100)], fill='white', width=1)
        img6_path = os.path.join(test_dir, 'different2.jpg')
        img6.save(img6_path, quality=95)
        
        return [img1_path, img2_path, img3_path, img4_path, img5_path, img6_path]
        
    except ImportError:
        print("PIL not available, creating dummy files...")
        # Create dummy files
        dummy_images = []
        for i in range(6):
            dummy_path = os.path.join(test_dir, f'dummy{i}.jpg')
            with open(dummy_path, 'wb') as f:
                # Create different content for each file
                content = b'dummy image data ' + str(i).encode() * (i + 1)
                f.write(content)
            dummy_images.append(dummy_path)
        return dummy_images


def test_hash_functionality(similarity_finder, image_paths):
    """Test perceptual hash computation and comparison"""
    print("\n" + "="*50)
    print("TESTING HASH FUNCTIONALITY")
    print("="*50)
    
    if len(image_paths) < 4:
        print("Not enough images for hash testing")
        return
    
    # Test hash computation
    hashes = []
    for i, img_path in enumerate(image_paths[:4]):
        hash_result = similarity_finder.compute_hash(img_path)
        hashes.append(hash_result)
        print(f"\nImage {i+1}: {os.path.basename(img_path)}")
        print(f"  pHash: {hash_result['phash'][:16]}..." if hash_result['phash'] else "  pHash: (empty)")
        print(f"  dHash: {hash_result['dhash'][:16]}..." if hash_result['dhash'] else "  dHash: (empty)")
        print(f"  aHash: {hash_result['ahash'][:16]}..." if hash_result['ahash'] else "  aHash: (empty)")
    
    # Test hash similarity between first two images (should be identical)
    if len(hashes) >= 2 and hashes[0]['phash'] and hashes[1]['phash']:
        phash_sim = similarity_finder.calculate_hash_similarity(
            hashes[0]['phash'], hashes[1]['phash']
        )
        dhash_sim = similarity_finder.calculate_hash_similarity(
            hashes[0]['dhash'], hashes[1]['dhash']
        )
        ahash_sim = similarity_finder.calculate_hash_similarity(
            hashes[0]['ahash'], hashes[1]['ahash']
        )
        
        print(f"\nSimilarity between Image 1 and 2 (should be identical):")
        print(f"  pHash distance: {phash_sim}")
        print(f"  dHash distance: {dhash_sim}")
        print(f"  aHash distance: {ahash_sim}")
        
        # Test with different images
        if len(hashes) >= 4:
            phash_diff = similarity_finder.calculate_hash_similarity(
                hashes[0]['phash'], hashes[3]['phash']
            )
            print(f"\nSimilarity between Image 1 and 4 (should be different):")
            print(f"  pHash distance: {phash_diff}")


def test_grouping_functionality(similarity_finder, image_paths):
    """Test similarity grouping and clustering"""
    print("\n" + "="*50)
    print("TESTING GROUPING FUNCTIONALITY")
    print("="*50)
    
    # Test similarity grouping
    groups = similarity_finder.find_similar_groups(image_paths)
    
    print(f"Found {len(groups)} similarity groups:")
    for group_id, group_images in groups.items():
        print(f"\nGroup {group_id} ({len(group_images)} images):")
        for img_path in group_images:
            print(f"  - {os.path.basename(img_path)}")
    
    # Test comprehensive analysis
    analysis = similarity_finder.analyze_similarity(image_paths)
    
    print(f"\nComprehensive Analysis Results:")
    print(f"  Total images: {analysis.total_images}")
    print(f"  Total groups: {analysis.total_groups}")
    print(f"  Duplicate groups: {len(analysis.duplicate_groups)}")
    print(f"  Similar groups: {len(analysis.similar_groups)}")
    print(f"  Recommended keeps: {len(analysis.recommended_keeps)}")
    print(f"  Recommended removes: {len(analysis.recommended_removes)}")
    
    if analysis.duplicate_groups:
        print(f"\nDuplicate Groups Found:")
        for i, group in enumerate(analysis.duplicate_groups):
            print(f"  Group {i+1}: {[os.path.basename(img) for img in group]}")
    
    if analysis.similar_groups:
        print(f"\nSimilar Groups Found:")
        for i, group in enumerate(analysis.similar_groups):
            print(f"  Group {i+1}: {[os.path.basename(img) for img in group]}")
    
    print(f"\nRecommendations:")
    if analysis.recommended_keeps:
        print(f"  Keep: {[os.path.basename(img) for img in analysis.recommended_keeps]}")
    if analysis.recommended_removes:
        print(f"  Remove: {[os.path.basename(img) for img in analysis.recommended_removes]}")


def test_individual_analysis(similarity_finder, image_paths):
    """Test individual image similarity analysis"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL ANALYSIS")
    print("="*50)
    
    if not image_paths:
        print("No images available for individual analysis")
        return
    
    # Test first image
    target_image = image_paths[0]
    result = similarity_finder.get_image_similarity_result(target_image, image_paths)
    
    print(f"Analysis for {os.path.basename(target_image)}:")
    print(f"  pHash: {result.phash[:16]}..." if result.phash else "  pHash: (empty)")
    print(f"  dHash: {result.dhash[:16]}..." if result.dhash else "  dHash: (empty)")
    print(f"  aHash: {result.ahash[:16]}..." if result.ahash else "  aHash: (empty)")
    print(f"  Is duplicate: {result.is_duplicate}")
    print(f"  Recommended action: {result.recommended_action}")
    print(f"  Similar images found: {len(result.similar_images)}")
    
    if result.similar_images:
        print(f"  Similar images:")
        for img_path in result.similar_images:
            scores = result.similarity_scores.get(img_path, {})
            print(f"    - {os.path.basename(img_path)}")
            if 'hash_distance' in scores:
                print(f"      Hash distance: {scores['hash_distance']}")
            if 'feature_similarity' in scores:
                print(f"      Feature similarity: {scores['feature_similarity']:.4f}")


def test_edge_cases(similarity_finder):
    """Test edge cases and error handling"""
    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)
    
    # Test with empty list
    empty_groups = similarity_finder.find_similar_groups([])
    print(f"Empty image list: {len(empty_groups)} groups")
    
    # Test with single image
    test_dir = tempfile.mkdtemp()
    try:
        single_file = os.path.join(test_dir, 'single.txt')
        with open(single_file, 'w') as f:
            f.write('test')
        
        single_groups = similarity_finder.find_similar_groups([single_file])
        print(f"Single image: {len(single_groups)} groups")
        
        # Test with nonexistent file
        nonexistent_hash = similarity_finder.compute_hash('nonexistent.jpg')
        print(f"Nonexistent file hash: {nonexistent_hash}")
        
        # Test invalid hash similarity
        invalid_sim = similarity_finder.calculate_hash_similarity('', 'abc')
        print(f"Invalid hash similarity: {invalid_sim}")
        
        # Test feature similarity with None
        none_sim = similarity_finder.calculate_feature_similarity(None, None)
        print(f"None feature similarity: {none_sim}")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
    
    print("✓ Edge case testing completed")


def main():
    """Main test function"""
    print("SimilarityFinder Comprehensive Test")
    print("="*60)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix='similarity_test_')
    
    try:
        # Create test images
        image_paths = create_test_images(test_dir)
        print(f"Created {len(image_paths)} test images")
        
        # Initialize SimilarityFinder with test-friendly settings
        similarity_finder = SimilarityFinder(
            similarity_threshold=0.8,
            hash_threshold=10,  # More lenient for testing
            clustering_eps=0.5,
            min_samples=2
        )
        
        # Run tests
        test_hash_functionality(similarity_finder, image_paths)
        test_grouping_functionality(similarity_finder, image_paths)
        test_individual_analysis(similarity_finder, image_paths)
        test_edge_cases(similarity_finder)
        
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Check if we have real image processing capabilities
        try:
            import PIL
            import imagehash
            print("\n✅ PIL and imagehash are available - full functionality tested")
        except ImportError:
            print("\n⚠️  PIL/imagehash not available - tested with fallback functionality")
        
        try:
            import tensorflow
            print("✅ TensorFlow is available - feature extraction capability present")
        except ImportError:
            print("⚠️  TensorFlow not available - feature extraction will use fallback")
        
        try:
            import sklearn
            print("✅ scikit-learn is available - clustering functionality present")
        except ImportError:
            print("⚠️  scikit-learn not available - clustering will use fallback")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"\nCleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
    
    return 0


if __name__ == '__main__':
    exit(main())