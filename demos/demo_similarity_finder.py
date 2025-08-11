#!/usr/bin/env python3
"""
Demo script for SimilarityFinder module

This script demonstrates the functionality of the SimilarityFinder class including:
- Perceptual hash computation
- Feature extraction
- Similarity detection
- Image grouping and clustering
- Duplicate detection and recommendations
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analyzers.similarity_finder import SimilarityFinder
import logging


def create_demo_images(demo_dir):
    """Create demo images for similarity testing"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        print("Creating demo images...")
        
        # Create identical images (duplicates)
        img1 = Image.new('RGB', (200, 200), color='red')
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([50, 50, 150, 150], fill='blue')
        img1_path = os.path.join(demo_dir, 'duplicate1.jpg')
        img1.save(img1_path, quality=95)
        
        img2 = Image.new('RGB', (200, 200), color='red')
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([50, 50, 150, 150], fill='blue')
        img2_path = os.path.join(demo_dir, 'duplicate2.jpg')
        img2.save(img2_path, quality=90)  # Slightly different quality
        
        # Create similar images (same pattern, different colors)
        img3 = Image.new('RGB', (200, 200), color='green')
        draw3 = ImageDraw.Draw(img3)
        draw3.rectangle([50, 50, 150, 150], fill='yellow')
        img3_path = os.path.join(demo_dir, 'similar1.jpg')
        img3.save(img3_path, quality=95)
        
        img4 = Image.new('RGB', (200, 200), color='green')
        draw4 = ImageDraw.Draw(img4)
        draw4.rectangle([45, 45, 155, 155], fill='yellow')  # Slightly different position
        img4_path = os.path.join(demo_dir, 'similar2.jpg')
        img4.save(img4_path, quality=95)
        
        # Create different images
        img5 = Image.new('RGB', (200, 200), color='white')
        draw5 = ImageDraw.Draw(img5)
        draw5.ellipse([25, 25, 175, 175], fill='purple')
        img5_path = os.path.join(demo_dir, 'different1.jpg')
        img5.save(img5_path, quality=95)
        
        img6 = Image.new('RGB', (200, 200), color='black')
        draw6 = ImageDraw.Draw(img6)
        for i in range(0, 200, 20):
            draw6.line([(i, 0), (i, 200)], fill='white', width=2)
        img6_path = os.path.join(demo_dir, 'different2.jpg')
        img6.save(img6_path, quality=95)
        
        # Create gradient images
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                arr[i, j, 0] = int((i / 200) * 255)  # Red gradient
                arr[i, j, 1] = int((j / 200) * 255)  # Green gradient
                arr[i, j, 2] = 128  # Constant blue
        
        img7 = Image.fromarray(arr)
        img7_path = os.path.join(demo_dir, 'gradient1.jpg')
        img7.save(img7_path, quality=95)
        
        # Similar gradient with slight modification
        arr2 = arr.copy()
        arr2[:, :, 2] = 140  # Slightly different blue
        img8 = Image.fromarray(arr2)
        img8_path = os.path.join(demo_dir, 'gradient2.jpg')
        img8.save(img8_path, quality=95)
        
        return [img1_path, img2_path, img3_path, img4_path, 
                img5_path, img6_path, img7_path, img8_path]
        
    except ImportError:
        print("PIL not available, creating dummy images...")
        # Create dummy files for testing
        dummy_images = []
        for i in range(8):
            dummy_path = os.path.join(demo_dir, f'dummy{i}.jpg')
            with open(dummy_path, 'wb') as f:
                f.write(b'dummy image data ' + str(i).encode())
            dummy_images.append(dummy_path)
        return dummy_images


def demo_hash_computation(similarity_finder, image_paths):
    """Demonstrate hash computation functionality"""
    print("\n" + "="*60)
    print("HASH COMPUTATION DEMO")
    print("="*60)
    
    for i, img_path in enumerate(image_paths[:4]):  # Test first 4 images
        print(f"\nImage {i+1}: {os.path.basename(img_path)}")
        hashes = similarity_finder.compute_hash(img_path)
        
        print(f"  pHash: {hashes['phash']}")
        print(f"  dHash: {hashes['dhash']}")
        print(f"  aHash: {hashes['ahash']}")
        
        # Test hash similarity with first image
        if i > 0:
            ref_hashes = similarity_finder.compute_hash(image_paths[0])
            phash_sim = similarity_finder.calculate_hash_similarity(
                hashes['phash'], ref_hashes['phash']
            )
            dhash_sim = similarity_finder.calculate_hash_similarity(
                hashes['dhash'], ref_hashes['dhash']
            )
            ahash_sim = similarity_finder.calculate_hash_similarity(
                hashes['ahash'], ref_hashes['ahash']
            )
            
            print(f"  Similarity to Image 1:")
            print(f"    pHash distance: {phash_sim}")
            print(f"    dHash distance: {dhash_sim}")
            print(f"    aHash distance: {ahash_sim}")


def demo_feature_extraction(similarity_finder, image_paths):
    """Demonstrate feature extraction functionality"""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION DEMO")
    print("="*60)
    
    features_list = []
    
    for i, img_path in enumerate(image_paths[:4]):  # Test first 4 images
        print(f"\nExtracting features for Image {i+1}: {os.path.basename(img_path)}")
        features = similarity_finder.extract_features(img_path)
        
        if features is not None:
            print(f"  Feature vector shape: {features.shape}")
            print(f"  Feature vector sample: {features[:5]}...")
            features_list.append(features)
        else:
            print("  Feature extraction failed or not available")
            features_list.append(None)
    
    # Test feature similarity
    if len(features_list) >= 2 and features_list[0] is not None and features_list[1] is not None:
        similarity = similarity_finder.calculate_feature_similarity(
            features_list[0], features_list[1]
        )
        print(f"\nFeature similarity between Image 1 and 2: {similarity:.4f}")


def demo_similarity_grouping(similarity_finder, image_paths):
    """Demonstrate similarity grouping functionality"""
    print("\n" + "="*60)
    print("SIMILARITY GROUPING DEMO")
    print("="*60)
    
    print(f"Analyzing {len(image_paths)} images for similarity groups...")
    
    # Find similar groups
    groups = similarity_finder.find_similar_groups(image_paths)
    
    print(f"\nFound {len(groups)} similarity groups:")
    
    for group_id, group_images in groups.items():
        print(f"\nGroup {group_id} ({len(group_images)} images):")
        for img_path in group_images:
            print(f"  - {os.path.basename(img_path)}")


def demo_comprehensive_analysis(similarity_finder, image_paths):
    """Demonstrate comprehensive similarity analysis"""
    print("\n" + "="*60)
    print("COMPREHENSIVE SIMILARITY ANALYSIS DEMO")
    print("="*60)
    
    print(f"Performing comprehensive analysis on {len(image_paths)} images...")
    
    # Perform analysis
    result = similarity_finder.analyze_similarity(image_paths)
    
    print(f"\nAnalysis Results:")
    print(f"  Total images: {result.total_images}")
    print(f"  Total groups: {result.total_groups}")
    print(f"  Duplicate groups: {len(result.duplicate_groups)}")
    print(f"  Similar groups: {len(result.similar_groups)}")
    print(f"  Recommended keeps: {len(result.recommended_keeps)}")
    print(f"  Recommended removes: {len(result.recommended_removes)}")
    
    print(f"\nStatistics:")
    for key, value in result.statistics.items():
        print(f"  {key}: {value}")
    
    if result.duplicate_groups:
        print(f"\nDuplicate Groups:")
        for i, group in enumerate(result.duplicate_groups):
            print(f"  Group {i+1}:")
            for img_path in group:
                print(f"    - {os.path.basename(img_path)}")
    
    if result.similar_groups:
        print(f"\nSimilar Groups:")
        for i, group in enumerate(result.similar_groups):
            print(f"  Group {i+1}:")
            for img_path in group:
                print(f"    - {os.path.basename(img_path)}")
    
    print(f"\nRecommended Actions:")
    print(f"  Keep ({len(result.recommended_keeps)} images):")
    for img_path in result.recommended_keeps[:5]:  # Show first 5
        print(f"    - {os.path.basename(img_path)}")
    if len(result.recommended_keeps) > 5:
        print(f"    ... and {len(result.recommended_keeps) - 5} more")
    
    print(f"  Remove ({len(result.recommended_removes)} images):")
    for img_path in result.recommended_removes[:5]:  # Show first 5
        print(f"    - {os.path.basename(img_path)}")
    if len(result.recommended_removes) > 5:
        print(f"    ... and {len(result.recommended_removes) - 5} more")


def demo_individual_analysis(similarity_finder, image_paths):
    """Demonstrate individual image similarity analysis"""
    print("\n" + "="*60)
    print("INDIVIDUAL IMAGE ANALYSIS DEMO")
    print("="*60)
    
    if len(image_paths) < 2:
        print("Need at least 2 images for individual analysis")
        return
    
    # Analyze first image against all others
    target_image = image_paths[0]
    print(f"Analyzing {os.path.basename(target_image)} against all other images...")
    
    result = similarity_finder.get_image_similarity_result(target_image, image_paths)
    
    print(f"\nResults for {os.path.basename(target_image)}:")
    print(f"  pHash: {result.phash}")
    print(f"  dHash: {result.dhash}")
    print(f"  aHash: {result.ahash}")
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


def main():
    """Main demo function"""
    print("SimilarityFinder Demo")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create temporary directory for demo images
    demo_dir = tempfile.mkdtemp(prefix='similarity_demo_')
    
    try:
        # Create demo images
        image_paths = create_demo_images(demo_dir)
        print(f"Created {len(image_paths)} demo images in {demo_dir}")
        
        # Initialize SimilarityFinder
        similarity_finder = SimilarityFinder(
            similarity_threshold=0.85,
            hash_threshold=5,
            clustering_eps=0.3,
            min_samples=2
        )
        
        # Run demos
        demo_hash_computation(similarity_finder, image_paths)
        demo_feature_extraction(similarity_finder, image_paths)
        demo_similarity_grouping(similarity_finder, image_paths)
        demo_comprehensive_analysis(similarity_finder, image_paths)
        demo_individual_analysis(similarity_finder, image_paths)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            shutil.rmtree(demo_dir, ignore_errors=True)
            print(f"\nCleaned up demo directory: {demo_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up demo directory: {e}")


if __name__ == '__main__':
    main()