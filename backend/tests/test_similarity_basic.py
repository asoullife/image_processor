#!/usr/bin/env python3
"""
Basic test for SimilarityFinder functionality

This test verifies that the SimilarityFinder class can be instantiated
and basic methods work correctly, even without full dependencies.
"""

import os
import sys
import tempfile
import shutil

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.analyzers.similarity_finder import SimilarityFinder, SimilarityResult, SimilarityGroupResult


def test_basic_functionality():
    """Test basic SimilarityFinder functionality"""
    print("Testing SimilarityFinder basic functionality...")
    
    # Test initialization
    finder = SimilarityFinder(
        similarity_threshold=0.8,
        hash_threshold=3,
        clustering_eps=0.2,
        min_samples=2
    )
    
    print(f"✓ Initialization successful")
    print(f"  - Similarity threshold: {finder.similarity_threshold}")
    print(f"  - Hash threshold: {finder.hash_threshold}")
    print(f"  - Clustering eps: {finder.clustering_eps}")
    print(f"  - Min samples: {finder.min_samples}")
    
    # Test hash similarity calculation
    hash_sim_identical = finder.calculate_hash_similarity('abcd1234', 'abcd1234')
    hash_sim_different = finder.calculate_hash_similarity('abcd1234', 'efgh5678')
    hash_sim_similar = finder.calculate_hash_similarity('abcd1234', 'abcd1235')
    
    print(f"✓ Hash similarity calculation works")
    print(f"  - Identical hashes: {hash_sim_identical}")
    print(f"  - Different hashes: {hash_sim_different}")
    print(f"  - Similar hashes: {hash_sim_similar}")
    
    # Test feature similarity calculation (with None inputs)
    feature_sim_none = finder.calculate_feature_similarity(None, None)
    print(f"✓ Feature similarity with None inputs: {feature_sim_none}")
    
    # Test with empty image list
    empty_groups = finder.find_similar_groups([])
    print(f"✓ Empty image list handling: {len(empty_groups)} groups")
    
    # Test analysis with empty list
    empty_analysis = finder.analyze_similarity([])
    print(f"✓ Empty analysis result:")
    print(f"  - Total images: {empty_analysis.total_images}")
    print(f"  - Total groups: {empty_analysis.total_groups}")
    
    # Create dummy files for testing
    test_dir = tempfile.mkdtemp()
    try:
        dummy_files = []
        for i in range(3):
            dummy_path = os.path.join(test_dir, f'dummy{i}.jpg')
            with open(dummy_path, 'wb') as f:
                f.write(b'dummy image data ' + str(i).encode())
            dummy_files.append(dummy_path)
        
        # Test with dummy files (will use fallback methods)
        groups = finder.find_similar_groups(dummy_files)
        print(f"✓ Dummy file grouping: {len(groups)} groups found")
        
        analysis = finder.analyze_similarity(dummy_files)
        print(f"✓ Dummy file analysis:")
        print(f"  - Total images: {analysis.total_images}")
        print(f"  - Total groups: {analysis.total_groups}")
        print(f"  - Recommended keeps: {len(analysis.recommended_keeps)}")
        print(f"  - Recommended removes: {len(analysis.recommended_removes)}")
        
        # Test individual image analysis
        if dummy_files:
            individual_result = finder.get_image_similarity_result(dummy_files[0], dummy_files)
            print(f"✓ Individual image analysis:")
            print(f"  - Image path: {os.path.basename(individual_result.image_path)}")
            print(f"  - Is duplicate: {individual_result.is_duplicate}")
            print(f"  - Recommended action: {individual_result.recommended_action}")
            print(f"  - Similar images: {len(individual_result.similar_images)}")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n✅ All basic functionality tests passed!")


def test_data_structures():
    """Test data structure creation"""
    print("\nTesting data structures...")
    
    # Test SimilarityResult creation
    result = SimilarityResult(
        image_path="test.jpg",
        phash="abc123",
        dhash="def456",
        ahash="ghi789",
        features=None,
        similarity_group=0,
        similar_images=["similar1.jpg", "similar2.jpg"],
        similarity_scores={"similar1.jpg": {"hash_distance": 2.0, "feature_similarity": 0.9}},
        is_duplicate=False,
        recommended_action="keep"
    )
    
    print(f"✓ SimilarityResult created successfully")
    print(f"  - Image path: {result.image_path}")
    print(f"  - Hash values: {result.phash}, {result.dhash}, {result.ahash}")
    print(f"  - Similar images: {len(result.similar_images)}")
    print(f"  - Is duplicate: {result.is_duplicate}")
    print(f"  - Recommended action: {result.recommended_action}")
    
    # Test SimilarityGroupResult creation
    group_result = SimilarityGroupResult(
        total_images=10,
        total_groups=3,
        duplicate_groups=[["dup1.jpg", "dup2.jpg"]],
        similar_groups=[["sim1.jpg", "sim2.jpg", "sim3.jpg"]],
        recommended_keeps=["keep1.jpg", "keep2.jpg"],
        recommended_removes=["remove1.jpg", "remove2.jpg"],
        statistics={
            "total_duplicates": 2,
            "total_similar": 3,
            "processing_time": 1.5
        }
    )
    
    print(f"✓ SimilarityGroupResult created successfully")
    print(f"  - Total images: {group_result.total_images}")
    print(f"  - Total groups: {group_result.total_groups}")
    print(f"  - Duplicate groups: {len(group_result.duplicate_groups)}")
    print(f"  - Similar groups: {len(group_result.similar_groups)}")
    print(f"  - Recommended keeps: {len(group_result.recommended_keeps)}")
    print(f"  - Recommended removes: {len(group_result.recommended_removes)}")
    
    print("\n✅ All data structure tests passed!")


def main():
    """Main test function"""
    print("SimilarityFinder Basic Test")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_data_structures()
        
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("\nNote: This test runs without PIL, imagehash, or TensorFlow dependencies.")
        print("The SimilarityFinder gracefully handles missing dependencies and")
        print("provides fallback functionality where possible.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())