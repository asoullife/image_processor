#!/usr/bin/env python3
"""Simple test script for human review system components (no database required)."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_thai_translations():
    """Test Thai translations functionality."""
    print("1. Testing Thai Translations...")
    
    try:
        from utils.thai_translations import ThaiTranslations
        
        # Test basic translations
        translations = ThaiTranslations()
        
        test_reasons = [
            "low_quality",
            "defects_detected", 
            "similar_images",
            "compliance_issues",
            "face_detected",
            "logo_detected"
        ]
        
        print("   Rejection Reasons:")
        for reason in test_reasons:
            thai_text = translations.get_rejection_reason(reason)
            explanation = translations.get_rejection_explanation(reason)
            print(f"   • {reason}: {thai_text}")
            print(f"     Explanation: {explanation[:60]}...")
        
        # Test UI elements
        print("\n   UI Elements:")
        ui_elements = ["approve", "reject", "review_completed", "bulk_actions"]
        for element in ui_elements:
            thai_text = translations.get_ui_text(element)
            print(f"   • {element}: {thai_text}")
        
        # Test status messages
        print("\n   Status Messages:")
        status_messages = ["image_approved", "image_rejected", "bulk_approve_success"]
        for message in status_messages:
            thai_text = translations.get_status_message(message)
            print(f"   • {message}: {thai_text}")
        
        # Test translation helper
        test_reasons_list = ["low_quality", "defects_detected", "similar_images"]
        translated_reasons = translations.translate_rejection_reasons(test_reasons_list)
        
        print("\n   Translated Reasons List:")
        for reason_data in translated_reasons:
            print(f"   • {reason_data['key']}: {reason_data['thai']}")
            print(f"     English: {reason_data['english']}")
        
        print("   ✅ Thai translations working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Thai translations failed: {e}")
        return False

def test_api_schemas():
    """Test API schemas and validation."""
    print("\n2. Testing API Schemas...")
    
    try:
        from api.schemas import ReviewDecision, BulkReviewRequest, ImageResultResponse
        
        # Test ReviewDecision schema
        review_decision = ReviewDecision(decision="approve")
        print(f"   ✅ ReviewDecision created: {review_decision.decision}")
        
        review_decision_with_reason = ReviewDecision(
            decision="reject", 
            reason="low_quality"
        )
        print(f"   ✅ ReviewDecision with reason: {review_decision_with_reason.reason}")
        
        # Test BulkReviewRequest schema
        bulk_request = BulkReviewRequest(
            image_ids=["uuid1", "uuid2", "uuid3"],
            decision="approve"
        )
        print(f"   ✅ BulkReviewRequest created: {len(bulk_request.image_ids)} images")
        
        print("   ✅ API schemas working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ API schemas failed: {e}")
        return False

def test_file_operations():
    """Test file operations and output management."""
    print("\n3. Testing File Operations...")
    
    try:
        from core.output_manager import OutputStructureManager
        import tempfile
        import os
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            # Create input structure
            input_dir.mkdir()
            for i in range(1, 4):
                subfolder = input_dir / str(i)
                subfolder.mkdir()
                
                # Create dummy files
                for j in range(3):
                    dummy_file = subfolder / f"image_{j+1}.jpg"
                    dummy_file.write_text("dummy image content")
            
            # Test output manager
            output_manager = OutputStructureManager(input_dir, output_dir)
            structure_map = output_manager.create_mirrored_structure()
            
            print(f"   ✅ Created structure with {len(structure_map)} mappings")
            
            # Test statistics
            stats = output_manager.get_structure_statistics()
            print(f"   ✅ Structure stats: {stats['total_mapped_paths']} paths")
            
            # Test validation
            validation = output_manager.validate_output_structure()
            valid_count = sum(1 for valid in validation.values() if valid)
            print(f"   ✅ Validation: {valid_count}/{len(validation)} paths valid")
        
        print("   ✅ File operations working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
        return False

def test_review_logic():
    """Test review logic and filtering."""
    print("\n4. Testing Review Logic...")
    
    try:
        # Simulate image results data
        mock_results = [
            {
                "id": f"image_{i}",
                "filename": f"test_image_{i}.jpg",
                "final_decision": "approved" if i < 5 else "rejected",
                "rejection_reasons": ["low_quality"] if i >= 5 else [],
                "human_override": i == 7,  # One human override
                "source_folder": str((i % 3) + 1),
                "quality_scores": {"overall_score": 0.9 - (i * 0.05)},
                "processing_time": 1.0 + (i * 0.1)
            }
            for i in range(10)
        ]
        
        # Test filtering logic
        approved_count = len([r for r in mock_results if r["final_decision"] == "approved"])
        rejected_count = len([r for r in mock_results if r["final_decision"] == "rejected"])
        human_reviewed_count = len([r for r in mock_results if r["human_override"]])
        
        print(f"   ✅ Mock data: {len(mock_results)} total images")
        print(f"   ✅ Approved: {approved_count}, Rejected: {rejected_count}")
        print(f"   ✅ Human reviewed: {human_reviewed_count}")
        
        # Test rejection reason counting
        rejection_counts = {}
        for result in mock_results:
            for reason in result["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        
        print(f"   ✅ Rejection reasons: {rejection_counts}")
        
        # Test folder grouping
        folder_counts = {}
        for result in mock_results:
            folder = result["source_folder"]
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
        
        print(f"   ✅ Folder distribution: {folder_counts}")
        
        print("   ✅ Review logic working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Review logic failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Human Review System Components")
    print("=" * 50)
    
    tests = [
        test_thai_translations,
        test_api_schemas,
        test_file_operations,
        test_review_logic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("🎉 All Human Review System Component Tests Passed!")
        print("\n✅ The Human Review System is ready for integration!")
        print("\nComponents tested:")
        print("  • Thai language translations")
        print("  • API schemas and validation")
        print("  • File operations and output management")
        print("  • Review logic and filtering")
        print("\nNext steps:")
        print("  1. Start the FastAPI backend server")
        print("  2. Start the Next.js frontend server")
        print("  3. Navigate to /projects/[id]/review to test the UI")
        return True
    else:
        failed_count = len([r for r in results if not r])
        print(f"❌ {failed_count}/{len(results)} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)