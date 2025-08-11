#!/usr/bin/env python3
"""Simple test for file protection functionality."""

import tempfile
import shutil
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_file_protection_basic():
    """Test basic file protection functionality."""
    print("Testing File Protection System...")
    print("=" * 40)
    
    temp_dir = tempfile.mkdtemp()
    success_count = 0
    total_tests = 0
    
    try:
        from core.file_protection import FileIntegrityProtector
        
        # Create test file
        test_file = Path(temp_dir) / "test.jpg"
        test_file.write_text("Test content for file protection")
        
        with FileIntegrityProtector(temp_dir) as protector:
            # Test 1: Hash calculation
            total_tests += 1
            try:
                hash1 = protector.calculate_file_hash(test_file)
                hash2 = protector.calculate_file_hash(test_file)
                assert hash1 == hash2
                assert len(hash1) == 64
                print("✅ Test 1: Hash calculation - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 1: Hash calculation - FAILED: {e}")
            
            # Test 2: Protected copy
            total_tests += 1
            try:
                protected_path = protector.create_protected_copy(test_file)
                assert Path(protected_path).exists()
                assert protected_path in protector.temp_files
                print("✅ Test 2: Protected copy - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 2: Protected copy - FAILED: {e}")
            
            # Test 3: Working copy
            total_tests += 1
            try:
                working_path = protector.create_safe_working_copy(test_file)
                assert Path(working_path).exists()
                assert working_path in protector.temp_files
                print("✅ Test 3: Working copy - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 3: Working copy - FAILED: {e}")
            
            # Test 4: Atomic copy
            total_tests += 1
            try:
                output_file = Path(temp_dir) / "output" / "atomic.jpg"
                success = protector.atomic_copy_to_output(test_file, output_file)
                assert success
                assert output_file.exists()
                print("✅ Test 4: Atomic copy - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 4: Atomic copy - FAILED: {e}")
            
            # Test 5: Integrity verification
            total_tests += 1
            try:
                original_hash = protector.calculate_file_hash(test_file)
                is_valid = protector.verify_file_integrity(test_file, original_hash)
                assert is_valid
                print("✅ Test 5: Integrity verification - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 5: Integrity verification - FAILED: {e}")
        
        print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_safe_operations_basic():
    """Test basic safe operations functionality."""
    print("\nTesting Safe Operations...")
    print("=" * 30)
    
    temp_dir = tempfile.mkdtemp()
    success_count = 0
    total_tests = 0
    
    try:
        from core.safe_operations import SafeImageProcessor
        
        # Create test files
        test_files = []
        for i in range(3):
            test_file = Path(temp_dir) / f"test_{i}.jpg"
            test_file.write_text(f"Test content {i}")
            test_files.append(test_file)
        
        with SafeImageProcessor(temp_dir) as processor:
            # Test 1: Metadata extraction
            total_tests += 1
            try:
                metadata = processor.safe_metadata_extraction(test_files[0])
                assert "file_path" in metadata
                assert "file_size" in metadata
                print("✅ Test 1: Metadata extraction - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 1: Metadata extraction - FAILED: {e}")
            
            # Test 2: Batch safe copy
            total_tests += 1
            try:
                output_dir = Path(temp_dir) / "batch_output"
                results = processor.batch_safe_copy(test_files, output_dir)
                assert len(results) == 3
                assert all(results.values())
                print("✅ Test 2: Batch safe copy - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 2: Batch safe copy - FAILED: {e}")
            
            # Test 3: Workspace creation
            total_tests += 1
            try:
                workspace_path = processor.create_processing_workspace(test_files[:2])
                assert Path(workspace_path).exists()
                workspace_files = list(Path(workspace_path).glob("*"))
                assert len(workspace_files) >= 2
                print("✅ Test 3: Workspace creation - PASSED")
                success_count += 1
            except Exception as e:
                print(f"❌ Test 3: Workspace creation - FAILED: {e}")
        
        print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_output_manager_basic():
    """Test basic output manager functionality."""
    print("\nTesting Output Manager...")
    print("=" * 25)
    
    temp_dir = tempfile.mkdtemp()
    success_count = 0
    total_tests = 0
    
    try:
        from core.output_manager import OutputStructureManager
        
        # Create input structure
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        (input_dir / "subdir").mkdir()
        (input_dir / "file1.jpg").write_text("content1")
        (input_dir / "subdir" / "file2.jpg").write_text("content2")
        
        output_dir = Path(temp_dir) / "output"
        
        manager = OutputStructureManager(input_dir, output_dir)
        
        # Test 1: Structure creation
        total_tests += 1
        try:
            structure_map = manager.create_mirrored_structure()
            assert len(structure_map) > 0
            assert (output_dir / "subdir").exists()
            print("✅ Test 1: Structure creation - PASSED")
            success_count += 1
        except Exception as e:
            print(f"❌ Test 1: Structure creation - FAILED: {e}")
        
        # Test 2: Path mapping
        total_tests += 1
        try:
            input_file = input_dir / "file1.jpg"
            output_path = manager.get_output_path(input_file)
            assert output_path is not None
            assert "output" in output_path
            print("✅ Test 2: Path mapping - PASSED")
            success_count += 1
        except Exception as e:
            print(f"❌ Test 2: Path mapping - FAILED: {e}")
        
        # Test 3: Statistics
        total_tests += 1
        try:
            stats = manager.get_structure_statistics()
            assert "total_mapped_paths" in stats
            assert stats["total_mapped_paths"] > 0
            print("✅ Test 3: Statistics - PASSED")
            success_count += 1
        except Exception as e:
            print(f"❌ Test 3: Statistics - FAILED: {e}")
        
        print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all tests."""
    print("🔒 File Integrity Protection System - Simple Tests")
    print("=" * 55)
    
    results = []
    
    # Run tests
    results.append(test_file_protection_basic())
    results.append(test_safe_operations_basic())
    results.append(test_output_manager_basic())
    
    # Summary
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"\n" + "=" * 55)
    print(f"📊 Overall Results: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("✅ All tests PASSED! File protection system is working correctly.")
        return True
    else:
        print("❌ Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)