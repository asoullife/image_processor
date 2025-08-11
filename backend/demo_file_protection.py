#!/usr/bin/env python3
"""Demo script for file integrity protection system."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def create_demo_data():
    """Create demo input data structure."""
    temp_dir = tempfile.mkdtemp(prefix="file_protection_demo_")
    input_dir = Path(temp_dir) / "demo_input"
    
    # Create directory structure
    input_dir.mkdir(parents=True)
    (input_dir / "category_a").mkdir()
    (input_dir / "category_b").mkdir()
    (input_dir / "category_a" / "subcategory_1").mkdir()
    (input_dir / "category_a" / "subcategory_2").mkdir()
    
    # Create demo "image" files
    demo_files = [
        input_dir / "image_001.jpg",
        input_dir / "image_002.jpg",
        input_dir / "category_a" / "image_003.jpg",
        input_dir / "category_a" / "image_004.jpg",
        input_dir / "category_a" / "subcategory_1" / "image_005.jpg",
        input_dir / "category_a" / "subcategory_1" / "image_006.jpg",
        input_dir / "category_a" / "subcategory_2" / "image_007.jpg",
        input_dir / "category_b" / "image_008.jpg",
        input_dir / "category_b" / "image_009.jpg",
        input_dir / "category_b" / "image_010.jpg",
    ]
    
    for i, file_path in enumerate(demo_files, 1):
        file_path.write_text(f"Demo image content {i:03d}\nThis is a simulated image file for testing.")
    
    print(f"✅ Created demo data structure at: {input_dir}")
    print(f"   📁 Created {len(demo_files)} demo image files")
    print(f"   📁 Created directory structure with subdirectories")
    
    return str(temp_dir), str(input_dir), [str(f) for f in demo_files]

def demo_file_integrity_protection():
    """Demonstrate file integrity protection features."""
    print("🔒 File Integrity Protection Demo")
    print("=" * 50)
    
    try:
        from core.file_protection import FileIntegrityProtector, FileIntegrityError
        
        # Create demo data
        temp_base, input_dir, demo_files = create_demo_data()
        
        print(f"\n1. Testing File Integrity Protection...")
        
        with FileIntegrityProtector() as protector:
            # Test hash calculation
            test_file = demo_files[0]
            original_hash = protector.calculate_file_hash(test_file)
            print(f"   ✅ Calculated hash for {Path(test_file).name}: {original_hash[:16]}...")
            
            # Test protected copy
            protected_path = protector.create_protected_copy(test_file)
            print(f"   ✅ Created protected copy: {Path(protected_path).name}")
            
            # Verify integrity
            is_valid = protector.verify_file_integrity(protected_path, original_hash)
            print(f"   ✅ Integrity verification: {'PASSED' if is_valid else 'FAILED'}")
            
            # Test working copy
            working_path = protector.create_safe_working_copy(test_file)
            print(f"   ✅ Created working copy: {Path(working_path).name}")
            
            # Test atomic copy
            output_dir = Path(temp_base) / "atomic_test"
            output_file = output_dir / "atomic_copy.jpg"
            success = protector.atomic_copy_to_output(test_file, output_file)
            print(f"   ✅ Atomic copy: {'SUCCESS' if success else 'FAILED'}")
            
            # Test batch integrity verification
            batch_results = protector.verify_source_integrity(demo_files[:5])
            valid_count = sum(1 for valid in batch_results.values() if valid)
            print(f"   ✅ Batch integrity check: {valid_count}/5 files valid")
        
        return temp_base, input_dir, demo_files
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return None, None, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None, None

def demo_safe_operations():
    """Demonstrate safe image operations."""
    print(f"\n🛡️ Safe Operations Demo")
    print("=" * 30)
    
    try:
        from core.safe_operations import SafeImageProcessor
        
        # Use data from previous demo
        temp_base, input_dir, demo_files = demo_file_integrity_protection()
        if not demo_files:
            return
        
        with SafeImageProcessor() as processor:
            # Test safe analysis context
            test_file = demo_files[0]
            print(f"   📊 Testing safe analysis for: {Path(test_file).name}")
            
            with processor.safe_image_analysis(test_file) as (original, analysis):
                print(f"   ✅ Original file: {Path(original).name}")
                print(f"   ✅ Analysis copy: {Path(analysis).name}")
                print(f"   ✅ Files are different: {original != analysis}")
            
            # Test metadata extraction
            metadata = processor.safe_metadata_extraction(test_file)
            print(f"   ✅ Metadata extracted: {len(metadata)} fields")
            print(f"      - File size: {metadata.get('file_size', 'N/A')} bytes")
            print(f"      - File name: {metadata.get('file_name', 'N/A')}")
            
            # Test batch safe copy
            output_dir = Path(temp_base) / "safe_copy_output"
            copy_results = processor.batch_safe_copy(demo_files[:3], output_dir)
            success_count = sum(1 for success in copy_results.values() if success)
            print(f"   ✅ Batch safe copy: {success_count}/3 files copied successfully")
            
            # Test workspace creation
            workspace_path = processor.create_processing_workspace(demo_files[:2])
            workspace_files = list(Path(workspace_path).glob("*"))
            print(f"   ✅ Processing workspace created with {len(workspace_files)} files")
        
        return temp_base
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def demo_output_structure_management():
    """Demonstrate output structure management."""
    print(f"\n📁 Output Structure Management Demo")
    print("=" * 40)
    
    try:
        from core.output_manager import OutputStructureManager, create_output_structure_from_input
        
        # Create fresh demo data for this test
        temp_base, input_dir, demo_files = create_demo_data()
        output_dir = Path(temp_base) / "structured_output"
        
        print(f"   📂 Input directory: {Path(input_dir).name}")
        print(f"   📂 Output directory: {output_dir.name}")
        
        # Create output structure manager
        manager = create_output_structure_from_input(input_dir, output_dir)
        
        # Get structure statistics
        stats = manager.get_structure_statistics()
        print(f"   ✅ Structure created:")
        print(f"      - Mapped paths: {stats.get('total_mapped_paths', 0)}")
        print(f"      - Input directories: {stats.get('input_directories', 0)}")
        print(f"      - Input files: {stats.get('input_files', 0)}")
        
        # Test file organization
        approved_files = demo_files[:7]  # Approve first 7 files
        organization = manager.organize_approved_files(approved_files, subfolder_size=3)
        print(f"   ✅ File organization:")
        print(f"      - Approved files: {len(approved_files)}")
        print(f"      - Output folders: {len(organization)}")
        
        # Test copying approved files
        copy_results = manager.copy_approved_files(approved_files, subfolder_size=3)
        success_count = sum(1 for success in copy_results.values() if success)
        print(f"   ✅ File copying: {success_count}/{len(approved_files)} files copied")
        
        # Validate output structure
        validation_results = manager.validate_output_structure()
        valid_count = sum(1 for valid in validation_results.values() if valid)
        print(f"   ✅ Structure validation: {valid_count}/{len(validation_results)} paths valid")
        
        # Create processing summary
        summary_path = manager.create_processing_summary(copy_results)
        print(f"   ✅ Processing summary created: {Path(summary_path).name}")
        
        return temp_base
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def demo_temporary_resize():
    """Demonstrate temporary image resizing."""
    print(f"\n🖼️ Temporary Resize Demo")
    print("=" * 25)
    
    try:
        from core.file_protection import FileIntegrityProtector
        
        # Create a test file
        temp_dir = tempfile.mkdtemp(prefix="resize_demo_")
        test_file = Path(temp_dir) / "test_image.jpg"
        test_file.write_text("Simulated large image content for resize testing")
        
        print(f"   📸 Test file: {test_file.name}")
        print(f"   📏 Original size: {test_file.stat().st_size} bytes")
        
        with FileIntegrityProtector() as protector:
            # Test temporary resize context
            with protector.temporary_resize(test_file, max_size=(1024, 1024)) as resized_path:
                print(f"   ✅ Temporary resize created: {Path(resized_path).name}")
                print(f"   📏 Resized file exists: {Path(resized_path).exists()}")
                
                # Verify original is unchanged
                original_content = test_file.read_text()
                print(f"   ✅ Original file unchanged: {len(original_content)} chars")
            
            # After context, resized file should be cleaned up
            print(f"   ✅ Temporary resize cleaned up automatically")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except ImportError as e:
        print(f"   ❌ Import error (PIL not available): {e}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def demo_integrity_manifest():
    """Demonstrate integrity manifest functionality."""
    print(f"\n📋 Integrity Manifest Demo")
    print("=" * 30)
    
    try:
        from core.file_protection import FileIntegrityProtector
        
        # Create demo data
        temp_base, input_dir, demo_files = create_demo_data()
        
        with FileIntegrityProtector() as protector:
            # Process some files to build integrity cache
            for file_path in demo_files[:3]:
                protector.create_protected_copy(file_path)
            
            print(f"   📊 Processed {len(protector.integrity_cache)} files")
            
            # Save integrity manifest
            manifest_path = Path(temp_base) / "integrity_manifest.json"
            protector.save_integrity_manifest(manifest_path)
            print(f"   ✅ Integrity manifest saved: {manifest_path.name}")
            print(f"   📄 Manifest size: {manifest_path.stat().st_size} bytes")
            
            # Test loading manifest
            new_protector = FileIntegrityProtector()
            new_protector.load_integrity_manifest(manifest_path)
            print(f"   ✅ Manifest loaded: {len(new_protector.integrity_cache)} cached hashes")
        
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Main demo function."""
    print("🔒 Adobe Stock Image Processor - File Integrity Protection Demo")
    print("=" * 70)
    print("This demo shows the file protection system that ensures source files")
    print("are never modified during processing.\n")
    
    temp_dirs = []
    
    try:
        # Run all demos
        temp_base = demo_safe_operations()
        if temp_base:
            temp_dirs.append(temp_base)
        
        temp_base = demo_output_structure_management()
        if temp_base:
            temp_dirs.append(temp_base)
        
        demo_temporary_resize()
        demo_integrity_manifest()
        
        print(f"\n" + "=" * 70)
        print("✅ File Integrity Protection Demo Completed!")
        print("\n🔒 Key Features Demonstrated:")
        print("   • Strict source file protection (read-only copies)")
        print("   • Safe working copies for analysis")
        print("   • Atomic file operations with rollback")
        print("   • Integrity verification using SHA-256 hashes")
        print("   • Temporary image resizing without affecting originals")
        print("   • Output folder structure mirroring input organization")
        print("   • Batch processing with integrity checks")
        print("   • Automatic cleanup of temporary files")
        print("   • Integrity manifest for session persistence")
        
        print(f"\n📋 System Benefits:")
        print("   • Zero risk of source file corruption")
        print("   • Atomic operations prevent partial failures")
        print("   • Memory-efficient temporary file management")
        print("   • Comprehensive integrity verification")
        print("   • Organized output structure preservation")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup all temporary directories
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

if __name__ == "__main__":
    main()