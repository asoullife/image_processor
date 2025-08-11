#!/usr/bin/env python3
"""Demonstration script for FileManager functionality."""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
from backend.utils.file_manager import FileManager

def demo_file_manager():
    """Demonstrate FileManager functionality with test images."""
    print("🚀 FileManager Demonstration")
    print("=" * 50)
    
    # Initialize FileManager with configuration
    try:
        from backend.config.config_loader import get_config
        config = get_config()
        images_per_folder = config.output.images_per_folder
        print(f"Using configuration: {images_per_folder} images per folder")
    except:
        images_per_folder = 5  # Demo with smaller folders
        print(f"Using demo configuration: {images_per_folder} images per folder")
    
    file_manager = FileManager(images_per_folder=images_per_folder)
    
    # Input and output directories
    input_dir = 'backend/data/input'
    output_dir = 'backend/data/output'
    
    # Clean up previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned up previous output directory: {output_dir}")
    
    print(f"\n📁 Input directory: {os.path.abspath(input_dir)}")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    
    # Step 1: Scan for images
    print(f"\n🔍 Step 1: Scanning for images...")
    try:
        found_images = file_manager.scan_images(input_dir)
        print(f"✅ Found {len(found_images)} valid image files")
        
        # Show some examples
        print("\n📋 Sample found images:")
        for i, img_path in enumerate(found_images[:5]):
            rel_path = os.path.relpath(img_path, input_dir)
            print(f"  {i+1}. {rel_path}")
        
        if len(found_images) > 5:
            print(f"  ... and {len(found_images) - 5} more")
    
    except Exception as e:
        print(f"❌ Error scanning images: {e}")
        return
    
    # Step 2: Extract metadata from a few images
    print(f"\n📊 Step 2: Extracting metadata samples...")
    for i, img_path in enumerate(found_images[:3]):
        metadata = file_manager.get_image_metadata(img_path)
        filename = metadata['filename']
        dimensions = metadata['dimensions']
        file_size = metadata['file_size']
        format_type = metadata['format']
        
        print(f"  📷 {filename}")
        print(f"     Size: {dimensions[0]}x{dimensions[1]} pixels")
        print(f"     File size: {file_size:,} bytes")
        print(f"     Format: {format_type}")
    
    # Step 3: Organize images
    print(f"\n📦 Step 3: Organizing images into output folders...")
    try:
        results = file_manager.organize_output(found_images, output_dir)
        
        print(f"✅ Organization complete!")
        print(f"   📊 Total images: {results['total_images']}")
        print(f"   📁 Folders created: {results['folders_created']}")
        print(f"   ✅ Successful copies: {results['successful_copies']}")
        print(f"   ❌ Failed copies: {results['failed_copies']}")
        
        # Show folder structure
        print(f"\n📂 Folder structure:")
        for folder_num, images in results['folder_structure'].items():
            print(f"   📁 Folder {folder_num}: {len(images)} images")
            for img in images[:3]:  # Show first 3 images
                print(f"      📷 {img}")
            if len(images) > 3:
                print(f"      ... and {len(images) - 3} more")
    
    except Exception as e:
        print(f"❌ Error organizing images: {e}")
        return
    
    # Step 4: Validate output structure
    print(f"\n🔍 Step 4: Validating output structure...")
    try:
        validation = file_manager.validate_output_structure(output_dir)
        
        if validation['valid']:
            print(f"✅ Output structure is valid")
            print(f"   📊 Total images validated: {validation['total_images']}")
            print(f"   📁 Folders found: {len(validation['folders'])}")
            
            # Show detailed folder info
            for folder_num in sorted(validation['folders'].keys()):
                folder_info = validation['folders'][folder_num]
                print(f"   📁 Folder {folder_num}: {folder_info['image_count']} images")
        else:
            print(f"❌ Output structure validation failed: {validation['error']}")
    
    except Exception as e:
        print(f"❌ Error validating structure: {e}")
        return
    
    # Step 5: Show actual file system structure
    print(f"\n🗂️  Step 5: Actual file system structure:")
    try:
        for item in sorted(os.listdir(output_dir)):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                files_in_folder = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   📁 {item}/: {len(files_in_folder)} images")
                for file in files_in_folder[:2]:  # Show first 2 files
                    print(f"      📷 {file}")
                if len(files_in_folder) > 2:
                    print(f"      ... and {len(files_in_folder) - 2} more")
    
    except Exception as e:
        print(f"❌ Error reading file system: {e}")
    
    # Step 6: Test file integrity
    print(f"\n🔐 Step 6: Testing file integrity (sample)...")
    try:
        # Test integrity of first few copied files
        sample_count = min(3, len(found_images))
        for i in range(sample_count):
            original_path = found_images[i]
            filename = os.path.basename(original_path)
            
            # Find the copied file
            copied_path = None
            for folder_num in validation['folders']:
                potential_path = os.path.join(output_dir, str(folder_num), filename)
                if os.path.exists(potential_path):
                    copied_path = potential_path
                    break
                
                # Check for renamed file (conflict resolution)
                base_name = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]
                counter = 1
                while True:
                    renamed_file = f"{base_name}_{counter}{extension}"
                    potential_path = os.path.join(output_dir, str(folder_num), renamed_file)
                    if os.path.exists(potential_path):
                        copied_path = potential_path
                        break
                    counter += 1
                    if counter > 10:  # Reasonable limit
                        break
                
                if copied_path:
                    break
            
            if copied_path:
                original_hash = file_manager._calculate_file_hash(original_path)
                copied_hash = file_manager._calculate_file_hash(copied_path)
                
                if original_hash == copied_hash:
                    print(f"   ✅ {filename}: Integrity verified")
                else:
                    print(f"   ❌ {filename}: Integrity check failed")
            else:
                print(f"   ⚠️  {filename}: Copy not found")
    
    except Exception as e:
        print(f"❌ Error checking integrity: {e}")
    
    print(f"\n🎉 FileManager demonstration complete!")
    print(f"📁 Check the '{output_dir}' directory to see the organized images")
    print("=" * 50)

if __name__ == '__main__':
    demo_file_manager()