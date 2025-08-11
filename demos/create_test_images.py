#!/usr/bin/env python3
"""Create test images for demonstrating FileManager functionality."""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from PIL import Image, ImageDraw, ImageFont

def create_test_images():
    """Create a variety of test images in test_input directory."""
    test_input_dir = 'backend/data/input'
    
    # Create test_input directory if it doesn't exist
    if not os.path.exists(test_input_dir):
        os.makedirs(test_input_dir)
    
    print(f"Creating test images in {test_input_dir}...")
    
    # Create main directory images
    for i in range(8):
        # Create image with different colors and sizes
        width = 200 + (i * 50)
        height = 150 + (i * 30)
        color = (i * 30 % 255, (i * 50) % 255, (i * 70) % 255)
        
        img = Image.new('RGB', (width, height), color=color)
        
        # Add some text to make images distinguishable
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Test Image {i+1}"
        if font:
            draw.text((10, 10), text, fill='white', font=font)
        else:
            draw.text((10, 10), text, fill='white')
        
        # Save as different formats
        if i % 3 == 0:
            img_path = os.path.join(test_input_dir, f'main_image_{i+1:02d}.jpg')
            img.save(img_path, 'JPEG', quality=95)
        elif i % 3 == 1:
            img_path = os.path.join(test_input_dir, f'main_image_{i+1:02d}.png')
            img.save(img_path, 'PNG')
        else:
            img_path = os.path.join(test_input_dir, f'main_image_{i+1:02d}.jpeg')
            img.save(img_path, 'JPEG', quality=90)
        
        print(f"Created: {img_path}")
    
    # Create subdirectories with images
    for sub_num in range(2):
        sub_dir = os.path.join(test_input_dir, f'category_{sub_num + 1}')
        os.makedirs(sub_dir, exist_ok=True)
        
        for i in range(4):
            img = Image.new('RGB', (180, 180), color=(sub_num * 100, i * 60, 200))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            text = f"Cat{sub_num+1} Img{i+1}"
            if font:
                draw.text((10, 10), text, fill='white', font=font)
            else:
                draw.text((10, 10), text, fill='white')
            
            img_path = os.path.join(sub_dir, f'category_{sub_num+1}_image_{i+1}.jpg')
            img.save(img_path, 'JPEG', quality=85)
            print(f"Created: {img_path}")
    
    # Create nested subdirectory
    nested_dir = os.path.join(test_input_dir, 'special', 'nested')
    os.makedirs(nested_dir, exist_ok=True)
    
    nested_img = Image.new('RGB', (250, 200), color='purple')
    draw = ImageDraw.Draw(nested_img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text = "Nested Image"
    if font:
        draw.text((10, 10), text, fill='white', font=font)
    else:
        draw.text((10, 10), text, fill='white')
    
    nested_path = os.path.join(nested_dir, 'nested_special.png')
    nested_img.save(nested_path, 'PNG')
    print(f"Created: {nested_path}")
    
    # Create some non-image files to test filtering
    text_file = os.path.join(test_input_dir, 'readme.txt')
    with open(text_file, 'w') as f:
        f.write('This is a text file that should be ignored by the scanner.')
    print(f"Created: {text_file}")
    
    # Create a fake image file (wrong content)
    fake_img = os.path.join(test_input_dir, 'fake_image.jpg')
    with open(fake_img, 'w') as f:
        f.write('This is not actually image data')
    print(f"Created: {fake_img}")
    
    print(f"\n✅ Test image creation complete!")
    print(f"Created images in: {os.path.abspath(test_input_dir)}")
    
    # Count total valid images
    from backend.utils.file_manager import FileManager
    fm = FileManager()
    found_images = fm.scan_images(test_input_dir)
    print(f"Total valid images that will be found: {len(found_images)}")

if __name__ == '__main__':
    create_test_images()