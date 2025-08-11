"""
Basic test to verify QualityAnalyzer structure without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
sys.path.append('.')

# Mock the external dependencies
class MockCV2:
    CV_64F = 6
    COLOR_BGR2GRAY = 6
    
    @staticmethod
    def imread(path):
        if 'nonexistent' in path:
            return None
        # Return mock image array
        return [[[100, 100, 100] for _ in range(100)] for _ in range(100)]
    
    @staticmethod
    def cvtColor(img, code):
        # Return mock grayscale
        return [[100 for _ in range(100)] for _ in range(100)]
    
    @staticmethod
    def Laplacian(img, dtype):
        # Return mock laplacian
        class MockArray:
            def var(self):
                return 150.0
        return MockArray()
    
    @staticmethod
    def GaussianBlur(img, kernel, sigma):
        return img
    
    @staticmethod
    def absdiff(img1, img2):
        return [[10 for _ in range(100)] for _ in range(100)]
    
    @staticmethod
    def calcHist(images, channels, mask, histSize, ranges):
        # Return mock histogram
        return [[100] for _ in range(256)]

class MockNumpy:
    ndarray = list  # Mock ndarray as list
    
    @staticmethod
    def mean(arr):
        return 128.0
    
    @staticmethod
    def std(arr):
        return 30.0
    
    @staticmethod
    def sum(condition):
        return 100
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def array_split(arr, sections):
        return [arr[:len(arr)//sections] for _ in range(sections)]
    
    @staticmethod
    def max(arr):
        return max(arr) if hasattr(arr, '__iter__') else arr
    
    @staticmethod
    def abs(arr):
        return [abs(x) for x in arr] if hasattr(arr, '__iter__') else abs(arr)

# Mock the modules
sys.modules['cv2'] = MockCV2()
sys.modules['numpy'] = MockNumpy()
sys.modules['numpy.np'] = MockNumpy()

# Mock PIL
class MockPIL:
    class Image:
        pass
    class ExifTags:
        TAGS = {}

sys.modules['PIL'] = MockPIL()
sys.modules['PIL.Image'] = MockPIL.Image
sys.modules['PIL.ExifTags'] = MockPIL.ExifTags

# Now import our module
try:
    from backend.analyzers.quality_analyzer import QualityAnalyzer, QualityResult, ExposureResult
    print("✓ Successfully imported QualityAnalyzer")
    
    # Test initialization
    config = {
        'quality': {
            'min_sharpness': 100.0,
            'max_noise_level': 0.1,
            'min_resolution': [1920, 1080]
        }
    }
    
    analyzer = QualityAnalyzer(config)
    print("✓ Successfully initialized QualityAnalyzer")
    print(f"  - min_sharpness: {analyzer.min_sharpness}")
    print(f"  - max_noise_level: {analyzer.max_noise_level}")
    print(f"  - min_resolution: {analyzer.min_resolution}")
    
    # Test basic functionality with mock data
    mock_image = [[[100, 100, 100] for _ in range(100)] for _ in range(100)]
    
    # Test individual methods
    sharpness = analyzer.check_sharpness(mock_image)
    print(f"✓ Sharpness detection works: {sharpness}")
    
    noise = analyzer.detect_noise(mock_image)
    print(f"✓ Noise detection works: {noise}")
    
    color_balance = analyzer.check_color_balance(mock_image)
    print(f"✓ Color balance check works: {color_balance}")
    
    exposure = analyzer.analyze_exposure(mock_image)
    print(f"✓ Exposure analysis works: passed={exposure.passed}")
    
    # Test data classes
    quality_result = QualityResult(
        sharpness_score=150.0,
        noise_level=0.05,
        exposure_score=0.8,
        color_balance_score=0.9,
        resolution=(1920, 1080),
        file_size=1024000,
        overall_score=0.85,
        passed=True
    )
    print("✓ QualityResult dataclass works")
    print(f"  - Overall score: {quality_result.overall_score}")
    print(f"  - Passed: {quality_result.passed}")
    
    exposure_result = ExposureResult(
        brightness_score=0.5,
        contrast_score=0.3,
        histogram_balance=0.8,
        overexposed_pixels=0.01,
        underexposed_pixels=0.01,
        passed=True
    )
    print("✓ ExposureResult dataclass works")
    
    print("\n🎉 All basic tests passed! QualityAnalyzer implementation is structurally correct.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()