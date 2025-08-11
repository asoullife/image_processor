"""
AI/ML Model Management System for Adobe Stock Image Processor

This module handles:
- TensorFlow models (ResNet50, VGG16) with GPU acceleration
- YOLO v8 integration for defect detection
- Model loading, caching, and memory optimization
- Fallback mechanisms when AI models are unavailable
- Performance mode configuration (Speed/Balanced/Smart)
"""

import os
import logging
import gc
import threading
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path
import json

try:
    # Lightweight alternatives - prioritize OpenCV and scikit-image
    import cv2
    import numpy as np
    from PIL import Image
    import psutil
    from skimage import feature, filters, measure, segmentation
    from skimage.metrics import structural_similarity as ssim
    import onnxruntime as ort
    import requests
    import hashlib
    CV2_AVAILABLE = True
    SKIMAGE_AVAILABLE = True
    ONNX_AVAILABLE = True
    PSUTIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Lightweight AI dependencies not available: {e}")
    cv2 = None
    np = None
    Image = None
    psutil = None
    feature = None
    filters = None
    measure = None
    segmentation = None
    ssim = None
    ort = None
    requests = None
    hashlib = None
    CV2_AVAILABLE = False
    SKIMAGE_AVAILABLE = False
    ONNX_AVAILABLE = False
    PSUTIL_AVAILABLE = False

# Optional heavy dependencies (only if available)
try:
    from transformers import pipeline, AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    pipeline = None
    AutoModel = None
    AutoTokenizer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for lightweight AI models"""
    name: str
    model_type: str  # 'onnx', 'opencv', 'transformers', 'traditional'
    model_path: Optional[str] = None
    download_url: Optional[str] = None
    input_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    use_cpu: bool = True  # Prioritize CPU for stability
    memory_limit: Optional[int] = None  # MB
    fallback_method: str = 'opencv'  # Always have OpenCV fallback


@dataclass
class PerformanceMode:
    """Performance mode configuration for lightweight processing"""
    name: str
    batch_size: int
    use_cpu_only: bool
    processing_precision: str  # 'float32', 'float16'
    memory_optimization: bool
    concurrent_models: int
    opencv_threads: int  # OpenCV thread count
    enable_onnx: bool  # Whether to use ONNX models


class AIModelManager:
    """
    Manages lightweight AI/ML models with CPU optimization and robust fallbacks
    
    Features:
    - Lightweight ONNX models for quality assessment
    - OpenCV-based computer vision algorithms
    - CPU-optimized processing (no GPU dependencies)
    - Progressive enhancement (OpenCV → ONNX → Transformers)
    - Robust fallback mechanisms
    - Memory-efficient batch processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Lightweight AI Model Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.performance_mode = "balanced"
        self.cpu_cores = None
        self.system_memory = None
        
        # Thread lock for model loading
        self._model_lock = threading.Lock()
        
        # Initialize system info
        self._initialize_system_info()
        
        # Setup lightweight model configurations
        self._setup_lightweight_model_configs()
        
        # Configure performance modes
        self._setup_performance_modes()
        
        # Initialize OpenCV optimizations
        self._setup_opencv_optimizations()
        
        # Re-check imports after initialization
        self._recheck_imports()
        
        logger.info(f"Lightweight AIModelManager initialized - "
                   f"OpenCV: {CV2_AVAILABLE}, ONNX: {ONNX_AVAILABLE}, "
                   f"Transformers: {TRANSFORMERS_AVAILABLE}, scikit-image: {SKIMAGE_AVAILABLE}")
    
    def _initialize_system_info(self):
        """Initialize system CPU and memory information"""
        try:
            # Get CPU information
            if PSUTIL_AVAILABLE:
                self.cpu_cores = psutil.cpu_count(logical=True)
                memory = psutil.virtual_memory()
                self.system_memory = memory.total // (1024**2)  # MB
                
                logger.info(f"System info - CPU cores: {self.cpu_cores}, "
                           f"RAM: {self.system_memory}MB ({memory.total // (1024**3)}GB)")
            else:
                # Fallback CPU detection
                import os
                self.cpu_cores = os.cpu_count() or 4
                self.system_memory = 8192  # Assume 8GB default
                logger.warning("psutil not available, using fallback system detection")
            
            # Check OpenCV build info
            if CV2_AVAILABLE:
                try:
                    import cv2
                    build_info = cv2.getBuildInformation()
                    if "Threading" in build_info:
                        logger.info("OpenCV threading support available")
                    if "OpenCL" in build_info:
                        logger.info("OpenCV OpenCL support available")
                except Exception as e:
                    logger.warning(f"Could not get OpenCV build info: {e}")
            
            # Check ONNX Runtime providers
            if ONNX_AVAILABLE:
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    logger.info(f"ONNX Runtime providers: {providers}")
                except Exception as e:
                    logger.warning(f"Could not get ONNX providers: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing system info: {e}")
            # Safe fallbacks
            self.cpu_cores = 4
            self.system_memory = 8192
    
    def _recheck_imports(self):
        """Re-check imports after initialization"""
        global CV2_AVAILABLE, ONNX_AVAILABLE, SKIMAGE_AVAILABLE, TRANSFORMERS_AVAILABLE
        
        try:
            import cv2
            import numpy as np
            CV2_AVAILABLE = True
        except ImportError:
            CV2_AVAILABLE = False
        
        try:
            import onnxruntime as ort
            ONNX_AVAILABLE = True
        except ImportError:
            ONNX_AVAILABLE = False
        
        try:
            from skimage import feature
            SKIMAGE_AVAILABLE = True
        except ImportError:
            SKIMAGE_AVAILABLE = False
        
        try:
            from transformers import pipeline
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    
    def _setup_lightweight_model_configs(self):
        """Setup configurations for lightweight AI models"""
        
        # ONNX models for quality analysis (lightweight alternatives)
        self.model_configs['mobilenet_onnx'] = ModelConfig(
            name='mobilenet_onnx',
            model_type='onnx',
            model_path='models/mobilenet_v2.onnx',
            download_url='https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
            input_size=(224, 224),
            batch_size=32,  # CPU can handle larger batches
            use_cpu=True,
            memory_limit=256,  # Much smaller
            fallback_method='opencv'
        )
        
        # Lightweight quality assessment model
        self.model_configs['quality_onnx'] = ModelConfig(
            name='quality_onnx',
            model_type='onnx',
            model_path='models/quality_assessment.onnx',
            input_size=(224, 224),
            batch_size=16,
            use_cpu=True,
            memory_limit=128,
            fallback_method='opencv'
        )
        
        # OpenCV-based defect detection (no external models needed)
        self.model_configs['opencv_defect'] = ModelConfig(
            name='opencv_defect',
            model_type='opencv',
            input_size=(512, 512),  # Smaller for faster processing
            batch_size=64,  # OpenCV is very fast
            use_cpu=True,
            memory_limit=64,
            fallback_method='traditional'
        )
        
        # Transformers model for similarity (optional, CPU inference)
        self.model_configs['clip_cpu'] = ModelConfig(
            name='clip_cpu',
            model_type='transformers',
            model_path='openai/clip-vit-base-patch32',
            input_size=(224, 224),
            batch_size=8,  # Conservative for CPU
            use_cpu=True,
            memory_limit=512,
            fallback_method='opencv'
        )
        
        # Traditional computer vision (always available)
        self.model_configs['traditional_cv'] = ModelConfig(
            name='traditional_cv',
            model_type='traditional',
            input_size=(512, 512),
            batch_size=128,  # Very fast
            use_cpu=True,
            memory_limit=32,
            fallback_method='basic'
        )
    
    def _setup_performance_modes(self):
        """Setup performance mode configurations for lightweight processing"""
        self.performance_modes = {
            'speed': PerformanceMode(
                name='speed',
                batch_size=64,  # Larger batches for OpenCV
                use_cpu_only=True,
                processing_precision='float32',
                memory_optimization=True,
                concurrent_models=1,  # Focus on speed
                opencv_threads=self.cpu_cores,
                enable_onnx=False  # OpenCV only for maximum speed
            ),
            'balanced': PerformanceMode(
                name='balanced',
                batch_size=32,
                use_cpu_only=True,
                processing_precision='float32',
                memory_optimization=True,
                concurrent_models=2,
                opencv_threads=max(2, self.cpu_cores // 2),
                enable_onnx=True  # Use ONNX models when available
            ),
            'smart': PerformanceMode(
                name='smart',
                batch_size=16,
                use_cpu_only=True,
                processing_precision='float32',
                memory_optimization=False,  # Keep models loaded
                concurrent_models=3,
                opencv_threads=max(1, self.cpu_cores // 4),
                enable_onnx=True  # Use all available models
            )
        }
    
    def _setup_opencv_optimizations(self):
        """Setup OpenCV optimizations for better performance"""
        if not CV2_AVAILABLE:
            return
        
        try:
            import cv2
            # Set OpenCV thread count
            cv2.setNumThreads(self.cpu_cores)
            
            # Enable OpenCL if available
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                logger.info("OpenCV OpenCL acceleration enabled")
            
            # Optimize for CPU cache
            cv2.setUseOptimized(True)
            
            logger.info(f"OpenCV optimized for {self.cpu_cores} threads")
            
        except Exception as e:
            logger.warning(f"OpenCV optimization failed: {e}")

    def set_performance_mode(self, mode: str):
        """
        Set performance mode for lightweight processing
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        if mode not in self.performance_modes:
            logger.warning(f"Unknown performance mode: {mode}, using 'balanced'")
            mode = 'balanced'
        
        self.performance_mode = mode
        perf_config = self.performance_modes[mode]
        
        # Update model batch sizes
        for model_config in self.model_configs.values():
            model_config.batch_size = perf_config.batch_size
            model_config.use_cpu = perf_config.use_cpu_only
        
        # Update OpenCV thread count
        if CV2_AVAILABLE:
            try:
                import cv2
                cv2.setNumThreads(perf_config.opencv_threads)
            except Exception as e:
                logger.warning(f"Failed to set OpenCV threads: {e}")
        
        logger.info(f"Performance mode set to: {mode} "
                   f"(batch_size: {perf_config.batch_size}, "
                   f"opencv_threads: {perf_config.opencv_threads})")
        
        # Clear models if memory optimization is enabled
        if perf_config.memory_optimization:
            self.clear_models()
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load AI model with caching and error handling
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if failed
        """
        with self._model_lock:
            # Return cached model if available
            if model_name in self.models:
                return self.models[model_name]
            
            # Check if model config exists
            if model_name not in self.model_configs:
                logger.error(f"Unknown model: {model_name}")
                return None
            
            model_config = self.model_configs[model_name]
            
            try:
                # Load model based on type (lightweight alternatives)
                if model_config.model_type == 'onnx':
                    model = self._load_onnx_model(model_config)
                elif model_config.model_type == 'opencv':
                    model = self._load_opencv_model(model_config)
                elif model_config.model_type == 'transformers':
                    model = self._load_transformers_model(model_config)
                elif model_config.model_type == 'traditional':
                    model = self._load_traditional_model(model_config)
                else:
                    logger.error(f"Unsupported model type: {model_config.model_type}")
                    return None
                
                if model is not None:
                    self.models[model_name] = model
                    logger.info(f"Successfully loaded model: {model_name}")
                    
                    # Memory cleanup after loading
                    self._cleanup_memory()
                    
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None
    
    def _load_onnx_model(self, config: ModelConfig) -> Optional[Any]:
        """Load ONNX model for lightweight inference"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available, falling back to OpenCV")
            return self._load_fallback_model(config)
        
        try:
            # Download model if not exists
            model_path = self._ensure_model_downloaded(config)
            if not model_path:
                return self._load_fallback_model(config)
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']  # CPU only for stability
            session = ort.InferenceSession(model_path, providers=providers)
            
            # Wrap in a simple interface
            model = {
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'output_name': session.get_outputs()[0].name,
                'input_shape': session.get_inputs()[0].shape,
                'config': config
            }
            
            logger.info(f"ONNX model {config.name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading ONNX model {config.name}: {e}")
            return self._load_fallback_model(config)
    
    def _load_opencv_model(self, config: ModelConfig) -> Optional[Any]:
        """Load OpenCV-based model (always available)"""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available")
            return None
        
        try:
            # OpenCV models are algorithm-based, not file-based
            model = {
                'type': 'opencv',
                'name': config.name,
                'config': config,
                'algorithms': self._setup_opencv_algorithms()
            }
            
            logger.info(f"OpenCV model {config.name} initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing OpenCV model {config.name}: {e}")
            return None
    
    def _load_transformers_model(self, config: ModelConfig) -> Optional[Any]:
        """Load Transformers model with CPU inference"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to OpenCV")
            return self._load_fallback_model(config)
        
        try:
            # Load model for CPU inference
            if config.name == 'clip_cpu':
                # Use pipeline for simplicity
                model = pipeline(
                    "feature-extraction",
                    model=config.model_path,
                    device=-1,  # CPU
                    framework="pt"
                )
            else:
                logger.error(f"Unknown Transformers model: {config.name}")
                return self._load_fallback_model(config)
            
            logger.info(f"Transformers model {config.name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading Transformers model {config.name}: {e}")
            return self._load_fallback_model(config)
    
    def _load_traditional_model(self, config: ModelConfig) -> Optional[Any]:
        """Load traditional computer vision model (always works)"""
        try:
            # Traditional CV is algorithm-based
            model = {
                'type': 'traditional',
                'name': config.name,
                'config': config,
                'methods': self._setup_traditional_methods()
            }
            
            logger.info(f"Traditional CV model {config.name} initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing traditional model {config.name}: {e}")
            return None
    
    def _load_fallback_model(self, config: ModelConfig) -> Optional[Any]:
        """Load fallback model based on config"""
        if config.fallback_method == 'opencv':
            return self._load_opencv_model(config)
        elif config.fallback_method == 'traditional':
            return self._load_traditional_model(config)
        else:
            logger.error(f"Unknown fallback method: {config.fallback_method}")
            return None
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get model (load if not cached)
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None
        """
        return self.load_model(model_name)
    
    def clear_models(self):
        """Clear all loaded models to free memory"""
        with self._model_lock:
            for model_name in list(self.models.keys()):
                try:
                    del self.models[model_name]
                    logger.debug(f"Cleared model: {model_name}")
                except Exception as e:
                    logger.error(f"Error clearing model {model_name}: {e}")
            
            self.models.clear()
            self._cleanup_memory()
            logger.info("All models cleared from memory")
    
    def _cleanup_memory(self):
        """Perform lightweight memory cleanup"""
        try:
            # Python garbage collection
            gc.collect()
            
            # PyTorch memory cleanup (if available)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear OpenCV cache
            if CV2_AVAILABLE:
                # OpenCV doesn't have explicit cache clearing, but we can force GC
                pass
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system and model status for lightweight processing
        
        Returns:
            Status information dictionary
        """
        status = {
            'cpu_cores': self.cpu_cores,
            'performance_mode': self.performance_mode,
            'loaded_models': list(self.models.keys()),
            'available_models': list(self.model_configs.keys()),
            'opencv_available': CV2_AVAILABLE,
            'onnx_available': ONNX_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'skimage_available': SKIMAGE_AVAILABLE,
            'processing_type': 'lightweight_cpu'
        }
        
        # Add OpenCV info
        if CV2_AVAILABLE:
            try:
                import cv2
                status['opencv_info'] = {
                    'version': cv2.__version__,
                    'threads': cv2.getNumThreads(),
                    'opencl_available': cv2.ocl.haveOpenCL(),
                    'optimized': cv2.useOptimized()
                }
            except Exception as e:
                logger.error(f"Error getting OpenCV info: {e}")
        
        # Add ONNX Runtime info
        if ONNX_AVAILABLE:
            try:
                import onnxruntime as ort
                status['onnx_info'] = {
                    'version': ort.__version__,
                    'providers': ort.get_available_providers()
                }
            except Exception as e:
                logger.error(f"Error getting ONNX info: {e}")
        
        # Add system memory info
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                status['system_info'] = {
                    'memory_total': f"{memory.total // (1024**3)}GB",
                    'memory_available': f"{memory.available // (1024**3)}GB",
                    'memory_used_percent': f"{memory.percent:.1f}%",
                    'cpu_usage': f"{cpu_percent:.1f}%",
                    'cpu_cores': self.cpu_cores
                }
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
        
        return status
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if lightweight model is available (can be loaded)
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model can be loaded
        """
        if model_name not in self.model_configs:
            return False
        
        config = self.model_configs[model_name]
        
        # Check if required framework is available
        if config.model_type == 'onnx' and not ONNX_AVAILABLE:
            # Check if fallback is available
            return self._is_fallback_available(config)
        elif config.model_type == 'opencv' and not CV2_AVAILABLE:
            return False
        elif config.model_type == 'transformers' and not TRANSFORMERS_AVAILABLE:
            return self._is_fallback_available(config)
        elif config.model_type == 'traditional' and not SKIMAGE_AVAILABLE:
            # Traditional can work with just OpenCV
            return CV2_AVAILABLE
        
        return True
    
    def _is_fallback_available(self, config: ModelConfig) -> bool:
        """Check if fallback method is available"""
        if config.fallback_method == 'opencv':
            return CV2_AVAILABLE
        elif config.fallback_method == 'traditional':
            return CV2_AVAILABLE or SKIMAGE_AVAILABLE
        return False
    
    def _ensure_model_downloaded(self, config: ModelConfig) -> Optional[str]:
        """Ensure ONNX model is downloaded"""
        if not config.model_path or not config.download_url:
            return None
        
        model_path = Path(config.model_path)
        
        # Create models directory if not exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download if not exists
        if not model_path.exists():
            try:
                logger.info(f"Downloading model {config.name} from {config.download_url}")
                
                if requests:
                    response = requests.get(config.download_url, stream=True)
                    response.raise_for_status()
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"Model {config.name} downloaded successfully")
                else:
                    logger.error("requests library not available for model download")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to download model {config.name}: {e}")
                return None
        
        return str(model_path)
    
    def _setup_opencv_algorithms(self) -> Dict[str, Any]:
        """Setup OpenCV algorithms for computer vision tasks"""
        algorithms = {}
        
        if CV2_AVAILABLE:
            try:
                # Edge detection
                algorithms['canny'] = cv2.Canny
                algorithms['sobel'] = cv2.Sobel
                
                # Feature detection
                algorithms['orb'] = cv2.ORB_create()
                algorithms['sift'] = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
                
                # Object detection (Haar cascades)
                algorithms['face_cascade'] = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                # Morphological operations
                algorithms['morphology'] = {
                    'open': cv2.MORPH_OPEN,
                    'close': cv2.MORPH_CLOSE,
                    'gradient': cv2.MORPH_GRADIENT
                }
                
                # Filters
                algorithms['gaussian_blur'] = cv2.GaussianBlur
                algorithms['bilateral_filter'] = cv2.bilateralFilter
                
                logger.info("OpenCV algorithms initialized successfully")
                
            except Exception as e:
                logger.error(f"Error setting up OpenCV algorithms: {e}")
        
        return algorithms
    
    def _setup_traditional_methods(self) -> Dict[str, Any]:
        """Setup traditional computer vision methods"""
        methods = {}
        
        if SKIMAGE_AVAILABLE:
            try:
                # Quality metrics
                methods['ssim'] = ssim
                methods['local_binary_pattern'] = feature.local_binary_pattern
                methods['hog'] = feature.hog
                
                # Filters
                methods['gaussian'] = filters.gaussian
                methods['sobel'] = filters.sobel
                methods['prewitt'] = filters.prewitt
                
                # Segmentation
                methods['watershed'] = segmentation.watershed
                methods['felzenszwalb'] = segmentation.felzenszwalb
                
                # Measurements
                methods['label'] = measure.label
                methods['regionprops'] = measure.regionprops
                
                logger.info("Traditional CV methods initialized successfully")
                
            except Exception as e:
                logger.error(f"Error setting up traditional methods: {e}")
        
        return methods

    def get_fallback_available(self) -> bool:
        """
        Check if OpenCV fallback is available
        
        Returns:
            True if OpenCV is available for fallback
        """
        return CV2_AVAILABLE and np is not None
    
    def optimize_for_batch_size(self, batch_size: int):
        """
        Optimize lightweight models for specific batch size
        
        Args:
            batch_size: Target batch size
        """
        # Update all model configs
        for config in self.model_configs.values():
            # For lightweight models, we can handle larger batch sizes
            if config.model_type in ['opencv', 'traditional']:
                config.batch_size = max(batch_size, config.batch_size)
            else:
                config.batch_size = min(batch_size, config.batch_size)
        
        # Clear models to force reload with new batch size
        if self.performance_modes[self.performance_mode].memory_optimization:
            self.clear_models()
        
        logger.info(f"Optimized lightweight models for batch size: {batch_size}")
    
    def preload_models(self, model_names: Optional[list] = None):
        """
        Preload lightweight models for faster inference
        
        Args:
            model_names: List of model names to preload, or None for recommended models
        """
        if model_names is None:
            # Default to most useful lightweight models
            model_names = ['opencv_defect', 'traditional_cv']
            
            # Add ONNX models if available
            if ONNX_AVAILABLE:
                model_names.append('mobilenet_onnx')
            
            # Add Transformers models if available and in smart mode
            if TRANSFORMERS_AVAILABLE and self.performance_mode == 'smart':
                model_names.append('clip_cpu')
        
        perf_mode = self.performance_modes[self.performance_mode]
        max_concurrent = perf_mode.concurrent_models
        
        loaded_count = 0
        for model_name in model_names:
            if loaded_count >= max_concurrent:
                logger.info(f"Reached concurrent model limit ({max_concurrent})")
                break
            
            if self.is_model_available(model_name):
                model = self.load_model(model_name)
                if model is not None:
                    loaded_count += 1
                    logger.info(f"Preloaded lightweight model: {model_name}")
        
        logger.info(f"Preloaded {loaded_count} lightweight models")
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended models based on current performance mode and available libraries"""
        recommended = []
        
        # Always recommend OpenCV and traditional (most stable)
        recommended.extend(['opencv_defect', 'traditional_cv'])
        
        # Add ONNX models if available
        if ONNX_AVAILABLE:
            recommended.append('mobilenet_onnx')
            if self.performance_mode in ['balanced', 'smart']:
                recommended.append('quality_onnx')
        
        # Add Transformers models only in smart mode
        if TRANSFORMERS_AVAILABLE and self.performance_mode == 'smart':
            recommended.append('clip_cpu')
        
        return recommended
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.clear_models()
        except:
            pass