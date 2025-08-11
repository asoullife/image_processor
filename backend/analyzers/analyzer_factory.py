"""Factory for creating analyzer instances."""

import logging
from typing import Dict, Any, Optional

try:
    from ..config.config_loader import AppConfig
    from .quality_analyzer import QualityAnalyzer
    from .defect_detector import DefectDetector
    from .similarity_finder import SimilarityFinder
    from .compliance_checker import ComplianceChecker
except ImportError:
    # Fallback for testing without full backend structure
    AppConfig = None
    QualityAnalyzer = None
    DefectDetector = None
    SimilarityFinder = None
    ComplianceChecker = None

# AI-enhanced analyzers
from .ai_model_manager import AIModelManager
from .ai_quality_analyzer import AIQualityAnalyzer
from .ai_defect_detector import AIDefectDetector
from .ai_similarity_finder import AISimilarityFinder
from .ai_compliance_checker import AIComplianceChecker
from .unified_ai_analyzer import UnifiedAIAnalyzer

logger = logging.getLogger(__name__)

class AnalyzerFactory:
    """Factory for creating and managing analyzer instances."""
    
    def __init__(self, config: AppConfig):
        """Initialize analyzer factory.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Convert config to dict format for analyzers
        self.config_dict = self._convert_config_to_dict(config)
        
        # Cache for analyzer instances
        self._analyzers: Dict[str, Any] = {}
        
        # AI model manager (shared across AI analyzers)
        self._ai_model_manager: Optional[AIModelManager] = None
    
    def _convert_config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to dictionary format for analyzers.
        
        Args:
            config: Application configuration
            
        Returns:
            Configuration dictionary
        """
        return {
            'processing': {
                'batch_size': config.processing.batch_size,
                'max_workers': config.processing.max_workers,
                'checkpoint_interval': config.processing.checkpoint_interval
            },
            'output': {
                'images_per_folder': config.output.images_per_folder,
                'preserve_metadata': config.output.preserve_metadata,
                'generate_thumbnails': config.output.generate_thumbnails
            },
            'quality': {
                'min_sharpness': config.quality.min_sharpness,
                'max_noise_level': config.quality.max_noise_level,
                'min_resolution': config.quality.min_resolution
            },
            'similarity': {
                'hash_threshold': config.similarity.hash_threshold,
                'feature_threshold': config.similarity.feature_threshold,
                'clustering_eps': config.similarity.clustering_eps
            },
            'compliance': {
                'logo_detection_confidence': config.compliance.logo_detection_confidence,
                'face_detection_enabled': config.compliance.face_detection_enabled,
                'metadata_validation': config.compliance.metadata_validation
            },
            'decision': {
                'quality_weight': config.decision.quality_weight,
                'defect_weight': config.decision.defect_weight,
                'similarity_weight': config.decision.similarity_weight,
                'compliance_weight': config.decision.compliance_weight,
                'technical_weight': config.decision.technical_weight,
                'approval_threshold': config.decision.approval_threshold,
                'rejection_threshold': config.decision.rejection_threshold,
                'quality_min_threshold': config.decision.quality_min_threshold,
                'defect_max_threshold': config.decision.defect_max_threshold,
                'similarity_max_threshold': config.decision.similarity_max_threshold,
                'compliance_min_threshold': config.decision.compliance_min_threshold,
                'quality_critical_threshold': config.decision.quality_critical_threshold,
                'defect_critical_threshold': config.decision.defect_critical_threshold,
                'compliance_critical_threshold': config.decision.compliance_critical_threshold
            }
        }
    
    def get_quality_analyzer(self) -> QualityAnalyzer:
        """Get quality analyzer instance.
        
        Returns:
            QualityAnalyzer instance
        """
        if 'quality' not in self._analyzers:
            self._analyzers['quality'] = QualityAnalyzer(self.config_dict)
            self.logger.info("Created QualityAnalyzer instance")
        
        return self._analyzers['quality']
    
    def get_defect_detector(self) -> DefectDetector:
        """Get defect detector instance.
        
        Returns:
            DefectDetector instance
        """
        if 'defect' not in self._analyzers:
            self._analyzers['defect'] = DefectDetector(self.config_dict)
            self.logger.info("Created DefectDetector instance")
        
        return self._analyzers['defect']
    
    def get_similarity_finder(self) -> SimilarityFinder:
        """Get similarity finder instance.
        
        Returns:
            SimilarityFinder instance
        """
        if 'similarity' not in self._analyzers:
            self._analyzers['similarity'] = SimilarityFinder(self.config_dict)
            self.logger.info("Created SimilarityFinder instance")
        
        return self._analyzers['similarity']
    
    def get_compliance_checker(self) -> ComplianceChecker:
        """Get compliance checker instance.
        
        Returns:
            ComplianceChecker instance
        """
        if 'compliance' not in self._analyzers:
            self._analyzers['compliance'] = ComplianceChecker(self.config_dict)
            self.logger.info("Created ComplianceChecker instance")
        
        return self._analyzers['compliance']
    
    def get_ai_model_manager(self) -> AIModelManager:
        """Get AI model manager instance.
        
        Returns:
            AIModelManager instance
        """
        if self._ai_model_manager is None:
            self._ai_model_manager = AIModelManager(self.config_dict)
            self.logger.info("Created AIModelManager instance")
        
        return self._ai_model_manager
    
    def get_ai_quality_analyzer(self) -> AIQualityAnalyzer:
        """Get AI quality analyzer instance.
        
        Returns:
            AIQualityAnalyzer instance
        """
        if 'ai_quality' not in self._analyzers:
            model_manager = self.get_ai_model_manager()
            self._analyzers['ai_quality'] = AIQualityAnalyzer(self.config_dict, model_manager)
            self.logger.info("Created AIQualityAnalyzer instance")
        
        return self._analyzers['ai_quality']
    
    def get_ai_defect_detector(self) -> AIDefectDetector:
        """Get AI defect detector instance.
        
        Returns:
            AIDefectDetector instance
        """
        if 'ai_defect' not in self._analyzers:
            model_manager = self.get_ai_model_manager()
            self._analyzers['ai_defect'] = AIDefectDetector(self.config_dict, model_manager)
            self.logger.info("Created AIDefectDetector instance")
        
        return self._analyzers['ai_defect']
    
    def get_ai_similarity_finder(self) -> AISimilarityFinder:
        """Get AI similarity finder instance.
        
        Returns:
            AISimilarityFinder instance
        """
        if 'ai_similarity' not in self._analyzers:
            model_manager = self.get_ai_model_manager()
            self._analyzers['ai_similarity'] = AISimilarityFinder(self.config_dict, model_manager)
            self.logger.info("Created AISimilarityFinder instance")
        
        return self._analyzers['ai_similarity']
    
    def get_ai_compliance_checker(self) -> AIComplianceChecker:
        """Get AI compliance checker instance.
        
        Returns:
            AIComplianceChecker instance
        """
        if 'ai_compliance' not in self._analyzers:
            model_manager = self.get_ai_model_manager()
            self._analyzers['ai_compliance'] = AIComplianceChecker(self.config_dict, model_manager)
            self.logger.info("Created AIComplianceChecker instance")
        
        return self._analyzers['ai_compliance']
    
    def get_unified_ai_analyzer(self) -> UnifiedAIAnalyzer:
        """Get unified AI analyzer instance.
        
        Returns:
            UnifiedAIAnalyzer instance
        """
        if 'unified_ai' not in self._analyzers:
            self._analyzers['unified_ai'] = UnifiedAIAnalyzer(self.config_dict)
            self.logger.info("Created UnifiedAIAnalyzer instance")
        
        return self._analyzers['unified_ai']
    
    def get_all_analyzers(self) -> Dict[str, Any]:
        """Get all analyzer instances.
        
        Returns:
            Dictionary of analyzer instances
        """
        return {
            'quality': self.get_quality_analyzer(),
            'defect': self.get_defect_detector(),
            'similarity': self.get_similarity_finder(),
            'compliance': self.get_compliance_checker()
        }
    
    def get_all_ai_analyzers(self) -> Dict[str, Any]:
        """Get all AI analyzer instances.
        
        Returns:
            Dictionary of AI analyzer instances
        """
        return {
            'ai_model_manager': self.get_ai_model_manager(),
            'ai_quality': self.get_ai_quality_analyzer(),
            'ai_defect': self.get_ai_defect_detector(),
            'ai_similarity': self.get_ai_similarity_finder(),
            'ai_compliance': self.get_ai_compliance_checker(),
            'unified_ai': self.get_unified_ai_analyzer()
        }
    
    def reset_analyzers(self):
        """Reset all analyzer instances (useful for configuration changes)."""
        self._analyzers.clear()
        self.logger.info("Reset all analyzer instances")
    
    def update_config(self, new_config: AppConfig):
        """Update configuration and reset analyzers.
        
        Args:
            new_config: New application configuration
        """
        self.config = new_config
        self.config_dict = self._convert_config_to_dict(new_config)
        self.reset_analyzers()
        self.logger.info("Updated configuration and reset analyzers")
    
    def set_performance_mode(self, mode: str):
        """Set performance mode for all AI analyzers.
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        try:
            # Update AI model manager
            if self._ai_model_manager is not None:
                self._ai_model_manager.set_performance_mode(mode)
            
            # Update AI analyzers
            ai_analyzers = ['ai_quality', 'ai_defect', 'ai_similarity', 'ai_compliance', 'unified_ai']
            for analyzer_name in ai_analyzers:
                if analyzer_name in self._analyzers:
                    analyzer = self._analyzers[analyzer_name]
                    if hasattr(analyzer, 'set_performance_mode'):
                        analyzer.set_performance_mode(mode)
            
            self.logger.info(f"Set performance mode to {mode} for all AI analyzers")
            
        except Exception as e:
            self.logger.error(f"Error setting performance mode: {e}")
    
    def preload_ai_models(self):
        """Preload AI models for faster processing."""
        try:
            model_manager = self.get_ai_model_manager()
            model_manager.preload_models(['resnet50', 'yolov8n'])
            
            # Also preload CLIP model if similarity finder is available
            if 'ai_similarity' in self._analyzers:
                similarity_finder = self._analyzers['ai_similarity']
                if hasattr(similarity_finder, '_load_clip_model'):
                    similarity_finder._load_clip_model()
            
            self.logger.info("AI models preloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error preloading AI models: {e}")
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of all analyzers.
        
        Returns:
            Status information for all analyzers
        """
        status = {
            'initialized_analyzers': list(self._analyzers.keys()),
            'available_analyzers': ['quality', 'defect', 'similarity', 'compliance'],
            'available_ai_analyzers': ['ai_quality', 'ai_defect', 'ai_similarity', 'ai_compliance', 'unified_ai'],
            'config_loaded': self.config is not None,
            'factory_ready': True,
            'ai_model_manager_initialized': self._ai_model_manager is not None
        }
        
        # Test each initialized analyzer
        for analyzer_name, analyzer in self._analyzers.items():
            try:
                # Basic health check - ensure analyzer has required methods
                if hasattr(analyzer, 'analyze') or hasattr(analyzer, 'detect_defects') or hasattr(analyzer, 'check_compliance') or hasattr(analyzer, 'find_similar_groups'):
                    status[f'{analyzer_name}_status'] = 'healthy'
                else:
                    status[f'{analyzer_name}_status'] = 'missing_methods'
            except Exception as e:
                status[f'{analyzer_name}_status'] = f'error: {str(e)}'
        
        # Add AI model manager status if available
        if self._ai_model_manager is not None:
            try:
                status['ai_model_manager_status'] = self._ai_model_manager.get_system_status()
            except Exception as e:
                status['ai_model_manager_status'] = f'error: {str(e)}'
        
        return status