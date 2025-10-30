"""
Metrics Collection and Monitoring

This module provides comprehensive monitoring for the Enlitens pipeline:
- GPU temperature and memory usage
- Processing speed and quality metrics
- Error tracking and analysis
- Performance optimization insights
"""

import time
import logging
import psutil
import pynvml
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

from src.monitoring.observability import get_observability


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_temperature: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_utilization: float
    disk_usage_percent: float


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    timestamp: datetime
    documents_processed: int
    documents_per_hour: float
    average_quality_score: float
    extraction_quality: float
    entity_extraction_quality: float
    synthesis_quality: float
    error_rate: float
    processing_time_seconds: float


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    timestamp: datetime
    overall_quality: float
    extraction_completeness: float
    entity_accuracy: float
    synthesis_appropriateness: float
    clinical_safety: float
    voice_consistency: float


class MetricsCollector:
    """
    Collects and manages metrics for the Enlitens pipeline
    
    Why monitoring is critical:
    - GPU temperature monitoring prevents hardware damage
    - Memory usage tracking prevents OOM errors
    - Quality metrics ensure clinical safety
    - Performance metrics optimize processing speed
    """
    
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize GPU monitoring
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
        except Exception as e:
            logger.warning(f"GPU monitoring not available: {e}")
        
        # Metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.processing_metrics: List[ProcessingMetrics] = []
        self.quality_metrics: List[QualityMetrics] = []

        # Performance tracking
        self.start_time = None
        self.documents_processed = 0
        self.total_processing_time = 0.0
        self.quality_scores = []
        self.errors = []

        # Observability bridge
        self.observability = get_observability()

        logger.info("MetricsCollector initialized")
    
    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = datetime.now()
        self.documents_processed = 0
        self.total_processing_time = 0.0
        self.quality_scores = []
        self.errors = []
        logger.info("Monitoring session started")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU metrics
            gpu_temperature = 0.0
            gpu_memory_used_gb = 0.0
            gpu_memory_total_gb = 0.0
            gpu_utilization = 0.0
            
            if self.gpu_available:
                try:
                    # Temperature
                    temp_info = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature = float(temp_info)
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_used_gb = mem_info.used / 1024**3
                    gpu_memory_total_gb = mem_info.total / 1024**3
                    
                    # Utilization
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_utilization = float(util_info.gpu)
                    
                except Exception as e:
                    logger.warning(f"Failed to get GPU metrics: {e}")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_temperature=gpu_temperature,
                gpu_memory_used_gb=gpu_memory_used_gb,
                gpu_memory_total_gb=gpu_memory_total_gb,
                gpu_utilization=gpu_utilization,
                disk_usage_percent=disk_usage_percent
            )
            
            self.system_metrics.append(metrics)
            try:
                self.observability.record_system_metrics(asdict(metrics))
            except Exception as exc:
                logger.debug(f"Failed to push system metrics to observability: {exc}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def record_processing_metrics(self, document_id: str, processing_time: float, 
                                quality_score: float, success: bool, error: Optional[str] = None):
        """Record processing metrics for a document"""
        self.documents_processed += 1
        self.total_processing_time += processing_time
        
        if success:
            self.quality_scores.append(quality_score)
        else:
            self.errors.append({
                'document_id': document_id,
                'error': error,
                'timestamp': datetime.now()
            })
        
        # Calculate derived metrics
        if self.start_time:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            documents_per_hour = self.documents_processed / elapsed_hours if elapsed_hours > 0 else 0
        else:
            documents_per_hour = 0
        
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
        error_rate = len(self.errors) / self.documents_processed if self.documents_processed > 0 else 0.0
        
        metrics = ProcessingMetrics(
            timestamp=datetime.now(),
            documents_processed=self.documents_processed,
            documents_per_hour=documents_per_hour,
            average_quality_score=avg_quality,
            extraction_quality=0.0,  # Would be calculated from actual results
            entity_extraction_quality=0.0,
            synthesis_quality=0.0,
            error_rate=error_rate,
            processing_time_seconds=processing_time
        )

        try:
            payload = {"stage": "processing", **asdict(metrics)}
            self.observability.record_quality_metrics(document_id, payload)
            if not success and error:
                self.observability.emit_alert(
                    "Document processing error",
                    "error",
                    error,
                    {"document_id": document_id},
                )
            if error_rate > 0.2 and self.documents_processed >= 3:
                self.observability.emit_alert(
                    "Error rate spike",
                    "warning",
                    f"Error rate reached {error_rate:.2%}",
                    {"documents_processed": self.documents_processed},
                )
        except Exception as exc:
            logger.debug(f"Failed to forward processing metrics: {exc}")

        self.processing_metrics.append(metrics)
        return metrics
    
    def record_quality_metrics(self, overall_quality: float, extraction_completeness: float,
                             entity_accuracy: float, synthesis_appropriateness: float,
                             clinical_safety: float, voice_consistency: float):
        """Record quality assessment metrics"""
        metrics = QualityMetrics(
            timestamp=datetime.now(),
            overall_quality=overall_quality,
            extraction_completeness=extraction_completeness,
            entity_accuracy=entity_accuracy,
            synthesis_appropriateness=synthesis_appropriateness,
            clinical_safety=clinical_safety,
            voice_consistency=voice_consistency
        )
        
        self.quality_metrics.append(metrics)
        return metrics
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system and processing status"""
        system_metrics = self.collect_system_metrics()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': system_metrics.cpu_percent if system_metrics else 0.0,
                'memory_percent': system_metrics.memory_percent if system_metrics else 0.0,
                'gpu_temperature': system_metrics.gpu_temperature if system_metrics else 0.0,
                'gpu_memory_used_gb': system_metrics.gpu_memory_used_gb if system_metrics else 0.0,
                'gpu_memory_total_gb': system_metrics.gpu_memory_total_gb if system_metrics else 0.0,
                'gpu_utilization': system_metrics.gpu_utilization if system_metrics else 0.0,
                'disk_usage_percent': system_metrics.disk_usage_percent if system_metrics else 0.0
            },
            'processing': {
                'documents_processed': self.documents_processed,
                'total_processing_time': self.total_processing_time,
                'average_quality_score': sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0,
                'error_count': len(self.errors),
                'error_rate': len(self.errors) / self.documents_processed if self.documents_processed > 0 else 0.0
            }
        }
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the current session"""
        if not self.start_time:
            return {'error': 'No monitoring session started'}
        
        elapsed_time = datetime.now() - self.start_time
        elapsed_hours = elapsed_time.total_seconds() / 3600
        
        summary = {
            'session_duration_hours': elapsed_hours,
            'documents_processed': self.documents_processed,
            'documents_per_hour': self.documents_processed / elapsed_hours if elapsed_hours > 0 else 0,
            'average_processing_time': self.total_processing_time / self.documents_processed if self.documents_processed > 0 else 0,
            'average_quality_score': sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0,
            'error_rate': len(self.errors) / self.documents_processed if self.documents_processed > 0 else 0.0,
            'total_errors': len(self.errors),
            'quality_distribution': self._get_quality_distribution()
        }
        
        return summary
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality scores"""
        if not self.quality_scores:
            return {}
        
        distribution = {
            'excellent (0.9-1.0)': 0,
            'good (0.8-0.9)': 0,
            'fair (0.7-0.8)': 0,
            'poor (0.6-0.7)': 0,
            'very_poor (<0.6)': 0
        }
        
        for score in self.quality_scores:
            if score >= 0.9:
                distribution['excellent (0.9-1.0)'] += 1
            elif score >= 0.8:
                distribution['good (0.8-0.9)'] += 1
            elif score >= 0.7:
                distribution['fair (0.7-0.8)'] += 1
            elif score >= 0.6:
                distribution['poor (0.6-0.7)'] += 1
            else:
                distribution['very_poor (<0.6)'] += 1
        
        return distribution
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze errors and provide insights"""
        if not self.errors:
            return {'error_count': 0, 'error_types': {}, 'recommendations': []}
        
        # Categorize errors
        error_types = {}
        for error in self.errors:
            error_msg = error['error'].lower()
            if 'extraction' in error_msg:
                error_types['extraction'] = error_types.get('extraction', 0) + 1
            elif 'entity' in error_msg:
                error_types['entity_extraction'] = error_types.get('entity_extraction', 0) + 1
            elif 'synthesis' in error_msg:
                error_types['synthesis'] = error_types.get('synthesis', 0) + 1
            elif 'ollama' in error_msg:
                error_types['ollama'] = error_types.get('ollama', 0) + 1
            elif 'memory' in error_msg:
                error_types['memory'] = error_types.get('memory', 0) + 1
            else:
                error_types['other'] = error_types.get('other', 0) + 1
        
        # Generate recommendations
        recommendations = []
        if error_types.get('memory', 0) > 0:
            recommendations.append("Consider reducing batch size or model memory usage")
        if error_types.get('ollama', 0) > 0:
            recommendations.append("Check Ollama service status and restart if needed")
        if error_types.get('extraction', 0) > 0:
            recommendations.append("Review PDF quality and extraction parameters")
        
        return {
            'error_count': len(self.errors),
            'error_types': error_types,
            'recommendations': recommendations,
            'recent_errors': self.errors[-5:]  # Last 5 errors
        }
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        data = {
            'session_info': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            },
            'system_metrics': [asdict(m) for m in self.system_metrics],
            'processing_metrics': [asdict(m) for m in self.processing_metrics],
            'quality_metrics': [asdict(m) for m in self.quality_metrics],
            'performance_summary': self.get_performance_summary(),
            'error_analysis': self.get_error_analysis()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Metrics saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return None
    
    def load_metrics(self, filename: str) -> Dict[str, Any]:
        """Load metrics from file"""
        filepath = self.metrics_dir / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Metrics loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for optimizing performance"""
        recommendations = []
        
        # Check system metrics
        if self.system_metrics:
            latest_system = self.system_metrics[-1]
            
            if latest_system.gpu_temperature > 80:
                recommendations.append("GPU temperature is high - consider improving cooling")
            
            if latest_system.gpu_memory_used_gb / latest_system.gpu_memory_total_gb > 0.9:
                recommendations.append("GPU memory usage is high - consider reducing model sizes")
            
            if latest_system.cpu_percent > 90:
                recommendations.append("CPU usage is high - consider reducing parallel processing")
        
        # Check processing metrics
        if self.processing_metrics:
            latest_processing = self.processing_metrics[-1]
            
            if latest_processing.documents_per_hour < 1:
                recommendations.append("Processing speed is slow - consider optimizing extraction parameters")
            
            if latest_processing.error_rate > 0.1:
                recommendations.append("Error rate is high - review error logs and fix issues")
        
        # Check quality metrics
        if self.quality_metrics:
            latest_quality = self.quality_metrics[-1]
            
            if latest_quality.overall_quality < 0.8:
                recommendations.append("Quality scores are low - review extraction and synthesis parameters")
            
            if latest_quality.clinical_safety < 0.9:
                recommendations.append("Clinical safety scores are concerning - review synthesis prompts")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Test the metrics collector
    collector = MetricsCollector()
    
    # Start monitoring
    collector.start_monitoring()
    
    # Collect some sample metrics
    for i in range(5):
        system_metrics = collector.collect_system_metrics()
        print(f"System metrics: CPU {system_metrics.cpu_percent}%, GPU {system_metrics.gpu_temperature}°C")
        
        # Simulate processing
        collector.record_processing_metrics(
            f"doc_{i}", 
            processing_time=30.0, 
            quality_score=0.85 + (i * 0.02), 
            success=True
        )
        
        time.sleep(1)
    
    # Get summary
    summary = collector.get_performance_summary()
    print("Performance summary:", summary)
    
    # Get recommendations
    recommendations = collector.get_optimization_recommendations()
    print("Recommendations:", recommendations)
    
    # Save metrics
    collector.save_metrics()
