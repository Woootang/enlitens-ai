"""
GPU Memory Manager - Monitor and optimize GPU memory usage
"""

import logging
import torch
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manage GPU memory allocation and provide monitoring."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        else:
            self.device_count = 0
            self.device_name = "CPU"
            self.total_memory = 0
    
    def get_memory_stats(self, device: int = 0) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return {
                "total_gb": 0,
                "allocated_gb": 0,
                "reserved_gb": 0,
                "free_gb": 0,
                "utilization_pct": 0
            }
        
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = self.total_memory - allocated
        utilization = (allocated / self.total_memory) * 100
        
        return {
            "total_gb": round(self.total_memory, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "utilization_pct": round(utilization, 1)
        }
    
    def log_memory_stats(self, prefix: str = "GPU Memory", device: int = 0):
        """Log current GPU memory statistics."""
        stats = self.get_memory_stats(device)
        
        if self.cuda_available:
            logger.info(
                f"{prefix}: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB "
                f"({stats['utilization_pct']:.1f}% used, {stats['free_gb']:.2f}GB free)"
            )
        else:
            logger.info(f"{prefix}: Running on CPU")
    
    def clear_cache(self, device: Optional[int] = None):
        """Clear GPU cache to free up memory."""
        if not self.cuda_available:
            return
        
        try:
            torch.cuda.empty_cache()
            if device is not None:
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize()
            
            logger.info("✅ GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def check_available_memory(self, required_gb: float, device: int = 0) -> bool:
        """Check if enough GPU memory is available."""
        if not self.cuda_available:
            return False
        
        stats = self.get_memory_stats(device)
        available = stats['free_gb']
        
        if available >= required_gb:
            logger.info(f"✅ Sufficient GPU memory: {available:.2f}GB available, {required_gb:.2f}GB required")
            return True
        else:
            logger.warning(f"⚠️  Insufficient GPU memory: {available:.2f}GB available, {required_gb:.2f}GB required")
            return False
    
    def get_optimal_batch_size(self, base_batch_size: int = 8, model_size_gb: float = 1.0) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        if not self.cuda_available:
            return 1
        
        stats = self.get_memory_stats()
        available_gb = stats['free_gb']
        
        # Reserve 2GB for safety margin
        usable_gb = max(0, available_gb - 2.0)
        
        # Calculate how many batches fit
        max_batches = int(usable_gb / model_size_gb)
        optimal_batch_size = min(base_batch_size, max(1, max_batches))
        
        logger.info(f"📊 Optimal batch size: {optimal_batch_size} (based on {usable_gb:.2f}GB available)")
        return optimal_batch_size
    
    def force_cleanup(self, device: int = 0):
        """Force aggressive GPU memory cleanup."""
        if not self.cuda_available:
            return
        
        try:
            import gc
            
            # Force Python garbage collection
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device)
            
            logger.info("✅ Aggressive GPU cleanup completed")
            self.log_memory_stats("After cleanup")
            
        except Exception as e:
            logger.warning(f"Failed to force cleanup: {e}")


# Global instance
_gpu_manager = None

def get_gpu_manager() -> GPUMemoryManager:
    """Get global GPU memory manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUMemoryManager()
    return _gpu_manager

