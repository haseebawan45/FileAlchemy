import os
import gc
import psutil
import asyncio
import tempfile
import shutil
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources for Railway's free tier constraints"""
    
    def __init__(self, max_memory_mb: int = 400, max_concurrent: int = 1):
        self.max_memory_mb = max_memory_mb
        self.max_concurrent = max_concurrent
        self.active_conversions = 0
        self.conversion_queue = []
        self.temp_files = set()
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        
        # Get process for memory monitoring
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of limit"""
        current_memory = self.get_memory_usage()
        return (current_memory / self.max_memory_mb) * 100
    
    def can_process_request(self, estimated_memory_mb: int = 50) -> bool:
        """Check if we can process a new request"""
        current_memory = self.get_memory_usage()
        
        # Check memory limit
        if current_memory + estimated_memory_mb > self.max_memory_mb:
            logger.warning(f"Memory limit would be exceeded: {current_memory + estimated_memory_mb}MB > {self.max_memory_mb}MB")
            return False
        
        # Check concurrent limit
        if self.active_conversions >= self.max_concurrent:
            logger.info(f"Concurrent limit reached: {self.active_conversions}/{self.max_concurrent}")
            return False
        
        return True
    
    def start_conversion(self, task_id: str, estimated_memory_mb: int = 50) -> bool:
        """Start a new conversion if resources allow"""
        if not self.can_process_request(estimated_memory_mb):
            return False
        
        self.active_conversions += 1
        logger.info(f"Started conversion {task_id}. Active: {self.active_conversions}")
        return True
    
    def end_conversion(self, task_id: str):
        """End a conversion and free up resources"""
        if self.active_conversions > 0:
            self.active_conversions -= 1
        
        # Force cleanup after each conversion
        self.cleanup_resources()
        logger.info(f"Ended conversion {task_id}. Active: {self.active_conversions}")
    
    def cleanup_resources(self):
        """Aggressive cleanup of resources"""
        try:
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # Update last cleanup time
            self.last_cleanup = datetime.now()
            
            current_memory = self.get_memory_usage()
            logger.info(f"Cleanup completed. Memory usage: {current_memory:.1f}MB ({self.get_memory_percent():.1f}%)")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up tracked temporary files"""
        cleaned_count = 0
        files_to_remove = set()
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    if os.path.isfile(temp_file):
                        os.remove(temp_file)
                    elif os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    cleaned_count += 1
                files_to_remove.add(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean temp file {temp_file}: {e}")
        
        # Remove cleaned files from tracking
        self.temp_files -= files_to_remove
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} temporary files")
    
    def register_temp_file(self, file_path: str):
        """Register a temporary file for cleanup"""
        self.temp_files.add(file_path)
    
    def periodic_cleanup(self):
        """Perform periodic cleanup if needed"""
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self.cleanup_resources()
    
    def get_status(self) -> Dict:
        """Get current resource status"""
        memory_usage = self.get_memory_usage()
        return {
            "memory_usage_mb": round(memory_usage, 2),
            "memory_limit_mb": self.max_memory_mb,
            "memory_percent": round(self.get_memory_percent(), 2),
            "active_conversions": self.active_conversions,
            "max_concurrent": self.max_concurrent,
            "temp_files_count": len(self.temp_files),
            "can_accept_requests": self.can_process_request()
        }
    
    def force_cleanup_if_needed(self):
        """Force cleanup if memory usage is high"""
        memory_percent = self.get_memory_percent()
        
        if memory_percent > 80:  # If using more than 80% of limit
            logger.warning(f"High memory usage detected: {memory_percent:.1f}%. Forcing cleanup.")
            self.cleanup_resources()
            
            # If still high after cleanup, more aggressive measures
            if self.get_memory_percent() > 90:
                logger.error("Critical memory usage after cleanup. Implementing emergency measures.")
                # Could implement more aggressive cleanup here
                gc.collect()
                gc.collect()  # Run twice for better effect

# Global resource manager instance
resource_manager = ResourceManager()

def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    return resource_manager