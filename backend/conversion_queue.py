import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversionRequest:
    task_id: str
    file_path: str
    convert_func: Callable
    priority: int = 1
    created_at: datetime = None
    estimated_memory_mb: int = 50
    timeout_seconds: int = 300  # 5 minutes default timeout
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class ConversionQueue:
    """Queue system for managing conversion requests on Railway's free tier"""
    
    def __init__(self, max_queue_size: int = 10):
        self.queue: List[ConversionRequest] = []
        self.processing: Dict[str, ConversionRequest] = {}
        self.max_queue_size = max_queue_size
        self.is_processing = False
        
    def add_request(self, request: ConversionRequest) -> bool:
        """Add a conversion request to the queue"""
        if len(self.queue) >= self.max_queue_size:
            logger.warning(f"Queue is full ({self.max_queue_size}). Rejecting request {request.task_id}")
            return False
        
        # Insert based on priority (higher priority first)
        inserted = False
        for i, queued_request in enumerate(self.queue):
            if request.priority > queued_request.priority:
                self.queue.insert(i, request)
                inserted = True
                break
        
        if not inserted:
            self.queue.append(request)
        
        logger.info(f"Added request {request.task_id} to queue. Queue size: {len(self.queue)}")
        return True
    
    def get_next_request(self) -> Optional[ConversionRequest]:
        """Get the next request from the queue"""
        if not self.queue:
            return None
        
        # Remove expired requests first
        self._cleanup_expired_requests()
        
        if self.queue:
            request = self.queue.pop(0)
            self.processing[request.task_id] = request
            logger.info(f"Processing request {request.task_id}. Queue size: {len(self.queue)}")
            return request
        
        return None
    
    def complete_request(self, task_id: str):
        """Mark a request as completed"""
        if task_id in self.processing:
            request = self.processing.pop(task_id)
            logger.info(f"Completed request {task_id}")
        
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            "queue_length": len(self.queue),
            "processing_count": len(self.processing),
            "max_queue_size": self.max_queue_size,
            "queue_full": len(self.queue) >= self.max_queue_size,
            "estimated_wait_time": self._estimate_wait_time()
        }
    
    def get_request_position(self, task_id: str) -> Optional[int]:
        """Get the position of a request in the queue"""
        for i, request in enumerate(self.queue):
            if request.task_id == task_id:
                return i + 1  # 1-based position
        return None
    
    def _cleanup_expired_requests(self):
        """Remove expired requests from the queue"""
        current_time = datetime.now()
        expired_requests = []
        
        for i, request in enumerate(self.queue):
            if current_time - request.created_at > timedelta(seconds=request.timeout_seconds):
                expired_requests.append(i)
        
        # Remove expired requests (in reverse order to maintain indices)
        for i in reversed(expired_requests):
            expired_request = self.queue.pop(i)
            logger.warning(f"Removed expired request {expired_request.task_id} from queue")
    
    def _estimate_wait_time(self) -> int:
        """Estimate wait time in seconds for new requests"""
        if not self.queue:
            return 0
        
        # Rough estimate: 30 seconds per request in queue
        return len(self.queue) * 30

class QueueProcessor:
    """Processes queued conversion requests"""
    
    def __init__(self, queue: ConversionQueue, resource_manager):
        self.queue = queue
        self.resource_manager = resource_manager
        self.is_running = False
        
    async def start_processing(self):
        """Start processing queued requests"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Started queue processor")
        
        while self.is_running:
            try:
                # Check if we can process a request
                if self.resource_manager.active_conversions < self.resource_manager.max_concurrent:
                    request = self.queue.get_next_request()
                    
                    if request:
                        # Process the request
                        asyncio.create_task(self._process_request(request))
                    else:
                        # No requests in queue, wait a bit
                        await asyncio.sleep(1)
                else:
                    # Resource limit reached, wait longer
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_request(self, request: ConversionRequest):
        """Process a single conversion request"""
        try:
            from server import conversion_progress
            
            # Update progress
            conversion_progress[request.task_id] = {
                "progress": 5,
                "status": "Starting conversion from queue"
            }
            
            # Start resource tracking
            if not self.resource_manager.start_conversion(request.task_id, request.estimated_memory_mb):
                conversion_progress[request.task_id] = {
                    "progress": 100,
                    "status": "Error: Unable to allocate resources"
                }
                self.queue.complete_request(request.task_id)
                return
            
            # Register temp file for cleanup
            self.resource_manager.register_temp_file(request.file_path)
            
            # Execute the conversion function
            import inspect
            
            if 'task_id' in inspect.signature(request.convert_func).parameters:
                if inspect.iscoroutinefunction(request.convert_func):
                    converted_file_path = await request.convert_func(request.file_path, task_id=request.task_id)
                else:
                    converted_file_path = request.convert_func(request.file_path, task_id=request.task_id)
            else:
                # Legacy function without progress tracking
                conversion_progress[request.task_id] = {"progress": 20, "status": "Processing conversion"}
                
                if inspect.iscoroutinefunction(request.convert_func):
                    converted_file_path = await request.convert_func(request.file_path)
                else:
                    converted_file_path = request.convert_func(request.file_path)
                
                conversion_progress[request.task_id] = {"progress": 90, "status": "Conversion processing complete"}
            
            # Update final progress
            if converted_file_path and request.task_id in conversion_progress:
                conversion_progress[request.task_id].update({
                    "progress": 100,
                    "status": "Conversion complete",
                    "file_path": converted_file_path,
                    "file_name": os.path.basename(converted_file_path)
                })
            
            # Clean up the uploaded file
            import os
            if os.path.exists(request.file_path):
                os.remove(request.file_path)
                
        except Exception as e:
            logger.error(f"Error processing request {request.task_id}: {e}")
            conversion_progress[request.task_id] = {
                "progress": 100,
                "status": f"Error: {str(e)}"
            }
        finally:
            # Always complete the request and cleanup
            self.resource_manager.end_conversion(request.task_id)
            self.queue.complete_request(request.task_id)
    
    def stop_processing(self):
        """Stop processing queued requests"""
        self.is_running = False
        logger.info("Stopped queue processor")

# Global queue instances
conversion_queue = ConversionQueue()
queue_processor = None

def get_conversion_queue() -> ConversionQueue:
    """Get the global conversion queue instance"""
    return conversion_queue

def get_queue_processor(resource_manager) -> QueueProcessor:
    """Get the global queue processor instance"""
    global queue_processor
    if queue_processor is None:
        queue_processor = QueueProcessor(conversion_queue, resource_manager)
    return queue_processor