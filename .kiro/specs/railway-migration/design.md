# Railway Migration Design Document

## Overview

This design document outlines the architecture and implementation strategy for migrating FileAlchemy from Fly.io to Railway's free tier. The migration focuses on optimizing resource usage, implementing Railway-specific configurations, and maintaining application functionality within the constraints of Railway's free tier (512MB RAM, 0.5 vCPU).

## Architecture

### Current Architecture Analysis
- **Backend**: FastAPI application with heavy dependencies (PyMuPDF, moviepy, easyocr, weasyprint)
- **Frontend**: Static HTML/CSS/JS files
- **Current Resources**: 1 CPU, 1GB RAM on Fly.io
- **Target Resources**: 0.5 vCPU, 512MB RAM on Railway free tier

### Proposed Architecture Changes

#### 1. Resource Optimization Layer
- **Memory Management**: Implement streaming file processing and immediate cleanup
- **Dependency Optimization**: Replace heavy libraries with lighter alternatives where possible
- **Request Queuing**: Implement a simple queue system to prevent concurrent heavy operations
- **File Size Limits**: Reduce maximum file size from 100MB to 25MB for free tier

#### 2. Railway-Specific Configuration
- **Railway Configuration**: Replace fly.toml with railway.json and proper environment setup
- **Port Binding**: Update to use Railway's PORT environment variable
- **Build Optimization**: Optimize Docker build for Railway's build system
- **Static File Serving**: Configure proper static file serving for Railway

#### 3. Performance Monitoring
- **Resource Monitoring**: Add memory and CPU usage tracking
- **Error Handling**: Enhanced error handling for resource-constrained environment
- **Logging**: Implement structured logging for Railway's log system

## Components and Interfaces

### 1. Resource Manager Component
```python
class ResourceManager:
    def __init__(self, max_memory_mb=400):  # Leave buffer for system
        self.max_memory_mb = max_memory_mb
        self.active_conversions = 0
        self.max_concurrent = 1  # Limit to 1 conversion at a time
    
    def can_process_request(self) -> bool:
        # Check memory usage and active conversions
        pass
    
    def cleanup_resources(self):
        # Force garbage collection and temp file cleanup
        pass
```

### 2. Lightweight Conversion Engine
```python
class OptimizedConverter:
    def __init__(self):
        self.conversion_methods = {
            'pdf_to_text': self.lightweight_pdf_text,
            'image_convert': self.streaming_image_convert,
            # Prioritize lightweight methods
        }
    
    def select_conversion_method(self, file_type, target_format):
        # Choose most memory-efficient method
        pass
```

### 3. Railway Configuration Manager
```python
class RailwayConfig:
    def __init__(self):
        self.port = os.getenv('PORT', 8080)
        self.environment = os.getenv('RAILWAY_ENVIRONMENT', 'production')
        self.static_url = os.getenv('RAILWAY_STATIC_URL', '')
    
    def get_cors_origins(self):
        # Configure CORS for Railway deployment
        pass
```

## Data Models

### 1. Conversion Request Model
```python
class ConversionRequest:
    task_id: str
    file_path: str
    source_format: str
    target_format: str
    file_size: int  # Track for memory planning
    priority: int   # Queue priority
    created_at: datetime
    memory_estimate: int  # Estimated memory usage
```

### 2. Resource Usage Model
```python
class ResourceUsage:
    timestamp: datetime
    memory_usage_mb: float
    cpu_usage_percent: float
    active_conversions: int
    queue_length: int
```

### 3. Railway Deployment Model
```python
class RailwayDeployment:
    app_name: str
    environment: str
    port: int
    static_files_path: str
    max_file_size_mb: int = 25  # Reduced for free tier
    max_concurrent_conversions: int = 1
```

## Error Handling

### 1. Resource Exhaustion Handling
- **Memory Limit Exceeded**: Return HTTP 503 with retry-after header
- **Queue Full**: Return HTTP 429 with queue status
- **File Too Large**: Return HTTP 413 with size limit information
- **Conversion Timeout**: Return HTTP 408 with partial results if available

### 2. Railway-Specific Error Handling
- **Port Binding Errors**: Fallback port configuration
- **Build Failures**: Detailed logging for Railway build process
- **Environment Variable Issues**: Default value fallbacks

### 3. Graceful Degradation
- **Heavy Operations**: Fallback to simpler conversion methods
- **Memory Pressure**: Automatic cleanup and request queuing
- **Service Unavailable**: Clear user messaging about free tier limitations

## Testing Strategy

### 1. Resource Constraint Testing
- **Memory Usage Tests**: Verify application stays under 400MB during normal operation
- **Concurrent Request Tests**: Test behavior with multiple simultaneous requests
- **Large File Tests**: Verify proper handling of files approaching size limits
- **Stress Tests**: Test application behavior under resource pressure

### 2. Railway Integration Testing
- **Deployment Tests**: Verify successful deployment to Railway
- **Environment Variable Tests**: Test all Railway-specific configurations
- **Static File Serving Tests**: Verify frontend assets load correctly
- **Port Binding Tests**: Test application starts on Railway's assigned port

### 3. Functionality Preservation Tests
- **Core Conversion Tests**: Verify all essential conversions still work
- **Error Handling Tests**: Test graceful handling of resource limitations
- **User Experience Tests**: Verify frontend still provides good user experience
- **Performance Tests**: Measure response times within Railway environment

## Implementation Phases

### Phase 1: Resource Optimization
1. Implement ResourceManager component
2. Add memory monitoring and cleanup
3. Reduce file size limits and optimize heavy operations
4. Test memory usage under various scenarios

### Phase 2: Railway Configuration
1. Create railway.json configuration
2. Update Dockerfile for Railway optimization
3. Configure environment variables and port binding
4. Set up static file serving for Railway

### Phase 3: Lightweight Conversion Engine
1. Implement OptimizedConverter with memory-efficient methods
2. Add request queuing system
3. Replace heavy dependencies where possible
4. Test conversion functionality with resource constraints

### Phase 4: Monitoring and Error Handling
1. Add resource usage monitoring
2. Implement comprehensive error handling
3. Add logging for Railway environment
4. Test error scenarios and recovery

### Phase 5: Deployment and Validation
1. Deploy to Railway and verify functionality
2. Monitor resource usage in production
3. Optimize based on real-world performance
4. Document deployment process and maintenance procedures

## Railway-Specific Considerations

### 1. Free Tier Limitations
- **Memory**: 512MB total (target 400MB for application)
- **CPU**: 0.5 vCPU shared
- **Storage**: Ephemeral (temporary files must be cleaned up)
- **Network**: Limited bandwidth
- **Sleep**: Application may sleep after inactivity

### 2. Railway Features to Leverage
- **Automatic HTTPS**: Built-in SSL certificates
- **Environment Variables**: Secure configuration management
- **Logs**: Integrated logging system
- **Metrics**: Basic performance monitoring
- **GitHub Integration**: Automatic deployments from repository

### 3. Optimization Strategies
- **Lazy Loading**: Load heavy dependencies only when needed
- **Streaming Processing**: Process files in chunks to minimize memory usage
- **Aggressive Cleanup**: Immediate cleanup of temporary files and memory
- **Request Throttling**: Limit concurrent operations to prevent resource exhaustion
- **Caching**: Cache frequently used conversion results (within memory limits)