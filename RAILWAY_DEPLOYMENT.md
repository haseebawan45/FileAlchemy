# Railway Deployment Guide for FileAlchemy

This guide will help you deploy FileAlchemy to Railway's free tier.

## Prerequisites

1. **Railway CLI**: Install the Railway CLI
   ```bash
   npm install -g @railway/cli
   ```

2. **Railway Account**: Sign up at [railway.app](https://railway.app)

3. **Git Repository**: Your code should be in a Git repository

## Quick Deployment

### Option 1: Using Deployment Script (Recommended)

**For Windows (PowerShell):**
```powershell
.\deploy-railway.ps1
```

**For Linux/macOS:**
```bash
chmod +x deploy-railway.sh
./deploy-railway.sh
```

### Option 2: Manual Deployment

1. **Login to Railway:**
   ```bash
   railway login
   ```

2. **Initialize Project:**
   ```bash
   railway init
   ```

3. **Set Environment Variables:**
   ```bash
   railway variables set RAILWAY_ENVIRONMENT=production
   railway variables set PORT=8080
   railway variables set PYTHONUNBUFFERED=1
   railway variables set PYTHONDONTWRITEBYTECODE=1
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

## Configuration

### Environment Variables

The following environment variables are automatically set by the deployment script:

- `RAILWAY_ENVIRONMENT=production`
- `PORT=8080`
- `PYTHONUNBUFFERED=1`
- `PYTHONDONTWRITEBYTECODE=1`

### Resource Limits (Free Tier)

Railway's free tier provides:
- **Memory**: 512MB (we target 400MB for the app)
- **CPU**: 0.5 vCPU shared
- **Storage**: Ephemeral (temporary files are cleaned up)
- **File Size Limit**: 25MB (reduced from 100MB)

## Monitoring

### Check Deployment Status
```bash
railway status
```

### View Logs
```bash
railway logs
railway logs --follow  # Follow logs in real-time
```

### Open Application
```bash
railway open
```

## Troubleshooting

### Common Issues

1. **Memory Limit Exceeded**
   - The app automatically limits file size to 25MB
   - Only 1 conversion runs at a time
   - Aggressive cleanup is implemented

2. **Build Failures**
   - Check logs: `railway logs`
   - Ensure all dependencies are in `requirements.txt`
   - Verify Dockerfile is optimized

3. **Application Not Starting**
   - Check if PORT environment variable is set
   - Verify health check endpoint: `/health`
   - Check resource usage: `/resource-status`

### Useful Commands

```bash
# Access deployment shell
railway shell

# Manage environment variables
railway variables
railway variables set KEY=value
railway variables delete KEY

# Restart deployment
railway up --detach

# Check resource usage
curl https://your-app.railway.app/resource-status

# Check queue status
curl https://your-app.railway.app/queue-status
```

## Performance Optimization

### File Size Limits
- Maximum file size: 25MB (enforced in frontend and backend)
- Files larger than 25MB will be rejected with a clear error message

### Memory Management
- Automatic cleanup after each conversion
- Streaming processing for large files
- Request queuing to prevent resource exhaustion

### Conversion Optimization
- Lightweight conversion methods prioritized
- Memory-efficient algorithms
- Aggressive garbage collection

## Frontend Configuration

Update your frontend to point to the Railway backend URL:

1. **Update API endpoints** in your frontend JavaScript
2. **Configure CORS** origins in the backend
3. **Test file size validation** (25MB limit)

## Monitoring Endpoints

Your deployed app provides these monitoring endpoints:

- `/health` - Health check
- `/resource-status` - Current resource usage
- `/queue-status` - Conversion queue status
- `/queue-position/{task_id}` - Position of specific request in queue

## Support

If you encounter issues:

1. Check the logs: `railway logs`
2. Verify resource usage: visit `/resource-status`
3. Check queue status: visit `/queue-status`
4. Review this guide for common solutions

## Railway-Specific Features

### Automatic HTTPS
Railway provides automatic HTTPS certificates for your domain.

### Environment Management
Use Railway's dashboard or CLI to manage environment variables securely.

### Automatic Deployments
Connect your GitHub repository for automatic deployments on push.

### Custom Domains
You can add custom domains through Railway's dashboard (paid feature).

---

**Note**: This deployment is optimized for Railway's free tier. For production use with higher traffic, consider upgrading to Railway's paid plans for better resource limits.