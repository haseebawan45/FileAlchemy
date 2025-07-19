# Railway Deployment Fix Guide

## 🔧 Issues Identified

1. **Authentication Problem**: Railway CLI authentication is failing
2. **Command Syntax**: Old Railway CLI command format being used
3. **Environment Variables**: Incorrect syntax for setting variables

## ✅ Solutions Applied

### 1. Fixed PowerShell Script
I've updated your `deploy-railway.ps1` script with:
- Correct Railway CLI command syntax
- Better error handling for authentication
- Proper environment variable setting format

### 2. Step-by-Step Fix Process

#### Step 1: Re-authenticate with Railway
```powershell
# First, logout to clear any cached auth
railway logout

# Then login again
railway login
```

#### Step 2: Initialize Project (if needed)
```powershell
# If you don't have a Railway project yet
railway init

# Select "Deploy from existing code"
# Choose your project name (e.g., "filealchemy")
```

#### Step 3: Set Environment Variables Manually
```powershell
# Use the correct syntax for setting variables
railway variables --set RAILWAY_ENVIRONMENT=production
railway variables --set PORT=8080
railway variables --set PYTHONUNBUFFERED=1
railway variables --set PYTHONDONTWRITEBYTECODE=1
```

#### Step 4: Deploy
```powershell
# Deploy using the correct command
railway deploy
```

## 🚀 Quick Fix Commands

Run these commands in order:

```powershell
# 1. Logout and login again
railway logout
railway login

# 2. Check if you're in a project
railway status

# 3. If no project, initialize one
railway init

# 4. Set environment variables
railway variables --set RAILWAY_ENVIRONMENT=production
railway variables --set PORT=8080
railway variables --set PYTHONUNBUFFERED=1
railway variables --set PYTHONDONTWRITEBYTECODE=1

# 5. Deploy
railway deploy
```

## 📋 Alternative: Manual Railway Setup

If the CLI continues to have issues, you can also:

1. **Go to Railway Dashboard**: https://railway.app/dashboard
2. **Create New Project**: Click "New Project"
3. **Deploy from GitHub**: Connect your repository
4. **Set Environment Variables**: In project settings
5. **Configure Build**: Point to `backend/Dockerfile`

## 🔍 Troubleshooting

### If authentication still fails:
```powershell
# Check Railway CLI version
railway --version

# Update Railway CLI if needed
npm install -g @railway/cli@latest

# Clear any cached credentials
railway logout
railway login
```

### If deployment fails:
```powershell
# Check logs
railway logs

# Check project status
railway status

# View environment variables
railway variables
```

## ✅ Next Steps After Successful Deployment

1. **Monitor deployment**: `railway logs --follow`
2. **Check app status**: `railway status`
3. **Open your app**: `railway open`
4. **Update frontend config**: Update API URLs in frontend to point to Railway deployment

## 🔗 Useful Railway Commands

- `railway logs --follow` - Follow logs in real-time
- `railway shell` - Access deployment shell
- `railway variables` - Manage environment variables
- `railway status` - Check deployment status
- `railway open` - Open your deployed app
- `railway restart` - Restart your service