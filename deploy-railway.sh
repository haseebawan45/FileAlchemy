#!/bin/bash

# Railway Deployment Script for FileAlchemy
# This script helps deploy FileAlchemy to Railway platform

echo "🚀 FileAlchemy Railway Deployment Script"
echo "========================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed."
    echo "Please install it first:"
    echo "npm install -g @railway/cli"
    echo "or visit: https://docs.railway.app/develop/cli"
    exit 1
fi

echo "✅ Railway CLI found"

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway first:"
    railway login
fi

echo "✅ Railway authentication verified"

# Check if we're in a Railway project
if ! railway status &> /dev/null; then
    echo "📦 Initializing new Railway project..."
    railway login
    railway init
else
    echo "✅ Railway project detected"
fi

# Set environment variables for Railway
echo "🔧 Setting up environment variables..."

# Set Railway-specific environment variables
railway variables set RAILWAY_ENVIRONMENT=production
railway variables set PORT=8080
railway variables set PYTHONUNBUFFERED=1
railway variables set PYTHONDONTWRITEBYTECODE=1

echo "✅ Environment variables configured"

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment initiated!"
echo ""
echo "📋 Next steps:"
echo "1. Monitor deployment: railway logs"
echo "2. Check status: railway status"
echo "3. Open app: railway open"
echo ""
echo "🔗 Useful Railway commands:"
echo "- railway logs --follow    # Follow logs in real-time"
echo "- railway shell           # Access deployment shell"
echo "- railway variables       # Manage environment variables"
echo "- railway status          # Check deployment status"
echo ""
echo "🎉 FileAlchemy deployment to Railway completed!"