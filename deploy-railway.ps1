# Railway Deployment Script for FileAlchemy (PowerShell)
# This script helps deploy FileAlchemy to Railway platform

Write-Host "🚀 FileAlchemy Railway Deployment Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if Railway CLI is installed
try {
    railway --version | Out-Null
    Write-Host "✅ Railway CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Railway CLI is not installed." -ForegroundColor Red
    Write-Host "Please install it first:" -ForegroundColor Yellow
    Write-Host "npm install -g @railway/cli" -ForegroundColor Cyan
    Write-Host "or visit: https://docs.railway.app/develop/cli" -ForegroundColor Cyan
    exit 1
}

# Check if user is logged in
$loginStatus = railway whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "🔐 Please log in to Railway first:" -ForegroundColor Yellow
    railway login
    # Wait for login to complete
    Start-Sleep -Seconds 2
    $loginStatus = railway whoami 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Login failed. Please try again." -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Railway authentication verified" -ForegroundColor Green

# Check if we're in a Railway project
$projectStatus = railway status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Initializing new Railway project..." -ForegroundColor Yellow
    railway init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to initialize Railway project." -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Railway project detected" -ForegroundColor Green

# Set environment variables for Railway
Write-Host "🔧 Setting up environment variables..." -ForegroundColor Blue

# Set Railway-specific environment variables
railway variables --set RAILWAY_ENVIRONMENT=production
railway variables --set PORT=8080
railway variables --set PYTHONUNBUFFERED=1
railway variables --set PYTHONDONTWRITEBYTECODE=1

Write-Host "✅ Environment variables configured" -ForegroundColor Green

# Deploy to Railway
Write-Host "🚀 Deploying to Railway..." -ForegroundColor Blue
railway deploy

Write-Host "✅ Deployment initiated!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Monitor deployment: railway logs" -ForegroundColor White
Write-Host "2. Check status: railway status" -ForegroundColor White
Write-Host "3. Open app: railway open" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Useful Railway commands:" -ForegroundColor Yellow
Write-Host "- railway logs --follow    # Follow logs in real-time" -ForegroundColor White
Write-Host "- railway shell           # Access deployment shell" -ForegroundColor White
Write-Host "- railway variables       # Manage environment variables" -ForegroundColor White
Write-Host "- railway status          # Check deployment status" -ForegroundColor White
Write-Host ""
Write-Host "🎉 FileAlchemy deployment to Railway completed!" -ForegroundColor Green