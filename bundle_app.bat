@echo off
echo FileAlchemy Bundling Process
echo ===========================
echo.

echo Step 1: Installing required packages...
pip install pyinstaller pillow pillow-heif fastapi uvicorn python-multipart
pip install moviepy reportlab weasyprint markdown pandas python-docx img2pdf
pip install numpy easyocr pdfkit pdfplumber imageio-ffmpeg
echo.

echo Step 2: Checking for FFmpeg...
if not exist ffmpeg\ffmpeg.exe (
    echo FFmpeg not found in ffmpeg directory.
    echo Please download FFmpeg and place ffmpeg.exe in a directory named 'ffmpeg'.
    echo You can download it from: https://ffmpeg.org/download.html
    pause
    exit /b 1
)
echo FFmpeg found. Proceeding with bundling.
echo.

echo Step 3: Preparing application for bundling...
python prepare_for_bundle.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to prepare application for bundling.
    exit /b 1
)
echo.

echo Step 4: Creating executable bundle...
python bundle.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create executable bundle.
    exit /b 1
)
echo.

echo Bundle process complete!
echo You can find the bundled application in: dist\FileAlchemy\
echo.
echo To run the application, execute: dist\FileAlchemy\FileAlchemy.bat
echo.
pause 