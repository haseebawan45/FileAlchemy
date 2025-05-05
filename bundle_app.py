#!/usr/bin/env python3
"""
Cross-platform script for bundling FileAlchemy application
"""
import os
import sys
import platform
import subprocess
import shutil

def main():
    """Main function to run the bundling process"""
    print("===========================")
    print("FileAlchemy Bundling Process")
    print("===========================")
    print()
    
    # Check Python version
    py_version = platform.python_version()
    print(f"Python version: {py_version}")
    
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required.")
        sys.exit(1)
    
    # Step 1: Installing required packages
    print("Step 1: Installing required packages...")
    packages = [
        "pyinstaller",
        "pillow", 
        "pillow-heif", 
        "fastapi", 
        "uvicorn", 
        "python-multipart",
        "moviepy", 
        "reportlab", 
        "weasyprint", 
        "markdown", 
        "pandas", 
        "python-docx", 
        "img2pdf",
        "numpy", 
        "easyocr", 
        "pdfkit", 
        "pdfplumber", 
        "imageio-ffmpeg"
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print("All required packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {str(e)}")
        print("Please install the required packages manually and try again.")
        sys.exit(1)
    
    # Step 2: Check for FFmpeg
    print("\nStep 2: Checking for FFmpeg...")
    system = platform.system()
    
    if system == "Windows":
        ffmpeg_path = os.path.join("ffmpeg", "ffmpeg.exe")
    else:  # Linux or macOS
        ffmpeg_path = os.path.join("ffmpeg", "ffmpeg")
    
    if not os.path.exists(ffmpeg_path):
        print(f"FFmpeg not found at: {ffmpeg_path}")
        print("Please download FFmpeg and place it in a directory named 'ffmpeg'.")
        print("You can download it from: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    print(f"FFmpeg found at: {ffmpeg_path}")
    print("Proceeding with bundling.")
    
    # Step 3: Prepare for bundling
    print("\nStep 3: Preparing application for bundling...")
    result = subprocess.run([sys.executable, "prepare_for_bundle.py"], check=False)
    
    if result.returncode != 0:
        print("Failed to prepare application for bundling.")
        sys.exit(1)
    
    # Step 4: Create bundle
    print("\nStep 4: Creating executable bundle...")
    result = subprocess.run([sys.executable, "bundle.py"], check=False)
    
    if result.returncode != 0:
        print("Failed to create executable bundle.")
        sys.exit(1)
    
    print("\nBundle process complete!")
    print("You can find the bundled application in: dist/FileAlchemy/")
    
    launcher = "FileAlchemy.bat" if system == "Windows" else "FileAlchemy.sh"
    print(f"To run the application, execute: dist/FileAlchemy/{launcher}")
    
    if system == "Windows":
        input("\nPress Enter to exit...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 