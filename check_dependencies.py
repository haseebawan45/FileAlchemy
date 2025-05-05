import importlib
import subprocess
import sys

def check_module(module_name, optional=False):
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        status = "✅ Installed"
    except ImportError:
        status = "❌ Not installed" if not optional else "⚠️ Optional dependency not installed"
    
    return status

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible in PATH."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return f"✅ Installed: {version}"
        else:
            return "❌ Installed but not working correctly"
    except FileNotFoundError:
        return "❌ Not installed or not in PATH"

def main():
    # Print header
    print("\n" + "="*60)
    print("FileAlchemy Dependency Check")
    print("="*60)
    
    # Required dependencies
    required_modules = [
        # Web Framework
        "fastapi", "uvicorn", "aiofiles", 
        # PDF Processing
        "fitz", "pdfplumber", "pdf2image", "pdfkit",
        # Document Processing
        "docx", "pptx", "pandas", "openpyxl",
        # Image Processing
        "PIL", "easyocr", "pytesseract", "skimage",
        # Format Conversion
        "markdown", "xmltodict", "weasyprint", "moviepy", "numpy",
        # Network & Utilities
        "requests", "tqdm", "colorama"
    ]
    
    # Optional dependencies
    optional_modules = [
        "pyngrok",
        "docx2pdf"
    ]
    
    # Check required modules
    print("\n📋 REQUIRED DEPENDENCIES:")
    for module in required_modules:
        status = check_module(module)
        print(f"{module:20} {status}")
    
    # Check optional modules
    print("\n📋 OPTIONAL DEPENDENCIES:")
    for module in optional_modules:
        status = check_module(module, optional=True)
        print(f"{module:20} {status}")
    
    # Check FFmpeg (system dependency)
    print("\n📋 SYSTEM DEPENDENCIES:")
    ffmpeg_status = check_ffmpeg()
    print(f"{'FFmpeg':20} {ffmpeg_status}")
    
    # Summary
    print("\n" + "="*60)
    print("Dependency Check Complete")
    print("="*60 + "\n")
    
    # Tips for missing dependencies
    missing_required = [module for module in required_modules if check_module(module) == "❌ Not installed"]
    missing_optional = [module for module in optional_modules if check_module(module, optional=True) == "⚠️ Optional dependency not installed"]
    
    if missing_required:
        print("📢 ATTENTION: The following required dependencies are missing:")
        for module in missing_required:
            print(f"  - {module}")
        print("\nInstall them using: pip install -r requirements.txt\n")
    
    if missing_optional:
        print("📝 NOTE: The following optional dependencies are missing:")
        for module in missing_optional:
            print(f"  - {module}")
        print("\nInstall them individually as needed.\n")
    
    if "❌" in ffmpeg_status:
        print("📢 FFmpeg is not installed or not in your PATH.")
        print("This is required for video conversion features.")
        print("Install FFmpeg from: https://ffmpeg.org/download.html\n")

if __name__ == "__main__":
    main() 