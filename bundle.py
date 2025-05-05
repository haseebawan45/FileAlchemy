"""
Bundle script for FileAlchemy - Creates a standalone executable using PyInstaller
"""
import os
import sys
import shutil
import subprocess
import glob
import platform

def create_bundle():
    """Create a standalone executable bundle of the FileAlchemy web application"""
    print("Starting FileAlchemy bundling process...")
    
    # Check Python version
    py_version = platform.python_version_tuple()
    py_version_str = platform.python_version()
    print(f"Python version: {py_version_str}")
    
    if int(py_version[0]) < 3 or (int(py_version[0]) == 3 and int(py_version[1]) < 7):
        print(f"ERROR: Python 3.7 or higher is required. You are using Python {py_version_str}")
        print("Please upgrade your Python installation or use a different environment.")
        return False
    
    # Check for required files
    required_files = ["server.py", "index.html"]
    
    # Add platform-specific requirements
    system = platform.system()
    print(f"Operating system: {system}")
    
    if system == "Windows":
        required_files.append("ffmpeg/ffmpeg.exe")
    elif system == "Linux":
        required_files.append("ffmpeg/ffmpeg")
    elif system == "Darwin":  # macOS
        required_files.append("ffmpeg/ffmpeg")
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"ERROR: The following required files are missing: {', '.join(missing_files)}")
        print("Please ensure all required files are present before bundling.")
        return False
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("PyInstaller installed successfully.")
        except Exception as e:
            print(f"Failed to install PyInstaller: {str(e)}")
            print("Please install PyInstaller manually: pip install pyinstaller")
            return False
    
    # Check for other critical dependencies
    critical_deps = ["pillow", "pillow-heif", "fastapi", "uvicorn"]
    for dep in critical_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "show", dep], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            print(f"WARNING: {dep} is not installed. This may cause issues in the bundled application.")
            print(f"Consider installing it: pip install {dep}")
    
    # Create a directory for temporary files
    build_dir = "build_files"
    os.makedirs(build_dir, exist_ok=True)
    
    # Create a spec file for PyInstaller
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Get the base path
import os
import platform
import sys
base_path = os.path.abspath(os.path.dirname(__file__))

# Define all data files to include
data_files = [
    # HTML, CSS, JavaScript files
    ('index.html', '.'),
    ('pages/*.html', 'pages'),
    ('css/*.css', 'css'),
    ('js/*.js', 'js'),
    ('assets/*', 'assets'),
    ('images/*', 'images'),
]

# Add platform-specific files
system = platform.system()
if system == "Windows":
    data_files.append(('ffmpeg/*.exe', 'ffmpeg'))
elif system in ["Linux", "Darwin"]:
    data_files.append(('ffmpeg/ffmpeg', 'ffmpeg'))

# Documentation
data_files.append(('build_files/README.txt', '.'))

# Create empty directories for uploads and conversions
data_files.append(('README.md', '.'))  # Just to have a file to copy (will be ignored)

# Add all data files to the bundle
added_files = []
for src_glob, dst_dir in data_files:
    src_paths = glob.glob(os.path.join(base_path, src_glob))
    for src in src_paths:
        filename = os.path.basename(src)
        dst = os.path.join(dst_dir, filename)
        added_files.append((src, dst))

# Create directories for runtime file storage
dirs_to_create = ['uploads', 'converted', 'temp']
for dir_name in dirs_to_create:
    os.makedirs(os.path.join(base_path, 'dist', 'FileAlchemy', dir_name), exist_ok=True)

a = Analysis(
    ['server.py'],
    pathex=[base_path],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi.openapi.docs',
        'fastapi.staticfiles',
        'PIL._tkinter_finder',  # For Pillow
        'moviepy',
        'moviepy.editor',
        'moviepy.video.io.VideoFileClip',
        'pillow_heif',
        'reportlab',
        'reportlab.pdfgen',
        'reportlab.lib.pagesizes',
        'easyocr',
        'pdfkit',
        'pdfplumber',
        'weasyprint',
        'markdown',
        'numpy',
        'pandas',
        'python-docx',
        'img2pdf',
        'imageio_ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FileAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/filealchemy.png',
    version='file_version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FileAlchemy',
)

# Create version info file for Windows
if system == "Windows":
    version_file_path = os.path.join(build_dir, "file_version_info.txt")
    version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'FileAlchemy'),
           StringStruct(u'FileDescription', u'FileAlchemy Conversion Suite'),
           StringStruct(u'FileVersion', u'1.0.0'),
           StringStruct(u'InternalName', u'FileAlchemy'),
           StringStruct(u'LegalCopyright', u'Copyright (c) 2025 FileAlchemy'),
           StringStruct(u'OriginalFilename', u'FileAlchemy.exe'),
           StringStruct(u'ProductName', u'FileAlchemy'),
           StringStruct(u'ProductVersion', u'1.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    with open(version_file_path, "w") as f:
        f.write(version_info)
    print(f"Created version info file at {version_file_path}")
else:
    # Replace version info reference with empty dict for non-Windows platforms
    spec_content = spec_content.replace("version='file_version_info.txt',", "")
    
# Add platform-specific console flag
if system == "Darwin":  # macOS
    spec_content = spec_content.replace("console=True,", "console=False,")

# Write the spec file
spec_path = os.path.join(build_dir, "FileAlchemy.spec")
with open(spec_path, "w") as f:
    f.write(spec_content)

print(f"Created PyInstaller spec file at {spec_path}")

# Create a platform-specific launch script
if system == "Windows":
    # Windows batch file
    batch_path = os.path.join(build_dir, "FileAlchemy.bat")
    
    # Check if the batch file already exists
    if not os.path.exists(batch_path):
        print(f"Creating Windows launch script at {batch_path}")
        batch_content = """@echo off
echo ===================================
echo    FileAlchemy Conversion Suite
echo ===================================
echo.
echo Starting FileAlchemy Server...
echo.
echo The web interface will open automatically in your default browser.
echo.
echo NOTE: Keep this window open while using FileAlchemy.
echo       Close this window to shut down the server when you're done.
echo.
echo Opening browser...
timeout /t 2 > nul
start "" http://localhost:8001
echo.
echo Server is running...
echo.
FileAlchemy.exe
echo.
echo Server has been shut down.
echo Thank you for using FileAlchemy!
pause
"""
        with open(batch_path, "w") as f:
            f.write(batch_content)
    else:
        print(f"Using existing Windows launch script: {batch_path}")
        
    launcher_path = batch_path
    launcher_dest = "dist/FileAlchemy/FileAlchemy.bat"
    
elif system in ["Linux", "Darwin"]:
    # Unix shell script
    shell_path = os.path.join(build_dir, "FileAlchemy.sh")
    
    if not os.path.exists(shell_path):
        print(f"Creating Unix launch script at {shell_path}")
        shell_content = """#!/bin/bash
echo "==================================="
echo "   FileAlchemy Conversion Suite"
echo "==================================="
echo
echo "Starting FileAlchemy Server..."
echo
echo "The web interface will open automatically in your default browser."
echo
echo "NOTE: Keep this terminal open while using FileAlchemy."
echo "      Press Ctrl+C to shut down the server when you're done."
echo
echo "Opening browser..."
sleep 2

# Attempt to open browser based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8001
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux with desktop
    xdg-open http://localhost:8001 &>/dev/null || python -m webbrowser http://localhost:8001 || echo "Please open http://localhost:8001 in your browser"
else
    # Fallback
    python -m webbrowser http://localhost:8001 || echo "Please open http://localhost:8001 in your browser"
fi

echo
echo "Server is running..."
echo

# Run the application
./FileAlchemy

echo
echo "Server has been shut down."
echo "Thank you for using FileAlchemy!"
read -p "Press Enter to exit..."
"""
        with open(shell_path, "w") as f:
            f.write(shell_content)
        
        # Make the shell script executable
        os.chmod(shell_path, 0o755)
    else:
        print(f"Using existing Unix launch script: {shell_path}")
        
    launcher_path = shell_path
    launcher_dest = "dist/FileAlchemy/FileAlchemy.sh"
    
else:
    print(f"Warning: Unsupported operating system: {system}")
    print("No launch script will be created.")
    launcher_path = None
    launcher_dest = None

print(f"Launch script ready at {launcher_path}")

# Run PyInstaller
print("Running PyInstaller to create the bundle (this may take several minutes)...")

# Determine PyInstaller command based on platform
pyinstaller_cmd = ["pyinstaller", "--clean"]

# Additional platform-specific flags
if system == "Windows":
    # Add Windows-specific icon
    pyinstaller_cmd.extend(["--icon=assets/filealchemy.png"])
elif system == "Darwin":  # macOS
    # Add macOS-specific options
    pyinstaller_cmd.extend(["--windowed", "--icon=assets/filealchemy.png"])

# Add spec file to command
pyinstaller_cmd.append(spec_path)

print(f"Running command: {' '.join(pyinstaller_cmd)}")
result = subprocess.run(pyinstaller_cmd, check=False)

if result.returncode == 0:
    print("PyInstaller completed successfully!")
    
    # Copy the batch file to the dist directory
    if launcher_path:
        shutil.copy(launcher_path, launcher_dest)
        print(f"Copied launch script to {launcher_dest}")
    
    # Create a README file in the dist directory if it doesn't exist
    readme_src = os.path.join(build_dir, "README.txt")
    readme_dst = os.path.join("dist", "FileAlchemy", "README.txt")
    if os.path.exists(readme_src):
        shutil.copy(readme_src, readme_dst)
        print("Copied README file to the distribution directory")
    
    # Create required directories in the dist directory
    for dir_name in ['uploads', 'converted', 'temp']:
        dir_path = os.path.join("dist", "FileAlchemy", dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory in distribution: {dir_name}")
    
    print("\nBundle created successfully! You can find it in: dist/FileAlchemy/")
    if launcher_path:
        print(f"To run the application, execute: {launcher_path}")
else:
    print("PyInstaller encountered an error. Please check the output above for details.")
    return False

return True

if __name__ == "__main__":
    create_bundle() 