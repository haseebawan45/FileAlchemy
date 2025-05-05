"""
Script to prepare the FileAlchemy application for bundling with PyInstaller
"""
import os
import re
import shutil
import sys

def prepare_server_file():
    """Modify server.py to work properly when bundled with PyInstaller"""
    print("Preparing server.py for bundling...")
    
    # Create a backup of the original server.py
    server_path = "server.py"
    backup_path = "server_pre_bundle_backup.py"
    
    if not os.path.exists(server_path):
        print(f"Error: {server_path} not found!")
        return False
    
    # Create backup
    shutil.copy2(server_path, backup_path)
    print(f"Created backup of server.py at {backup_path}")
    
    # Read the content of server.py
    with open(server_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add code to handle PyInstaller bundled paths
    pyinstaller_code = """
# PyInstaller bundling support
def get_bundled_path(relative_path):
    '''Get the correct path for bundled resources in PyInstaller'''
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
        return os.path.join(base_path, relative_path)
    else:
        # Running in a normal Python environment
        return relative_path

# Update paths for bundled environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as bundled application
    UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
    CONVERTED_DIR = os.path.join(SCRIPT_DIR, 'converted')
    TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
    
    # Set library paths in bundled environment
    import sys
    import platform
    
    # Platform-specific handling
    system = platform.system()
    if system == "Windows":
        # Windows-specific paths
        os.environ["PATH"] = f"{os.path.join(sys._MEIPASS, 'ffmpeg')};{os.environ['PATH']}"
    else:
        # Unix-like systems (Linux/macOS)
        os.environ["PATH"] = f"{os.path.join(sys._MEIPASS, 'ffmpeg')}:{os.environ['PATH']}"
    
    # Create directories if they don't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CONVERTED_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print(f"Running in bundled mode")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Converted directory: {CONVERTED_DIR}")
    print(f"Temp directory: {TEMP_DIR}")
    
    # Ensure the directories are writable
    try:
        test_file_path = os.path.join(UPLOAD_DIR, 'test_write.tmp')
        with open(test_file_path, 'w') as f:
            f.write('test')
        os.remove(test_file_path)
        print("Directory permissions verified: Write access confirmed")
    except Exception as e:
        print(f"WARNING: Directory permission issue: {str(e)}")
        print("The application may not function correctly without write permissions")
"""
    
    # Add the PyInstaller code after imports
    import_pattern = re.compile(r'(import.*?\n+)', re.DOTALL)
    match = import_pattern.search(content)
    if match:
        insert_pos = match.end()
        modified_content = content[:insert_pos] + pyinstaller_code + content[insert_pos:]
    else:
        # If no match, just add the code at the beginning
        modified_content = pyinstaller_code + content
    
    # Modify the FFmpeg path detection to work with PyInstaller bundle
    ffmpeg_pattern = re.compile(r'(local_ffmpeg = os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\), "ffmpeg", "ffmpeg\.exe"\))')
    modified_content = ffmpeg_pattern.sub(r'local_ffmpeg = get_bundled_path(os.path.join("ffmpeg", "ffmpeg.exe"))', modified_content)
    
    # Modify imageio_ffmpeg path detection for bundled environment
    imageio_ffmpeg_pattern = re.compile(r'(import imageio_ffmpeg\s+ffmpeg_path = imageio_ffmpeg\.get_ffmpeg_exe\(\))', re.DOTALL)
    if imageio_ffmpeg_pattern.search(modified_content):
        modified_content = imageio_ffmpeg_pattern.sub(
            r'import imageio_ffmpeg\n'
            r'        try:\n'
            r'            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()\n'
            r'        except:\n'
            r'            # In bundled environment, use local FFmpeg\n'
            r'            ffmpeg_path = get_bundled_path(os.path.join("ffmpeg", "ffmpeg.exe"))\n'
            r'            if os.path.exists(ffmpeg_path):\n'
            r'                print(f"Using bundled FFmpeg: {ffmpeg_path}")\n'
            r'            else:\n'
            r'                ffmpeg_path = None', 
            modified_content
        )
    
    # Modify static files directory
    static_pattern = re.compile(r'(app\.mount\("/", StaticFiles\(directory=".", html=True\), name="static"\))')
    modified_content = static_pattern.sub(r'app.mount("/", StaticFiles(directory=get_bundled_path("."), html=True), name="static")', modified_content)
    
    # Write the modified content back to server.py
    with open(server_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    print("Successfully modified server.py for bundling")
    return True

def check_required_files():
    """Check if all required files for bundling are present"""
    print("Checking for required files...")
    
    required_files = [
        "server.py",
        "index.html",
        "js/main_page.js",
        "css/main_page.css",
        "pages/main_page.html"
    ]
    
    # Add platform-specific requirements
    import platform
    system = platform.system()
    
    if system == "Windows":
        required_files.append("ffmpeg/ffmpeg.exe")
    else:  # Linux or macOS
        required_files.append("ffmpeg/ffmpeg")
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"ERROR: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure all required files are present before bundling.")
        return False
    
    print("All required files are present.")
    return True

def main():
    """Main function to prepare the application for bundling"""
    print("Preparing FileAlchemy for bundling...")
    
    # Check for required files
    if not check_required_files():
        print("\nFailed to prepare the application for bundling due to missing files.")
        sys.exit(1)
    
    # Create required directories
    dirs_to_create = ['uploads', 'converted', 'temp']
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Prepare server.py
    if prepare_server_file():
        print("\nApplication is ready for bundling!")
        print("Now run: python bundle.py")
    else:
        print("\nFailed to prepare the application for bundling.")
        sys.exit(1)

if __name__ == "__main__":
    main() 