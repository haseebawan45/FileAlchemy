# FileAlchemy - Your Ultimate Document Conversion Solution

FileAlchemy is a powerful file conversion application that allows you to convert between various document formats with high accuracy and formatting preservation.

## Features

- **PDF Conversions**: Convert PDFs to Word, Excel, PowerPoint, Text, and HTML
- **Document Conversions**: Convert Word documents to PDF, HTML, and Markdown
- **Spreadsheet Conversions**: Convert Excel files to CSV, JSON, and XML
- **Presentation Conversions**: Convert PowerPoint files to PDF, images, and video
- **Image Conversions**: Convert between image formats (JPG, PNG, WebP, GIF, etc.)
- **OCR Capabilities**: Extract text from images with advanced preprocessing
- **Batch Conversion**: Process multiple files efficiently
- **Local First**: All conversions happen locally on your machine for privacy

## Server Deployment Guide

### Prerequisites

1. Python 3.7+ and pip
2. Required packages (see requirements.txt)
3. For optimal PPTX/DOC conversions: LibreOffice

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HaseebTariq45/FileAlchemy.git
   cd FileAlchemy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install LibreOffice for high-quality document conversion:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install libreoffice
     ```
   
   - **CentOS/RHEL**:
     ```bash
     sudo yum install libreoffice
     ```
     
   - **Windows Server**:
     Download and install from [libreoffice.org](https://www.libreoffice.org/download/download/)

### Configuration

The application automatically detects if it's running in a server environment and will:
- Skip desktop-only conversion methods (like PowerPoint COM automation)
- Use LibreOffice for high-quality document conversion if available
- Fall back to pure Python methods when needed

### Running the Server

Start the server with:
```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

For production deployment, consider using Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
```

### Docker Deployment

For easier deployment, you can use Docker:

1. Build the Docker image:
   ```bash
   docker build -t filealchemy .
   ```

2. Run the container:
   ```bash
   docker run -p 8001:8001 filealchemy
   ```

### Optimizing PPTX Conversions on Servers

For best results with PPTX conversions on servers:

1. **Install LibreOffice**: This provides the highest quality conversions on servers
   ```bash
   sudo apt install libreoffice  # Ubuntu/Debian
   ```

2. **Allocate sufficient memory**: PPTX conversion can be memory-intensive
   ```bash
   # Example: Setting environment variable for Java memory allocation
   export _JAVA_OPTIONS="-Xmx2048m"
   ```

3. **Troubleshooting**:
   - If conversions hang: Check if LibreOffice processes remain running after conversion
   - If permission errors occur: Ensure your web server user has permissions to execute LibreOffice

## Development

### Local Development Setup

For development on Windows with the best conversion quality:
- Install Microsoft Office (for COM automation)
- Install LibreOffice (as fallback)
- Install required Python packages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

FileAlchemy uses various open-source libraries for file conversion. We're grateful to all the developers who maintain these libraries. 