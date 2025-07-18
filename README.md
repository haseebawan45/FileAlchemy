# FileAlchemy - Your Ultimate Document Conversion Solution

FileAlchemy is a powerful file conversion application that allows you to convert between various document formats with high accuracy and formatting preservation.

## Project Structure

The project is split into two main parts:
- `frontend/`: Contains all the frontend code (HTML, CSS, JS)
- `backend/`: Contains the FastAPI server code

## Deployment Instructions

### Frontend Deployment (GitHub Pages)

1. Create a GitHub repository for the project
2. Push the code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/filealchemy.git
   git push -u origin main
   ```
3. Go to the repository settings on GitHub
4. Navigate to "Pages" in the sidebar
5. Under "Source", select "Deploy from a branch"
6. Select the "main" branch and "/frontend" folder
7. Click "Save"

Your frontend will be available at `https://yourusername.github.io/filealchemy/`

### Backend Deployment (fly.io)

1. Install the Fly CLI:
   ```bash
   # For Windows using PowerShell
   iwr https://fly.io/install.ps1 -useb | iex
   
   # For macOS/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. Login to Fly:
   ```bash
   fly auth login
   ```

3. Navigate to the backend directory:
   ```bash
   cd backend
   ```

4. Deploy the application:
   ```bash
   fly launch
   ```
   - Choose a unique app name (e.g., filealchemy-api)
   - Select a region close to your users
   - Skip PostgreSQL and Redis setup

5. Deploy the app:
   ```bash
   fly deploy
   ```

Your API will be available at `https://your-app-name.fly.dev`

## Features

- **PDF Conversions**: Convert PDFs to Word, Excel, PowerPoint, Text, and HTML
- **Document Conversions**: Convert Word documents to PDF, HTML, and Markdown
- **Spreadsheet Conversions**: Convert Excel files to CSV, JSON, and XML
- **Presentation Conversions**: Convert PowerPoint files to PDF, images, and video
- **Image Conversions**: Convert between image formats (JPG, PNG, WebP, GIF, etc.)
- **OCR Capabilities**: Extract text from images with advanced preprocessing
- **Batch Conversion**: Process multiple files efficiently
- **Local First**: All conversions happen locally on your machine for privacy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

FileAlchemy uses various open-source libraries for file conversion. We're grateful to all the developers who maintain these libraries. 