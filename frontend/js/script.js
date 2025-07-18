// Format mappings for conversion options
const formatMappings = {
    'jpg': ['pdf', 'text (ocr)', 'png', 'webp', 'gif', 'bmp', 'tiff', 'ico', 'jpeg', 'compressed', 'heic'],
    'jpeg': ['pdf', 'text (ocr)', 'png', 'webp', 'gif', 'bmp', 'tiff', 'ico', 'jpg', 'compressed', 'heic'],
    'png': ['pdf', 'text (ocr)', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'ico', 'compressed', 'heic'],
    'webp': ['pdf', 'text (ocr)', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'ico', 'compressed', 'heic'],
    'gif': ['pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'ico', 'compressed', 'heic'],
    'bmp': ['pdf', 'text (ocr)', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'tiff', 'ico', 'compressed', 'heic'],
    'tiff': ['pdf', 'text (ocr)', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'ico', 'compressed', 'heic'],
    'ico': ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'compressed', 'heic'],
    'heic': ['pdf', 'text (ocr)', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'ico', 'compressed'],
    'pdf': ['docx', 'xlsx', 'pptx', 'txt', 'html'],
    'docx': ['pdf', 'html', 'md'],
    'xlsx': ['csv', 'json', 'xml'],
    'pptx': ['pdf', 'images', 'video'],
    'txt': ['pdf', 'docx', 'md'],
    'html': ['pdf', 'docx'],
    'csv': ['json', 'xlsx'],
    'json': ['xml'],
    'xml': ['json'],
    'md': ['html', 'docx']
};

// Format display names for better UI presentation
const formatDisplayNames = {
    'jpg': 'JPG',
    'jpeg': 'JPEG',
    'png': 'PNG',
    'webp': 'WebP',
    'gif': 'GIF',
    'bmp': 'BMP',
    'tiff': 'TIFF',
    'ico': 'ICO',
    'heic': 'HEIC',
    'pdf': 'PDF',
    'docx': 'Word Document',
    'xlsx': 'Excel Spreadsheet',
    'pptx': 'PowerPoint',
    'txt': 'Text File',
    'html': 'HTML',
    'csv': 'CSV',
    'json': 'JSON',
    'xml': 'XML',
    'md': 'Markdown',
    'compressed': 'Compressed Image',
    'text (ocr)': 'Extract Text (OCR)',
    'images': 'Image Sequence',
    'video': 'Video'
};

// Update available target formats when a file is selected
function updateTargetFormats(file) {
    const targetFormatSelect = document.getElementById('targetFormat');
    targetFormatSelect.innerHTML = ''; // Clear existing options
    
    // Get file extension
    const extension = file.name.split('.').pop().toLowerCase();
    
    // Get available target formats
    const targetFormats = formatMappings[extension] || [];
    
    // Add options to select
    targetFormats.forEach(format => {
        const option = document.createElement('option');
        option.value = format;
        option.textContent = formatDisplayNames[format] || format;
        targetFormatSelect.appendChild(option);
    });
    
    // Enable/disable select based on available formats
    targetFormatSelect.disabled = targetFormats.length === 0;
    
    // Show warning if no formats available
    const warningElement = document.getElementById('formatWarning');
    if (warningElement) {
        warningElement.style.display = targetFormats.length === 0 ? 'block' : 'none';
    }
}

// Initialize the file input handler
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateTargetFormats(e.target.files[0]);
            }
        });
    }
});