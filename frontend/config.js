// Configuration for FileAlchemy Frontend
// Update RAILWAY_API_URL with your actual Railway deployment URL

const CONFIG = {
    // Railway API URL - Update this after deploying to Railway
    RAILWAY_API_URL: 'https://filealchemy-api.up.railway.app',

    // File size limits
    MAX_FILE_SIZE_MB: 25,
    MAX_FILE_SIZE_BYTES: 25 * 1024 * 1024,

    // UI Configuration
    SUPPORTED_FORMATS: {
        "pdf": ["docx", "xlsx", "pptx", "txt", "html"],
        "docx": ["pdf", "html", "md"],
        "xlsx": ["csv", "json", "xml"],
        "pptx": ["pdf", "images", "video"],
        "txt": ["pdf", "docx", "md"],
        "html": ["pdf", "docx"],
        "jpg": ["pdf", "text (ocr)", "png", "webp", "gif", "bmp", "tiff", "compressed", "heic"],
        "jpeg": ["pdf", "text (ocr)", "png", "webp", "gif", "bmp", "tiff", "compressed", "heic"],
        "png": ["pdf", "text (ocr)", "jpg", "webp", "gif", "bmp", "tiff", "compressed", "heic"],
        "webp": ["pdf", "text (ocr)", "png", "jpg", "gif", "compressed", "heic"],
        "gif": ["pdf", "png", "jpg", "webp", "heic"],
        "bmp": ["pdf", "text (ocr)", "png", "jpg", "webp", "heic"],
        "tiff": ["pdf", "text (ocr)", "png", "jpg", "webp", "heic"],
        "heic": ["jpg", "png", "webp", "pdf", "tiff", "bmp"],
        "csv": ["json", "xlsx"],
        "json": ["xml"],
        "xml": ["json"],
        "md": ["html", "docx"],
        // Video formats for audio extraction
        "mp4": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "avi": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "mov": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "mkv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "webm": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "flv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "wmv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        // Audio-to-audio conversion
        "mp3": ["wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "wav": ["mp3", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "aac": ["mp3", "wav", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "ogg": ["mp3", "wav", "aac", "flac", "opus", "ac3", "m4a", "alac"],
        "flac": ["mp3", "wav", "aac", "ogg", "opus", "ac3", "m4a", "alac"],
        "opus": ["mp3", "wav", "aac", "ogg", "flac", "ac3", "m4a", "alac"],
        "ac3": ["mp3", "wav", "aac", "ogg", "flac", "opus", "m4a", "alac"],
        "m4a": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "alac"],
        "wma": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"]
    },

    // Error messages
    ERROR_MESSAGES: {
        FILE_TOO_LARGE: 'File size exceeds the 25MB limit for Railway free tier.',
        SERVER_CAPACITY: 'Server is at capacity. Please try again in a few moments.',
        QUEUE_FULL: 'Server queue is full. Please try again later.',
        NETWORK_ERROR: 'Network error. Please check your connection and try again.',
        UNSUPPORTED_FORMAT: 'File format not supported for this conversion.'
    }
};

// Make CONFIG available globally
window.FILEALCHEMY_CONFIG = CONFIG;