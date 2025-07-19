// API configuration
const API = {
    // Use environment-based API URL
    BASE_URL: (() => {
        const hostname = window.location.hostname;

        // Local development
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8080';
        }

        // Railway deployment
        if (hostname.includes('github.io') || hostname.includes('railway.app')) {
            return window.FILEALCHEMY_CONFIG?.RAILWAY_API_URL || 'https://filealchemy-api.up.railway.app';
        }

        // Fallback to Railway URL
        return window.FILEALCHEMY_CONFIG?.RAILWAY_API_URL || 'https://filealchemy-api.up.railway.app';
    })(),

    // API endpoints
    ENDPOINTS: {
        CONVERT: '/convert/',
        DOWNLOAD: '/download/',
        PROGRESS: '/conversion-progress/'
    },

    // Method to convert a file
    async convertFile(file, targetFormat) {
        try {
            // Check file size before sending (25MB limit for Railway)
            const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25MB
            if (file.size > MAX_FILE_SIZE) {
                throw new Error(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds the 25MB limit for Railway free tier.`);
            }

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.BASE_URL}${this.ENDPOINTS.CONVERT}?target_format=${targetFormat}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                if (response.status === 503) {
                    throw new Error('Server is at capacity. Please try again in a few moments.');
                } else if (response.status === 413) {
                    throw new Error('File too large. Maximum size is 25MB.');
                } else if (response.status === 429) {
                    throw new Error('Too many requests. Please wait before trying again.');
                } else {
                    throw new Error(`Server error: ${response.status}`);
                }
            }

            return await response.json();
        } catch (error) {
            console.error('Error converting file:', error);
            throw error;
        }
    },

    // Method to check conversion progress
    async checkProgress(taskId) {
        try {
            const response = await fetch(`${this.BASE_URL}${this.ENDPOINTS.PROGRESS}${taskId}`);
            return await response.json();
        } catch (error) {
            console.error('Error checking progress:', error);
            throw error;
        }
    },

    // Method to download the converted file
    getDownloadUrl(taskId) {
        return `${this.BASE_URL}${this.ENDPOINTS.DOWNLOAD}${taskId}`;
    },

    // Method to check queue status
    async getQueueStatus() {
        try {
            const response = await fetch(`${this.BASE_URL}/queue-status`);
            return await response.json();
        } catch (error) {
            console.error('Error checking queue status:', error);
            return null;
        }
    },

    // Method to check resource status
    async getResourceStatus() {
        try {
            const response = await fetch(`${this.BASE_URL}/resource-status`);
            return await response.json();
        } catch (error) {
            console.error('Error checking resource status:', error);
            return null;
        }
    }
};

export default API; 