// API configuration
const API = {
    // Use environment-based API URL
    BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8080'  // Local development
        : 'https://filealchemy-api.fly.dev',  // Production URL

    // API endpoints
    ENDPOINTS: {
        CONVERT: '/convert/',
        DOWNLOAD: '/download/',
        PROGRESS: '/conversion-progress/'
    },

    // Method to convert a file
    async convertFile(file, targetFormat) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.BASE_URL}${this.ENDPOINTS.CONVERT}?target_format=${targetFormat}`, {
                method: 'POST',
                body: formData,
            });

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
    }
};

export default API; 