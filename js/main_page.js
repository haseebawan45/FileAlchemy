document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const fileInfo = document.getElementById("fileInfo");
    const conversionOptions = document.getElementById("conversionOptions");
    const uploadProgressContainer = document.getElementById("uploadProgressContainer");
    const uploadProgressBar = document.getElementById("uploadProgressBar");
    const uploadStatus = document.getElementById("uploadStatus");
    const uploadPercentage = document.getElementById("uploadPercentage");
    const uploadSpeed = document.getElementById("uploadSpeed");

    // File formats mapping
    const conversionMapping = {
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
        // Add video formats for audio extraction with expanded format options
        "mp4": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "avi": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "mov": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "mkv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "webm": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "flv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "wmv": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        // Audio-to-audio conversion formats
        "mp3": ["wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "wav": ["mp3", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "aac": ["mp3", "wav", "ogg", "flac", "opus", "ac3", "m4a", "alac"],
        "ogg": ["mp3", "wav", "aac", "flac", "opus", "ac3", "m4a", "alac"],
        "flac": ["mp3", "wav", "aac", "ogg", "opus", "ac3", "m4a", "alac"],
        "opus": ["mp3", "wav", "aac", "ogg", "flac", "ac3", "m4a", "alac"],
        "ac3": ["mp3", "wav", "aac", "ogg", "flac", "opus", "m4a", "alac"],
        "m4a": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "alac"],
        "wma": ["mp3", "wav", "aac", "ogg", "flac", "opus", "ac3", "m4a", "alac"]
    };

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];

        if (file) {
            const fileExtension = file.name.split('.').pop().toLowerCase();
            fileInfo.textContent = `Selected file: ${file.name} (${formatFileSize(file.size)})`;
            fileInfo.style.display = "block";

            if (conversionMapping[fileExtension]) {
                conversionOptions.innerHTML = ""; // Clear previous options
                const heading = document.createElement("h3");
                heading.textContent = "Convert to:";
                conversionOptions.appendChild(heading);

                conversionMapping[fileExtension].forEach(option => {
                    const button = document.createElement("button");
                    button.textContent = option.toUpperCase();
                    button.className = "conversion-option";
                    
                    // Add icon based on format type
                    const icon = document.createElement("i");
                    
                    // Determine icon based on file type
                    if (option === "mp3" || option === "wav" || option === "aac" || option === "ogg" || 
                        option === "flac" || option === "opus" || option === "ac3" || option === "m4a" || option === "alac") {
                        icon.className = "fas fa-music"; // Audio icon
                        button.setAttribute("data-format", "audio");
                    } else if (option === "pdf") {
                        icon.className = "fas fa-file-pdf";
                    } else if (option === "docx") {
                        icon.className = "fas fa-file-word";
                    } else if (option === "xlsx") {
                        icon.className = "fas fa-file-excel";
                    } else if (option === "pptx") {
                        icon.className = "fas fa-file-powerpoint";
                    } else if (option === "txt") {
                        icon.className = "fas fa-file-alt";
                    } else if (option === "html") {
                        icon.className = "fas fa-file-code";
                    } else if (option === "images" || option.includes("jpg") || option.includes("png") || option.includes("webp") || option.includes("gif")) {
                        icon.className = "fas fa-image";
                    } else if (option === "video") {
                        icon.className = "fas fa-video";
                    } else if (option === "text (ocr)") {
                        icon.className = "fas fa-font";
                    } else if (option === "json" || option === "xml") {
                        icon.className = "fas fa-file-code";
                    } else if (option === "md") {
                        icon.className = "fab fa-markdown";
                    } else if (option === "compressed") {
                        icon.className = "fas fa-compress";
                    } else {
                        icon.className = "fas fa-file";
                    }
                    
                    // Prepend icon to button
                    button.prepend(icon, " ");
                    
                    button.addEventListener("click", () => {
                        // Hide the conversion options during upload and conversion
                        // to prevent multiple conversions from being started
                        conversionOptions.style.display = "none";
                        
                        // Add animated loading effect to the button
                        button.classList.add("loading");
                        button.disabled = true;
                        
                        // Add a small spinner to the button
                        const spinner = document.createElement("span");
                        spinner.className = "spinner";
                        button.appendChild(spinner);
                        
                        startConversion(file, option);
                    });
                    conversionOptions.appendChild(button);
                });

                conversionOptions.style.display = "block";
                
                // Ensure the upload progress container is initially hidden
                uploadProgressContainer.style.display = "none";
            } else {
                fileInfo.textContent = `Selected file type (${fileExtension}) is not supported.`;
                conversionOptions.style.display = "none";
            }
        }
    });

    async function startConversion(file, targetFormat) {
        // Create and add loading indicator with progress bar
        const statusDiv = document.createElement("div");
        statusDiv.className = "conversion-status";

        // Create inner elements
        const statusText = document.createElement("div");
        statusText.textContent = `Converting ${file.name} to ${targetFormat}...`;
        statusDiv.appendChild(statusText);

        // Create progress container and bar
        const progressContainer = document.createElement("div");
        progressContainer.className = "progress-container";

        const progressBar = document.createElement("div");
        progressBar.className = "progress-bar";
        progressContainer.appendChild(progressBar);
        statusDiv.appendChild(progressContainer);

        document.body.appendChild(statusDiv);

        // Format data for upload
        const formData = new FormData();
        formData.append("file", file);

        try {
            console.log("Sending request to server...");
            const startTime = new Date();

            // Show upload progress container first
            uploadProgressContainer.style.display = "block";
            uploadStatus.textContent = "Uploading...";
            uploadProgressBar.style.width = "0%";
            uploadPercentage.textContent = "0%";
            uploadSpeed.textContent = `Starting upload...`;
            
            // Ensure the progress container is visible and positioned correctly
            // by scrolling to it if necessary
            uploadProgressContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            // Track upload speed
            let lastLoaded = 0;
            let lastTime = Date.now();
            let uploadStartTime = Date.now();

            // Create XMLHttpRequest to track upload progress
            const xhr = new XMLHttpRequest();
            
            // Add these utility functions at the beginning of the file
            function calculateSpeed(loaded, startTime) {
                const elapsedSeconds = (Date.now() - startTime) / 1000;
                return loaded / elapsedSeconds;
            }

            function formatTimeRemaining(seconds) {
                if (seconds < 60) return `${Math.round(seconds)}s`;
                if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
                return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
            }

            function smoothProgress(current, target, callback) {
                let progress = current;
                const step = (target - current) / 20; // Smaller steps for smoother animation
                
                function updateProgress() {
                    progress += step;
                    if ((step > 0 && progress <= target) || (step < 0 && progress >= target)) {
                        callback(progress);
                        requestAnimationFrame(updateProgress);
                    } else {
                        callback(target);
                    }
                }
                
                requestAnimationFrame(updateProgress);
            }

            // Replace the existing XMLHttpRequest upload progress handling with this:
            xhr.upload.addEventListener("progress", function(event) {
                if (event.lengthComputable) {
                    const currentTime = Date.now();
                    const uploadStartTime = this.uploadStartTime || (this.uploadStartTime = currentTime);
                    const elapsedTime = (currentTime - uploadStartTime) / 1000;
                    
                    // Calculate current speed (bytes per second)
                    const currentSpeed = calculateSpeed(event.loaded, uploadStartTime);
                    
                    // Calculate progress percentage
                    const percentComplete = (event.loaded / event.total) * 100;
                    
                    // Estimate time remaining
                    const remainingBytes = event.total - event.loaded;
                    const timeRemaining = currentSpeed > 0 ? remainingBytes / currentSpeed : 0;
                    
                    // Add uploading animation class
                    uploadProgressBar.classList.add('uploading');
                    
                    // Smooth progress bar update
                    smoothProgress(
                        parseFloat(uploadProgressBar.style.width) || 0,
                        percentComplete,
                        (progress) => {
                            uploadProgressBar.style.width = `${progress}%`;
                            uploadPercentage.textContent = `${Math.round(progress)}%`;
                        }
                    );
                    
                    // Update speed and progress information
                    uploadStatus.textContent = `Uploading... (${formatTimeRemaining(timeRemaining)} remaining)`;
                    uploadSpeed.textContent = `Upload speed: ${formatSpeed(currentSpeed)} | ` +
                        `Uploaded: ${formatFileSize(event.loaded)} of ${formatFileSize(event.total)}`;
                    
                    // Throttle updates for better performance
                    this.lastUpdate = currentTime;
                }
            });

            // Update the upload complete handler
            xhr.upload.addEventListener("load", function() {
                uploadProgressBar.classList.remove('uploading');
                uploadStatus.textContent = "Processing...";
                uploadProgressBar.style.width = "100%";
                uploadPercentage.textContent = "100%";
                
                const uploadDuration = ((Date.now() - this.uploadStartTime) / 1000).toFixed(1);
                uploadSpeed.textContent = `Upload completed in ${uploadDuration}s`;
            });

            // Add error handling for the upload
            xhr.upload.addEventListener("error", function() {
                uploadProgressBar.classList.remove('uploading');
                uploadStatus.textContent = "Upload failed!";
                uploadSpeed.textContent = "An error occurred during upload. Please try again.";
                uploadProgressBar.style.backgroundColor = "#e74c3c";
            });

            // Set up promise to handle the response
            const xhrPromise = new Promise((resolve, reject) => {
                xhr.onreadystatechange = function() {
                    if (xhr.readyState === 4) {
                        if (xhr.status >= 200 && xhr.status < 300) {
                            try {
                                const response = JSON.parse(xhr.responseText);
                                resolve(response);
                            } catch (e) {
                                reject(new Error("Invalid JSON response"));
                            }
                        } else {
                            reject(new Error(`Server returned ${xhr.status}: ${xhr.statusText}`));
                        }
                    }
                };
                
                xhr.onerror = function() {
                    reject(new Error("Network error occurred"));
                };
            });
            
            // Open and send the request
            xhr.open("POST", `http://localhost:8001/convert/?target_format=${targetFormat}`);
            xhr.send(formData);
            
            // Wait for the response
            const data = await xhrPromise;
            const taskId = data.task_id;
            
            if (!taskId) {
                throw new Error("Server did not return a task ID");
            }

            console.log(`Conversion started with task ID: ${taskId}`);
            
            // Poll progress until conversion is complete
            let isComplete = false;
            while (!isComplete) {
                // Fetch progress
                const progressResponse = await fetch(`http://localhost:8001/conversion-progress/${taskId}`);
                
                if (!progressResponse.ok) {
                    throw new Error(`Failed to get progress: ${progressResponse.status}`);
                }
                
                const progressData = await progressResponse.json();
                console.log(`Conversion progress: ${progressData.progress}%, Status: ${progressData.status}`);
                
                // Update progress bar and status text
                progressBar.style.width = `${progressData.progress}%`;
                statusText.textContent = progressData.status;
                
                // Check if conversion is complete
                if (progressData.progress >= 100) {
                    isComplete = true;
                    
                    if (progressData.status.includes("Error")) {
                        // Conversion failed
                        statusDiv.style.backgroundColor = "rgba(231, 76, 60, 0.9)";
                        progressBar.style.backgroundColor = "#e74c3c";
                        throw new Error(progressData.status);
                    }
                    
                    // Calculate processing time
                    const endTime = new Date();
                    const processingTime = ((endTime - startTime) / 1000).toFixed(1);
                    
                    // Get file download link from progress data if available
                    if (progressData.file_path) {
                        // Use file_name from progress data if available
                        const fileName = progressData.file_name || `converted_file.${targetFormat}`;
                        
                        // Update status
                        statusText.textContent = `Conversion complete in ${processingTime}s! Downloading ${fileName}`;
                        statusDiv.style.backgroundColor = "rgba(39, 174, 96, 0.9)";
                        
                        // Download the file directly from the server
                        window.location.href = `http://localhost:8001/download/${taskId}`;
                        
                        // Remove status div after a delay
                        setTimeout(() => {
                            document.body.removeChild(statusDiv);
                            // Show conversion options again
                            conversionOptions.style.display = "block";
                            // Reset all conversion buttons
                            resetConversionButtons();
                            // Hide upload progress
                            uploadProgressContainer.style.display = "none";
                        }, 3000);
                    } else {
                        throw new Error("No file available for download");
                    }
                } else {
                    // Wait before polling again
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }
        } catch (error) {
            console.error("Conversion error:", error);
            
            // Update status div to show error
            let errorMessage = error.message;
            
            // Check for specific error types and provide more helpful messages
            if (errorMessage.includes("FFmpeg") || errorMessage.includes("moviepy")) {
                statusText.textContent = `Error: Video processing requires FFmpeg. Please contact support.`;
                console.error("FFmpeg error details:", errorMessage);
            } else {
                statusText.textContent = `Error: ${errorMessage}`;
            }
            
            statusDiv.style.backgroundColor = "rgba(231, 76, 60, 0.9)";
            progressBar.style.backgroundColor = "#e74c3c";
            progressBar.style.width = "100%";
            
            // Hide upload progress
            uploadProgressContainer.style.display = "none";
            
            setTimeout(() => {
                document.body.removeChild(statusDiv);
                // Show conversion options again
                conversionOptions.style.display = "block";
                // Reset all conversion buttons
                resetConversionButtons();
            }, 5000);
            alert("Conversion failed: " + errorMessage);
        }
    }
    
    // Function to reset all conversion buttons
    function resetConversionButtons() {
        const buttons = document.querySelectorAll('.conversion-option');
        buttons.forEach(button => {
            button.classList.remove('loading');
            button.disabled = false;
            
            // Remove spinner if exists
            const spinner = button.querySelector('.spinner');
            if (spinner) {
                button.removeChild(spinner);
            }
        });
    }
    
    // Format file size in KB, MB etc.
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Format speed in KB/s, MB/s
    function formatSpeed(bytesPerSecond) {
        return formatFileSize(bytesPerSecond) + '/s';
    }
    
    // Format time in seconds to mm:ss format
    function formatTime(seconds) {
        seconds = Math.round(seconds);
        if (seconds < 60) {
            return `${seconds}s`;
        } else {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds}s`;
        }
    }
});