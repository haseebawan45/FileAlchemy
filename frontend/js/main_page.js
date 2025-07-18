import API from './api.js';

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

        try {
            console.log("Sending request to server...");

            // Show upload progress container
            uploadProgressContainer.style.display = "block";
            uploadStatus.textContent = "Uploading...";
            uploadProgressBar.style.width = "0%";
            uploadPercentage.textContent = "0%";
            uploadSpeed.textContent = `Starting upload...`;

            // Call the API to convert the file
            const result = await API.convertFile(file, targetFormat);

            if (result.error) {
                statusText.textContent = `Error: ${result.error}`;
                progressBar.style.backgroundColor = "#ff5252";
                resetConversionButtons();
                return;
            }

            const taskId = result.task_id;

            // Start checking progress
            checkConversionProgress(taskId, statusText, progressBar);
        } catch (error) {
            console.error("Error during conversion:", error);
            statusText.textContent = "An error occurred during conversion. Please try again.";
            progressBar.style.backgroundColor = "#ff5252";
            resetConversionButtons();
        }
    }

    async function checkConversionProgress(taskId, statusText, progressBar) {
        try {
            const progress = await API.checkProgress(taskId);

            if (progress.progress < 100) {
                // Update progress bar
                progressBar.style.width = `${progress.progress}%`;
                statusText.textContent = progress.status;

                // Check again in 1 second
                setTimeout(() => checkConversionProgress(taskId, statusText, progressBar), 1000);
            } else {
                // Conversion complete
                progressBar.style.width = "100%";
                statusText.textContent = "Conversion complete!";

                // Show download button
                const downloadButton = document.createElement("a");
                downloadButton.href = API.getDownloadUrl(taskId);
                downloadButton.className = "download-button";
                downloadButton.innerHTML = '<i class="fas fa-download"></i> Download';
                downloadButton.download = "";

                // Clear the status div and add the download button
                const statusDiv = statusText.parentElement;
                statusDiv.innerHTML = "";
                statusDiv.appendChild(downloadButton);

                // Reset conversion buttons
                resetConversionButtons();
            }
        } catch (error) {
            console.error("Error checking progress:", error);
            statusText.textContent = "An error occurred while checking progress. Please try again.";
            progressBar.style.backgroundColor = "#ff5252";
            resetConversionButtons();
        }
    }

    function resetConversionButtons() {
        // Reset conversion options display
        conversionOptions.style.display = "block";

        // Reset all buttons
        const buttons = document.querySelectorAll(".conversion-option");
        buttons.forEach(button => {
            button.classList.remove("loading");
            button.disabled = false;

            // Remove spinner if exists
            const spinner = button.querySelector(".spinner");
            if (spinner) {
                button.removeChild(spinner);
            }
        });
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " bytes";
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        else if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
        else return (bytes / 1073741824).toFixed(1) + " GB";
    }

    function formatSpeed(bytesPerSecond) {
        if (bytesPerSecond < 1024) return bytesPerSecond.toFixed(1) + " B/s";
        else if (bytesPerSecond < 1048576) return (bytesPerSecond / 1024).toFixed(1) + " KB/s";
        else return (bytesPerSecond / 1048576).toFixed(1) + " MB/s";
    }

    function formatTime(seconds) {
        if (seconds < 60) return seconds.toFixed(0) + " seconds";
        else return Math.floor(seconds / 60) + " min " + (seconds % 60).toFixed(0) + " sec";
    }
}); 