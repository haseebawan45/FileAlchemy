// Dark mode toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('Theme.js loaded');
    
    // Initialize theme based on saved preference
    initializeTheme();
    
    // Set up theme toggle and dropdown functionality if they exist in the page
    setupThemeControls();
});

// Initialize theme based on saved preference
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    console.log('Initializing with theme:', savedTheme);
    setTheme(savedTheme, true); // true = skip animation on initial load
}

// Set up theme toggle and dropdown functionality
function setupThemeControls() {
    const themeToggle = document.getElementById('theme-toggle');
    const themeOptions = document.querySelectorAll('.theme-option');
    
    if (themeToggle) {
        console.log('Theme toggle found');
        // Update toggle button to match current theme
        updateToggleButton();
        
        // Toggle button click handler (cycles between light and dark)
        themeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            console.log('Toggle clicked, switching to:', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Add animation class for the toggle button
            themeToggle.classList.add('animating');
            setTimeout(() => {
                themeToggle.classList.remove('animating');
            }, 600);
            
            setTheme(newTheme, false);
            
            // Update active state in dropdown
            themeOptions.forEach(opt => opt.classList.remove('active'));
            document.querySelector(`.theme-option[data-theme="${newTheme}"]`)?.classList.add('active');
        });
    } else {
        console.log('Theme toggle not found');
    }
    
    // Theme option buttons click handlers
    if (themeOptions.length > 0) {
        console.log('Theme options found:', themeOptions.length);
        // Set active class on the current theme option
        const savedTheme = localStorage.getItem('theme') || 'light';
        
        // Remove active class from all options
        themeOptions.forEach(opt => opt.classList.remove('active'));
        // Add active class to saved theme option
        document.querySelector(`.theme-option[data-theme="${savedTheme}"]`)?.classList.add('active');
        
        themeOptions.forEach(option => {
            option.addEventListener('click', function() {
                const selectedTheme = this.getAttribute('data-theme');
                console.log('Option clicked:', selectedTheme);
                
                // Remove active class from all options
                themeOptions.forEach(opt => opt.classList.remove('active'));
                // Add active class to selected option
                this.classList.add('active');
                
                localStorage.setItem('theme', selectedTheme);
                setTheme(selectedTheme, false);
            });
        });
    } else {
        console.log('No theme options found');
    }
}

// Set theme on the document and update UI
function setTheme(theme, skipAnimation) {
    console.log('Setting theme to:', theme);
    
    // Create and append transition overlay for smooth animation
    if (!skipAnimation) {
        createTransitionOverlay();
    }
    
    // Apply theme to HTML element
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        // Forcing CSS variable updates for index.html
        document.documentElement.style.setProperty('--bg-color', 'linear-gradient(135deg, #1a1a2e, #16213e)');
        document.documentElement.style.setProperty('--text-color', '#e6e6e6');
        document.documentElement.style.setProperty('--heading-color', '#ecf0f1');
        document.documentElement.style.setProperty('--card-bg', 'rgba(30, 39, 46, 0.85)');
    } else {
        document.documentElement.setAttribute('data-theme', '');
        // Forcing CSS variable updates for index.html
        document.documentElement.style.setProperty('--bg-color', 'linear-gradient(135deg, #f5f7fa, #c3cfe2)');
        document.documentElement.style.setProperty('--text-color', '#444');
        document.documentElement.style.setProperty('--heading-color', '#2C3E50');
        document.documentElement.style.setProperty('--card-bg', 'rgba(255, 255, 255, 0.8)');
    }
    
    // Update toggle button appearance
    updateToggleButton();
    
    // Broadcast theme change event for other scripts
    const event = new CustomEvent('themeChanged', { detail: { theme } });
    document.dispatchEvent(event);
}

// Create a transition overlay for smooth theme changes
function createTransitionOverlay() {
    // Check if an overlay already exists and remove it
    const existingOverlay = document.getElementById('theme-transition-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    // Add theme-changing class to html element
    document.documentElement.classList.add('theme-changing');
    
    // Create new overlay
    const overlay = document.createElement('div');
    overlay.id = 'theme-transition-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    overlay.style.backdropFilter = 'blur(5px)';
    overlay.style.zIndex = '9999';
    overlay.style.opacity = '0';
    overlay.style.pointerEvents = 'none';
    overlay.style.transition = 'opacity 0.5s ease';
    
    document.body.appendChild(overlay);
    
    // Trigger animation
    setTimeout(() => {
        overlay.style.opacity = '0.2';
        
        setTimeout(() => {
            overlay.style.opacity = '0';
            
            // Remove overlay after animation completes
            setTimeout(() => {
                overlay.remove();
                // Remove theme-changing class
                document.documentElement.classList.remove('theme-changing');
            }, 500);
        }, 300);
    }, 10);
}

// Update toggle button appearance based on current theme
function updateToggleButton() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const lightIcon = themeToggle.querySelector('.light-icon');
    const darkIcon = themeToggle.querySelector('.dark-icon');
    
    if (lightIcon && darkIcon) {
        lightIcon.style.display = 'block';
        darkIcon.style.display = 'block';
        
        // We're now using CSS transitions on the icons themselves
        // The opacity and transform are controlled in theme.css
    }
} 