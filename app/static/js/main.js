/**
 * AutoJudge - Main JavaScript File
 * Version: 2.0
 * Description:  Handles form submission, API calls, and dynamic UI updates
 */

// ============================================
// GLOBAL VARIABLES
// ============================================

// Store start time for calculating processing duration
let predictionStartTime = null;

// ============================================
// DOM ELEMENT REFERENCES
// ============================================

// Form elements
const predictionForm = document.getElementById('predictionForm');
const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const inputDescInput = document.getElementById('input_description');
const outputDescInput = document.getElementById('output_description');

// Button elements
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');

// UI feedback elements
const loadingDiv = document.getElementById('loading');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');

// Result display elements
const scoreValue = document.getElementById('scoreValue');
const scoreCircle = document.getElementById('scoreCircle');
const difficultyClass = document.getElementById('difficultyClass');
const difficultyInterpretation = document.getElementById('difficultyInterpretation');
const difficultyBarFill = document.getElementById('difficultyBarFill');
const probabilityBars = document.getElementById('probabilityBars');
const processingTime = document.getElementById('processingTime');

// ============================================
// EVENT LISTENERS
// ============================================

/**
 * Form submission handler
 * Prevents default form submission and triggers AJAX prediction
 */
predictionForm. addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent page reload
    await makePrediction();
});

/**
 * Clear button handler
 * Resets form and hides results
 */
clearBtn.addEventListener('click', () => {
    clearForm();
});

/**
 * Input validation on description field
 * Provides real-time feedback for required field
 */
descriptionInput.addEventListener('input', () => {
    if (descriptionInput.value.trim().length > 0) {
        descriptionInput.style.borderColor = 'var(--border-color)';
    }
});

// ============================================
// MAIN FUNCTIONS
// ============================================

/**
 * Makes a prediction by sending form data to the API
 * 
 * Process:
 * 1. Validate input
 * 2. Show loading state
 * 3. Send POST request to /predict endpoint
 * 4. Handle response and display results
 * 5. Handle errors gracefully
 */
async function makePrediction() {
    // Hide previous results and errors
    hideError();
    hideResults();
    
    // Validate required fields
    if (!validateForm()) {
        return;
    }
    
    // Show loading state
    showLoading();
    disableForm();
    
    // Record start time for performance measurement
    predictionStartTime = Date.now();
    
    // Prepare request data
    const requestData = {
        title: titleInput.value.trim(),
        description: descriptionInput.value.trim(),
        input_description: inputDescInput.value.trim(),
        output_description: outputDescInput.value.trim()
    };
    
    try {
        // Send POST request to prediction API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        // Parse JSON response
        const data = await response.json();
        
        // Check if request was successful
        if (response. ok && data.success) {
            // Calculate processing time
            const processingTimeMs = Date.now() - predictionStartTime;
            
            // Display results
            displayResults(data, processingTimeMs);
        } else {
            // Show error message from API
            showError(data. error || 'Prediction failed. Please try again.');
        }
        
    } catch (error) {
        // Handle network or parsing errors
        console.error('Prediction error:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        // Hide loading state and re-enable form
        hideLoading();
        enableForm();
    }
}

/**
 * Validates form inputs before submission
 * 
 * @returns {boolean} True if form is valid, false otherwise
 */
function validateForm() {
    const description = descriptionInput.value.trim();
    
    // Check if description is empty
    if (description === '') {
        showError('Please provide a problem description.');
        descriptionInput.focus();
        descriptionInput.style.borderColor = 'var(--hard-color)';
        return false;
    }
    
    // Check minimum length (at least 20 characters for meaningful prediction)
    if (description.length < 20) {
        showError('Problem description is too short. Please provide more details (at least 20 characters).');
        descriptionInput.focus();
        descriptionInput.style. borderColor = 'var(--hard-color)';
        return false;
    }
    
    return true;
}

/**
 * Displays prediction results in the UI
 * 
 * @param {Object} data - Prediction data from API
 * @param {number} processingTimeMs - Time taken for prediction (milliseconds)
 */
function displayResults(data, processingTimeMs) {
    // Extract data from response
    const score = data.predicted_score;
    const className = data.predicted_class;
    const interpretation = data.score_interpretation;
    const probabilities = data.probabilities;
    
    // Update score display
    scoreValue.textContent = score;
    
    // Update difficulty class
    difficultyClass.textContent = className;
    
    // Update interpretation
    difficultyInterpretation.textContent = interpretation;
    
    // Update difficulty bar width (percentage)
    difficultyBarFill.style. width = `${score}%`;
    
    // Update score circle color based on difficulty
    updateScoreCircleColor(score);
    
    // Display probability distribution if available
    if (probabilities) {
        displayProbabilities(probabilities);
    } else {
        probabilityBars.innerHTML = '<p style="color: var(--text-secondary);">Probability distribution not available for this model.</p>';
    }
    
    // Display processing time
    processingTime. textContent = `${processingTimeMs} ms`;
    
    // Show results container with animation
    resultsContainer.style.display = 'block';
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

/**
 * Updates the color of the score circle based on difficulty score
 * 
 * @param {number} score - Difficulty score (0-100)
 */
function updateScoreCircleColor(score) {
    let gradient;
    
    if (score < 20) {
        // Very Easy - Green
        gradient = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
    } else if (score < 40) {
        // Easy - Light Green to Yellow
        gradient = 'linear-gradient(135deg, #84cc16 0%, #65a30d 100%)';
    } else if (score < 60) {
        // Medium - Yellow to Orange
        gradient = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
    } else if (score < 80) {
        // Hard - Orange to Red
        gradient = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
    } else {
        // Very Hard - Red
        gradient = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
    }
    
    scoreCircle. style.background = gradient;
}

/**
 * Displays probability distribution as visual bars
 * 
 * @param {Object} probabilities - Object with Easy, Medium, Hard probabilities
 */
function displayProbabilities(probabilities) {
    // Clear previous content
    probabilityBars.innerHTML = '';
    
    // Define class order and styling
    const classes = [
        { name: 'Easy', key: 'Easy', class: 'easy' },
        { name: 'Medium', key: 'Medium', class: 'medium' },
        { name: 'Hard', key: 'Hard', class: 'hard' }
    ];
    
    // Create probability bar for each class
    classes.forEach(cls => {
        const probability = probabilities[cls.key];
        const percentage = (probability * 100).toFixed(1);
        
        // Create HTML structure
        const itemDiv = document.createElement('div');
        itemDiv.className = 'probability-item';
        
        itemDiv.innerHTML = `
            <div class="probability-header">
                <div class="probability-label">
                    <div class="probability-icon ${cls. class}"></div>
                    <span>${cls.name}</span>
                </div>
                <span class="probability-value">${percentage}%</span>
            </div>
            <div class="probability-bar-bg">
                <div class="probability-bar ${cls.class}" style="width: ${percentage}%"></div>
            </div>
        `;
        
        probabilityBars.appendChild(itemDiv);
    });
}

/**
 * Clears the form and resets UI to initial state
 */
function clearForm() {
    // Reset form fields
    predictionForm.reset();
    
    // Hide results and errors
    hideResults();
    hideError();
    
    // Reset input styling
    descriptionInput.style.borderColor = 'var(--border-color)';
    
    // Focus on first input
    titleInput.focus();
}

// ============================================
// UI STATE MANAGEMENT FUNCTIONS
// ============================================

/**
 * Shows loading spinner and message
 */
function showLoading() {
    loadingDiv.style.display = 'block';
}

/**
 * Hides loading spinner
 */
function hideLoading() {
    loadingDiv.style.display = 'none';
}

/**
 * Shows error alert with message
 * 
 * @param {string} message - Error message to display
 */
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.style.display = 'flex';
    
    // Scroll to error message
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hides error alert
 */
function hideError() {
    errorAlert.style.display = 'none';
    errorMessage.textContent = '';
}

/**
 * Shows results container
 */
function showResults() {
    resultsContainer.style.display = 'block';
}

/**
 * Hides results container
 */
function hideResults() {
    resultsContainer.style.display = 'none';
}

/**
 * Disables form inputs and buttons during prediction
 */
function disableForm() {
    titleInput.disabled = true;
    descriptionInput.disabled = true;
    inputDescInput.disabled = true;
    outputDescInput.disabled = true;
    predictBtn.disabled = true;
    clearBtn.disabled = true;
}

/**
 * Enables form inputs and buttons after prediction
 */
function enableForm() {
    titleInput.disabled = false;
    descriptionInput. disabled = false;
    inputDescInput.disabled = false;
    outputDescInput.disabled = false;
    predictBtn.disabled = false;
    clearBtn.disabled = false;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Formats milliseconds to a human-readable duration
 * 
 * @param {number} ms - Duration in milliseconds
 * @returns {string} Formatted duration string
 */
function formatDuration(ms) {
    if (ms < 1000) {
        return `${ms} ms`;
    }
    return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Validates email format (utility for future use)
 * 
 * @param {string} email - Email address to validate
 * @returns {boolean} True if valid email format
 */
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the application when DOM is fully loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('AutoJudge v2.0 initialized');
    
    // Focus on first input field
    titleInput.focus();
    
    // Check if models are loaded by calling health endpoint
    checkSystemHealth();
});

/**
 * Checks if the prediction models are loaded and ready
 * Displays a warning if models are not available
 */
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (! data.models_loaded) {
            showError('Warning:  Prediction models are not loaded. Please train the models first.');
        } else {
            console.log('✓ System healthy - Models loaded successfully');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================

/**
 * Handle keyboard shortcuts for better UX
 */
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e. metaKey) && e.key === 'Enter') {
        if (! predictBtn.disabled) {
            makePrediction();
        }
    }
    
    // Escape to clear form
    if (e.key === 'Escape') {
        clearForm();
    }
});

// ============================================
// ANALYTICS & LOGGING (Optional)
// ============================================

/**
 * Logs prediction event for analytics (optional)
 * 
 * @param {Object} data - Prediction data
 */
function logPrediction(data) {
    // This function can be expanded to send analytics to a tracking service
    console. log('Prediction logged:', {
        timestamp: new Date().toISOString(),
        score: data.predicted_score,
        class: data.predicted_class
    });
}