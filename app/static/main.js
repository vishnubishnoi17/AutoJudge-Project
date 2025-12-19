document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const darkModeToggle = document.getElementById('darkModeToggle');
    
    // Dark Mode Logic
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }

    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        darkModeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results/errors
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        
        // Show loading state
        document.querySelector('.btn-text').style.display = 'none';
        document.querySelector('.btn-loader').style.display = 'inline-block';
        document.querySelector('.btn-loader-text').style.display = 'inline-block';
        predictBtn.disabled = true;

        // Get form data
        const formData = {
            title: document.getElementById('title').value,
            description: document.getElementById('description').value,
            input_description: document.getElementById('input_description').value,
            output_description: document.getElementById('output_description').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Display results
            displayResults(data);
        } catch (error) {
            errorMessage.textContent = error.message;
            errorDiv.style.display = 'block';
        } finally {
            // Reset button state
            document.querySelector('.btn-text').style.display = 'inline-block';
            document.querySelector('.btn-loader').style.display = 'none';
            document.querySelector('.btn-loader-text').style.display = 'none';
            predictBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        form.reset();
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
    });

    function displayResults(data) {
        // Set class and score
        const classElem = document.getElementById('predictedClass');
        classElem.textContent = data.predicted_class;
        
        // Reset classes
        classElem.className = 'display-6 fw-bold';
        
        if (data.predicted_class === 'Easy') classElem.classList.add('text-success');
        else if (data.predicted_class === 'Medium') classElem.classList.add('text-warning');
        else if (data.predicted_class === 'Hard') classElem.classList.add('text-danger');

        document.getElementById('predictedScore').textContent = data.predicted_score;

        // Display probabilities if available
        if (data.probabilities) {
            document.getElementById('probabilitiesSection').style.display = 'block';
            
            const probEasy = data.probabilities.Easy * 100;
            const probMedium = data.probabilities.Medium * 100;
            const probHard = data.probabilities.Hard * 100;

            document.getElementById('probEasy').style.width = probEasy + '%';
            document.getElementById('probEasyValue').textContent = probEasy.toFixed(1) + '%';

            document.getElementById('probMedium').style.width = probMedium + '%';
            document.getElementById('probMediumValue').textContent = probMedium.toFixed(1) + '%';

            document.getElementById('probHard').style.width = probHard + '%';
            document.getElementById('probHardValue').textContent = probHard.toFixed(1) + '%';
        } else {
            document.getElementById('probabilitiesSection').style.display = 'none';
        }

        // Show results
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});