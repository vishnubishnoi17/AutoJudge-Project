"""
Flask Web Application for AutoJudge
Provides a user-friendly web interface for programming problem difficulty prediction.

This application:
- Serves an HTML interface for input
- Accepts problem descriptions via API
- Returns difficulty predictions (class and score on 0-100 scale)
- Supports both classification and regression predictions
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import numpy as np

# Add parent directory to Python path to import src modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predict import load_predictor

# Initialize Flask application
app = Flask(__name__)

# Global variable to store predictor instance
predictor = None

# Load predictor at startup
try:
    print("Loading AutoJudge prediction models...")
    predictor = load_predictor('../models')  # Adjust path relative to app directory
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("Please ensure models are trained by running train_classifier.py and train_regressor.py")
    predictor = None


@app.route('/')
def index():
    """
    Render the main application page.
    
    Returns:
        HTML:  Main interface for problem difficulty prediction
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction API requests.
    
    Expected JSON input:
    {
        "title": "Problem title",
        "description": "Problem description",
        "input_description": "Input format",
        "output_description": "Output format"
    }
    
    Returns:
        JSON:  Prediction results or error message
        {
            "success": true,
            "predicted_class": "Medium",
            "predicted_score": 55. 23,
            "score_interpretation": "Medium",
            "probabilities": {
                "Easy": 0.15,
                "Medium": 0.70,
                "Hard": 0.15
            }
        }
    """
    # Check if models are loaded
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please train the models first by running train_classifier.py and train_regressor.py'
        }), 500
    
    try:
        # Get input data from request
        data = request.get_json()
        
        # Extract fields (with defaults for missing fields)
        title = data.get('title', '')
        description = data.get('description', '')
        input_description = data.get('input_description', '')
        output_description = data.get('output_description', '')
        
        # Validate input - at minimum, description should be provided
        if not description or description.strip() == '':
            return jsonify({
                'success': False,
                'error':  'Problem description is required and cannot be empty.'
            }), 400
        
        # Make prediction
        print(f"Making prediction for problem: {title[: 50]}...")
        results = predictor.predict(
            title=title,
            description=description,
            input_description=input_description,
            output_description=output_description
        )
        
        # Format response
        response = {
            'success': True,
            'predicted_class': results['predicted_class'],
            'predicted_score':  results['predicted_score'],
            'score_interpretation': results['score_interpretation']
        }
        
        # Add probability distribution if available
        if results['probabilities'] is not None:
            probs = results['probabilities']
            response['probabilities'] = {
                'Easy': round(float(probs[0]), 4),
                'Medium': round(float(probs[1]), 4),
                'Hard': round(float(probs[2]), 4)
            }
        
        print(f"✓ Prediction complete:  {results['predicted_class']}, Score: {results['predicted_score']}/100")
        
        return jsonify(response)
    
    except Exception as e:  
        # Log error and return error response
        print(f"✗ Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint for monitoring application status.
    
    Returns:
        JSON: Application health status
        {
            "status": "healthy",
            "models_loaded": true,
            "scale":  "0-100"
        }
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None,
        'scale': '0-100',
        'version': '2.0'
    })


@app.route('/about')
def about():
    """
    Provide information about the application and models.
    
    Returns:
        JSON: Application and model information
    """
    info = {
        'application': 'AutoJudge - Programming Problem Difficulty Predictor',
        'version': '2.0',
        'scale': '0-100 (updated from 0-10)',
        'models':  {
            'classifier': 'Predicts difficulty class (Easy/Medium/Hard)',
            'regressor': 'Predicts difficulty score (0-100)',
            'supported_algorithms': [
                'Logistic Regression',
                'Random Forest',
                'SVM',
                'LightGBM',
                'XGBoost',
                'Gradient Boosting'
            ]
        },
        'features': [
            'Text length analysis',
            'Mathematical notation detection',
            'Algorithm keyword extraction',
            'TF-IDF text vectorization'
        ]
    }
    
    return jsonify(info)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error':  'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success':  False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    """
    Run the Flask development server.
    
    Configuration:
    - debug=True:  Enable debug mode for development
    - host='0.0.0.0': Accept connections from any IP
    - port=5000: Run on port 5000
    """
    print("="*60)
    print("AutoJudge Web Application")
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)