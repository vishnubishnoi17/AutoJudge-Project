"""
Flask Web Application
Provides web interface for problem difficulty prediction
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add parent directory to path
sys.path. append(str(Path(__file__).parent. parent / 'src'))

from predict import load_predictor

app = Flask(__name__)

# Load predictor at startup
try:
    predictor = load_predictor('../models')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models:  {e}")
    predictor = None


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index. html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    """
    if predictor is None:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        title = data.get('title', '')
        description = data.get('description', '')
        input_description = data.get('input_description', '')
        output_description = data.get('output_description', '')
        
        # Validate input
        if not description:
            return jsonify({
                'error': 'Problem description is required.'
            }), 400
        
        # Make prediction
        results = predictor.predict(
            title,
            description,
            input_description,
            output_description
        )
        
        # Format response
        response = {
            'success': True,
            'predicted_class': results['predicted_class'],
            'predicted_score':  results['predicted_score']
        }
        
        if results['probabilities'] is not None:
            response['probabilities'] = {
                'Easy': round(float(results['probabilities'][0]), 4),
                'Medium': round(float(results['probabilities'][1]), 4),
                'Hard': round(float(results['probabilities'][2]), 4)
            }
        
        return jsonify(response)
    
    except Exception as e: 
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })


if __name__ == '__main__':
    app. run(debug=True, host='0.0.0.0', port=5000)