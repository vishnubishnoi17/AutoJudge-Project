"""
Prediction Module
Provides utilities for making predictions on new programming problems.  

This module loads trained models and makes predictions for:  
- Difficulty class (Easy, Medium, Hard)
- Difficulty score (0-100 scale)
- Class probabilities

Supports both classification and regression models including LightGBM and XGBoost.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class ProblemDifficultyPredictor:  
    """
    Unified predictor for programming problem difficulty assessment.
    
    This class:
    - Loads trained classification and regression models
    - Preprocesses input text data
    - Predicts difficulty class and score
    - Provides probability distributions for classes
    """
    
    def __init__(self, classifier_path, regressor_path, 
                 feature_extractor_classifier_path, 
                 feature_extractor_regressor_path,
                 label_encoder_path):
        """
        Initialize the predictor with trained models. 
        
        Args:  
            classifier_path (str): Path to trained classification model (. pkl file)
            regressor_path (str): Path to trained regression model (. pkl file)
            feature_extractor_classifier_path (str): Path to classifier's feature extractor
            feature_extractor_regressor_path (str): Path to regressor's feature extractor
            label_encoder_path (str): Path to label encoder for decoding predictions
        """
        # Load classification model
        self.classifier = joblib.load(classifier_path)
        print(f"✓ Classifier loaded from {classifier_path}")
        
        # Load regression model
        self.regressor = joblib.load(regressor_path)
        print(f"✓ Regressor loaded from {regressor_path}")
        
        # Load feature extractors (contain fitted TF-IDF vectorizers)
        self.feature_extractor_classifier = joblib.load(feature_extractor_classifier_path)
        print(f"✓ Classifier feature extractor loaded")
        
        self.feature_extractor_regressor = joblib.load(feature_extractor_regressor_path)
        print(f"✓ Regressor feature extractor loaded")
        
        # Load label encoder for decoding predictions
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"✓ Label encoder loaded")
    
    def preprocess_input(self, title, description, input_description, output_description):
        """
        Preprocess input text into a format suitable for feature extraction.
        
        This method:
        - Creates a DataFrame from input fields
        - Combines text fields into a single column
        - Handles missing/empty values
        
        Args:
            title (str): Problem title
            description (str): Main problem description
            input_description (str): Input format description
            output_description (str): Expected output description
            
        Returns:
            pd. DataFrame: Preprocessed dataframe with combined_text column
        """
        # Create DataFrame from input fields
        df = pd.DataFrame({
            'title': [title],
            'description': [description],
            'input_description': [input_description],
            'output_description':   [output_description]
        })
        
        # Combine all text fields (same as training preprocessing)
        df['combined_text'] = (
            df['title']. fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['input_description'].fillna('') + ' ' +
            df['output_description'].fillna('')
        )
        
        return df
    
    def predict_class(self, df):
        """
        Predict the difficulty class of a problem.
        
        Uses the trained classifier to predict: 
        - Difficulty class (Easy, Medium, or Hard)
        - Probability distribution across classes (if available)
        
        Args:  
            df (pd.DataFrame): Preprocessed input dataframe
            
        Returns:  
            tuple: (predicted_class, probabilities)
                - predicted_class (str): 'Easy', 'Medium', or 'Hard'
                - probabilities (np.array): Class probabilities [P(Easy), P(Medium), P(Hard)]
        """
        # Extract features using the classifier's feature extractor
        # fit_tfidf=False because TF-IDF is already fitted during training
        X, _ = self.feature_extractor_classifier.extract_all_features(df, fit_tfidf=False)
        
        # Predict class (returns encoded number:  0, 1, or 2)
        prediction_encoded = self.classifier.predict(X)[0]
        
        # Decode back to text label (easy, medium, hard)
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Capitalize first letter for display
        prediction = prediction. capitalize()
        
        # Get probability distribution if model supports it
        probabilities = None
        if hasattr(self. classifier, 'predict_proba'):
            probabilities = self. classifier.predict_proba(X)[0]
        
        return prediction, probabilities
    
    def predict_score(self, df):
        """
        Predict the difficulty score of a problem (0-100 scale).
        
        Uses the trained regressor to predict a continuous difficulty score
        where 0 represents easiest and 100 represents hardest problems.
        
        Args:
            df (pd.DataFrame): Preprocessed input dataframe
            
        Returns:
            float:  Predicted difficulty score (0-100)
        """
        # Extract features using the regressor's feature extractor
        X, _ = self.feature_extractor_regressor.extract_all_features(df, fit_tfidf=False)
        
        # Predict score
        prediction = self.regressor.predict(X)[0]
        
        # Ensure score is within valid range [0, 100]
        prediction = np.clip(prediction, 0, 100)
        
        return prediction
    
    def predict(self, title, description, input_description, output_description):
        """
        Make complete difficulty prediction for a programming problem.
        
        This method combines classification and regression predictions to provide
        a comprehensive difficulty assessment. 
        
        Args:  
            title (str): Problem title
            description (str): Main problem description
            input_description (str): Input format description
            output_description (str): Expected output description
            
        Returns:
            dict: Prediction results containing:
                - predicted_class:  Difficulty class (Easy/Medium/Hard)
                - predicted_score: Difficulty score (0-100)
                - probabilities: Class probability distribution
                - score_interpretation: Human-readable score interpretation
        """
        # Step 1: Preprocess input text
        df = self.preprocess_input(title, description, input_description, output_description)
        
        # Step 2: Predict difficulty class
        predicted_class, probabilities = self. predict_class(df)
        
        # Step 3: Predict difficulty score
        predicted_score = self.predict_score(df)
        
        # Step 4: Interpret score
        score_interpretation = self._interpret_score(predicted_score)
        
        # Compile results
        results = {
            'predicted_class': predicted_class,
            'predicted_score': round(float(predicted_score), 2),
            'probabilities': probabilities,
            'score_interpretation':  score_interpretation
        }
        
        return results
    
    def _interpret_score(self, score):
        """
        Provide human-readable interpretation of difficulty score.
        
        Score ranges:
        - 0-20: Very Easy
        - 20-40: Easy
        - 40-60: Medium
        - 60-80: Hard
        - 80-100: Very Hard
        
        Args:
            score (float): Difficulty score (0-100)
            
        Returns:
            str: Human-readable interpretation
        """
        if score < 20:
            return "Very Easy"
        elif score < 40:
            return "Easy"
        elif score < 60:
            return "Medium"
        elif score < 80:
            return "Hard"
        else:
            return "Very Hard"


def load_predictor(models_dir='models'):
    """
    Load the predictor with all trained models.
    
    This convenience function loads all necessary model files and creates
    a ready-to-use ProblemDifficultyPredictor instance.
    
    Args:  
        models_dir (str): Directory containing trained model files
        
    Returns:  
        ProblemDifficultyPredictor:  Initialized predictor instance
        
    Raises:
        FileNotFoundError: If any required model file is missing
    """
    models_path = Path(models_dir)
    
    # Define paths to all required model files
    classifier_path = models_path / 'classifier.pkl'
    regressor_path = models_path / 'regressor.pkl'
    feature_extractor_classifier_path = models_path / 'feature_extractor_classifier.pkl'
    feature_extractor_regressor_path = models_path / 'feature_extractor_regressor.pkl'
    label_encoder_path = models_path / 'label_encoder_classifier.pkl'
    
    # Verify all files exist
    required_files = [
        classifier_path,
        regressor_path,
        feature_extractor_classifier_path,
        feature_extractor_regressor_path,
        label_encoder_path
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required model files:  {[str(f) for f in missing_files]}\n"
            f"Please train the models first by running train_classifier.py and train_regressor.py"
        )
    
    # Load and return predictor
    print("Loading prediction models...")
    predictor = ProblemDifficultyPredictor(
        classifier_path=str(classifier_path),
        regressor_path=str(regressor_path),
        feature_extractor_classifier_path=str(feature_extractor_classifier_path),
        feature_extractor_regressor_path=str(feature_extractor_regressor_path),
        label_encoder_path=str(label_encoder_path)
    )
    print("✓ All models loaded successfully!\n")
    
    return predictor


def main():
    """
    Main function to demonstrate prediction on sample problems.
    
    This serves as a test/demo showing how to use the predictor.
    """
    print("="*60)
    print("AutoJudge Prediction Demo")
    print("="*60)
    
    # Load predictor
    try:
        predictor = load_predictor()
    except FileNotFoundError as e:  
        print(f"Error: {e}")
        return
    
    # Sample problem 1: Easy problem
    print("\nSample Problem 1: Easy")
    print("-" * 60)
    results1 = predictor.predict(
        title="Sum of Two Numbers",
        description="Given two integers a and b, return their sum.",
        input_description="Two integers a and b on a single line.",
        output_description="A single integer representing a + b."
    )
    print(f"Predicted Class: {results1['predicted_class']}")
    print(f"Predicted Score: {results1['predicted_score']}/100")
    print(f"Interpretation: {results1['score_interpretation']}")
    if results1['probabilities'] is not None:
        print(f"Probabilities: Easy={results1['probabilities'][0]:.3f}, "
              f"Medium={results1['probabilities'][1]:.3f}, "
              f"Hard={results1['probabilities'][2]:.3f}")
    
    # Sample problem 2: Hard problem
    print("\n\nSample Problem 2: Hard")
    print("-" * 60)
    results2 = predictor.predict(
        title="Shortest Path in Weighted Graph with Dynamic Edges",
        description="Given a weighted directed graph with n nodes and m edges that can change weights dynamically, "
                   "process q queries.  Each query either updates an edge weight or asks for the shortest path between two nodes using Dijkstra's algorithm with dynamic programming optimization.",
        input_description="First line:  n, m, q.  Next m lines: edge definitions (u, v, w). Next q lines: queries.",
        output_description="For each shortest path query, output the minimum distance or -1 if no path exists."
    )
    print(f"Predicted Class: {results2['predicted_class']}")
    print(f"Predicted Score: {results2['predicted_score']}/100")
    print(f"Interpretation: {results2['score_interpretation']}")
    if results2['probabilities'] is not None:
        print(f"Probabilities: Easy={results2['probabilities'][0]:.3f}, "
              f"Medium={results2['probabilities'][1]:.3f}, "
              f"Hard={results2['probabilities'][2]:.3f}")


if __name__ == "__main__":
    main()