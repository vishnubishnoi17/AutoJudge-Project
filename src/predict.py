"""
Prediction Module
Provides utilities for making predictions on new data
"""

import pandas as pd
import joblib
from pathlib import Path


class ProblemDifficultyPredictor: 
    def __init__(self, classifier_path, regressor_path, 
                 feature_extractor_classifier_path, 
                 feature_extractor_regressor_path):
        """
        Initialize the predictor with trained models
        
        Args: 
            classifier_path (str): Path to classification model
            regressor_path (str): Path to regression model
            feature_extractor_classifier_path (str): Path to classifier's feature extractor
            feature_extractor_regressor_path (str): Path to regressor's feature extractor
        """
        self. classifier = joblib.load(classifier_path)
        self.regressor = joblib.load(regressor_path)
        self.feature_extractor_classifier = joblib.load(feature_extractor_classifier_path)
        self.feature_extractor_regressor = joblib.load(feature_extractor_regressor_path)
    
    def preprocess_input(self, title, description, input_description, output_description):
        """
        Preprocess input text
        
        Args:
            title, description, input_description, output_description: Input text fields
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Create dataframe
        df = pd.DataFrame({
            'title': [title],
            'description': [description],
            'input_description':  [input_description],
            'output_description': [output_description]
        })
        
        # Combine text fields
        df['combined_text'] = (
            df['title']. fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['input_description'].fillna('') + ' ' +
            df['output_description'].fillna('')
        )
        
        return df
    
    def predict_class(self, df):
        """
        Predict difficulty class
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            str: Predicted class
        """
        # Extract features
        X, _ = self.feature_extractor_classifier.extract_all_features(df, fit_tfidf=False)
        
        # Predict
        prediction = self.classifier.predict(X)[0]
        
        # Get probability if available
        if hasattr(self. classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(X)[0]
            return prediction, probabilities
        
        return prediction, None
    
    def predict_score(self, df):
        """
        Predict difficulty score
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            float: Predicted score
        """
        # Extract features
        X, _ = self.feature_extractor_regressor.extract_all_features(df, fit_tfidf=False)
        
        # Predict
        prediction = self.regressor.predict(X)[0]
        
        return prediction
    
    def predict(self, title, description, input_description, output_description):
        """
        Make complete prediction
        
        Args: 
            title, description, input_description, output_description: Input text fields
            
        Returns:
            dict: Prediction results
        """
        # Preprocess input
        df = self. preprocess_input(title, description, input_description, output_description)
        
        # Predict class
        predicted_class, probabilities = self. predict_class(df)
        
        # Predict score
        predicted_score = self.predict_score(df)
        
        results = {
            'predicted_class': predicted_class,
            'predicted_score': round(float(predicted_score), 2),
            'probabilities': probabilities
        }
        
        return results


def load_predictor(models_dir='models'):
    """
    Load the predictor with trained models
    
    Args: 
        models_dir (str): Directory containing trained models
        
    Returns: 
        ProblemDifficultyPredictor: Loaded predictor
    """
    models_path = Path(models_dir)
    
    # Find model files
    classifier_files = list(models_path.glob('classifier_*.pkl'))
    regressor_files = list(models_path.glob('regressor_*.pkl'))
    
    if not classifier_files or not regressor_files:
        raise FileNotFoundError("Model files not found.  Please train the models first.")
    
    classifier_path = str(classifier_files[0])
    regressor_path = str(regressor_files[0])
    feature_extractor_classifier_path = str(models_path / 'feature_extractor_classifier.pkl')
    feature_extractor_regressor_path = str(models_path / 'feature_extractor_regressor.pkl')
    
    predictor = ProblemDifficultyPredictor(
        classifier_path,
        regressor_path,
        feature_extractor_classifier_path,
        feature_extractor_regressor_path
    )
    
    return predictor


def main():
    """
    Main execution function for testing
    """
    # Load predictor
    predictor = load_predictor()
    
    # Test example
    title = "Two Sum"
    description = "Given an array of integers, return indices of the two numbers that add up to a specific target."
    input_description = "An array of integers and a target integer"
    output_description = "Two indices of the numbers"
    
    # Make prediction
    results = predictor.predict(title, description, input_description, output_description)
    
    print("Prediction Results:")
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Predicted Score: {results['predicted_score']}")
    
    if results['probabilities'] is not None:
        print(f"Class Probabilities: {results['probabilities']}")


if __name__ == "__main__":
    main()