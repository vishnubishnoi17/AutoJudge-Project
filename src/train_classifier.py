"""
Classification Model Training Module
Trains models to predict problem difficulty class (Easy/Medium/Hard)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn. metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import FeatureExtractor


class DifficultyClassifier:
    def __init__(self):
        """
        Initialize the DifficultyClassifier
        """
        self.feature_extractor = FeatureExtractor()
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
    
    def load_and_prepare_data(self, data_path='data/processed_data.csv'):
        """
        Load and prepare data for training
        
        Args:
            data_path (str): Path to processed data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Extract features
        X, feature_names = self.feature_extractor.extract_all_features(df, fit_tfidf=True)
        y = df['problem_class']. values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        
        Args: 
            X_train, X_test, y_train, y_test:  Train and test data
        """
        results = {}
        best_accuracy = 0
        
        for name, model in self.models. items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model. predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self. best_model = model
                self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Accuracy: {best_accuracy:. 4f}")
        print(f"{'='*50}")
        
        return results
    
    def save_model(self, output_dir='models'):
        """
        Save the best model and feature extractor
        
        Args: 
            output_dir (str): Output directory path
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = f"{output_dir}/classifier_{self.best_model_name}. pkl"
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save feature extractor
        extractor_path = f"{output_dir}/feature_extractor_classifier. pkl"
        joblib.dump(self.feature_extractor, extractor_path)
        print(f"Feature extractor saved to {extractor_path}")
    
    def plot_results(self, results):
        """
        Plot model comparison results
        
        Args: 
            results (dict): Dictionary of model results
        """
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        accuracies = list(results.values())
        
        plt.bar(models, accuracies, color=['blue', 'green', 'red'])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Classification Model Comparison')
        plt.ylim([0, 1])
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('models/classifier_comparison.png')
        print("\nComparison plot saved to models/classifier_comparison.png")


def main():
    """
    Main execution function
    """
    # Initialize classifier
    classifier = DifficultyClassifier()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = classifier.load_and_prepare_data()
    
    # Train and evaluate models
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save best model
    classifier.save_model()
    
    # Plot results
    classifier.plot_results(results)
    
    print("\nClassification model training completed!")


if __name__ == "__main__": 
    main()