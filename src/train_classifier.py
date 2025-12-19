"""
Classification Model Training Module
Trains multiple machine learning models to predict problem difficulty class. 

This module supports:
- Traditional ML models:   Logistic Regression, Random Forest, SVM
- Gradient Boosting models: LightGBM, XGBoost
- Difficulty classes: Easy, Medium, Hard

The best performing model is automatically selected based on accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn. metrics import accuracy_score, classification_report, confusion_matrix
from sklearn. preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import FeatureExtractor


class DifficultyClassifier:
    """
    Comprehensive classifier for predicting programming problem difficulty levels.
    
    This class: 
    - Trains multiple classification models
    - Evaluates and compares model performance
    - Selects the best model automatically
    - Provides visualization of results
    - Saves the best model for deployment
    """
    
    def __init__(self):
        """
        Initialize the DifficultyClassifier with multiple model architectures.
        
        Models included:
        1. Logistic Regression: Fast linear baseline
        2. Random Forest: Ensemble of decision trees
        3. SVM:  Support Vector Machine with RBF kernel
        4. LightGBM: Fast gradient boosting framework
        5. XGBoost: Powerful gradient boosting implementation
        """
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()  # For encoding text labels to numbers
        
        # Dictionary of all classification models to train
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=20,
                min_samples_split=5
            ),
            'svm':  SVC(
                kernel='rbf', 
                random_state=42,
                probability=True  # Enable probability predictions
            ),
            'lightgbm':  LGBMClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=31,
                verbose=-1  # Suppress warnings
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=10,
                eval_metric='mlogloss'  # Multi-class log loss
            )
        }
        
        self.best_model = None
        self.best_model_name = None
    
    def load_and_prepare_data(self, data_path='data/processed_data.csv'):
        """
        Load preprocessed data and prepare it for training.
        
        This method:
        - Loads the CSV dataset
        - Extracts features using FeatureExtractor
        - Encodes text labels to numbers for XGBoost compatibility
        - Splits data into training and testing sets
        - Ensures stratified split to maintain class distribution
        
        Args:
            data_path (str): Path to preprocessed CSV file
            
        Returns:  
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading data for classification...")
        df = pd.read_csv(data_path)
        
        # Extract features from text data
        X, feature_names = self.feature_extractor.extract_all_features(df, fit_tfidf=True)
        
        # Encode labels to numbers for XGBoost compatibility
        # 'easy' -> 0, 'hard' -> 1, 'medium' -> 2 (or similar)
        y_text = df['problem_class'].values
        y = self.label_encoder. fit_transform(y_text)
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y  # Ensures proportional class distribution in train/test
        )
        
        print(f"✓ Training set size: {len(X_train)}")
        print(f"✓ Test set size: {len(X_test)}")
        print(f"✓ Number of features: {X_train.shape[1]}")
        
        # Display class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training set:")
        for cls_encoded, count in zip(unique, counts):
            cls_name = self.label_encoder.inverse_transform([cls_encoded])[0]
            print(f"  {cls_name}:  {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all classification models.
        
        For each model:  
        1. Train on training data
        2. Make predictions on test data
        3. Calculate accuracy and other metrics
        4. Display classification report
        5. Show confusion matrix
        
        Args:  
            X_train, X_test:   Feature matrices for training and testing
            y_train, y_test:  Target labels for training and testing
            
        Returns:
            dict:  Dictionary of model names and their accuracies
        """
        results = {}
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name. upper().replace('_', ' ')}...")
            print(f"{'='*60}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model. predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Detailed classification metrics
            # Convert encoded labels back to text for readable report
            y_test_text = self.label_encoder.inverse_transform(y_test)
            y_pred_text = self.label_encoder.inverse_transform(y_pred)
            
            print("\nClassification Report:")
            print(classification_report(y_test_text, y_pred_text))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Track the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        # Display final results
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name. upper().replace('_', ' ')}")
        print(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        return results
    
    def plot_model_comparison(self, results, output_dir='models'):
        """
        Create and save a bar plot comparing model accuracies.
        
        Args:
            results (dict): Dictionary of model names and accuracies
            output_dir (str): Directory to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        accuracies = list(results.values())
        
        # Create bar plot
        bars = plt.bar(models, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Highlight the best model
        best_idx = accuracies.index(max(accuracies))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Classification Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (model, acc) in enumerate(zip(models, accuracies)):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / 'classifier_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {plot_path}")
        
        plt.close()
    
    def save_model(self, output_dir='models'):
        """
        Save the best classification model and feature extractor to disk.
        
        Saves:
        - Best classifier model (joblib format)
        - Feature extractor with fitted TF-IDF vectorizer
        - Label encoder for converting predictions back to text
        
        Args:
            output_dir (str): Directory to save model files
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the best model
        model_path = output_path / 'classifier. pkl'
        joblib.dump(self.best_model, model_path)
        print(f"✓ Best classifier saved to {model_path}")
        
        # Save feature extractor
        extractor_path = output_path / 'feature_extractor_classifier.pkl'
        joblib. dump(self.feature_extractor, extractor_path)
        print(f"✓ Feature extractor saved to {extractor_path}")
        
        # Save label encoder
        encoder_path = output_path / 'label_encoder_classifier.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder saved to {encoder_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': 'classifier',
            'scale':  '0-100',
            'classes': list(self.label_encoder.classes_)
        }
        metadata_path = output_path / 'classifier_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        print(f"✓ Model metadata saved to {metadata_path}")


def main():
    """
    Main function to run the complete classification training pipeline.
    
    Pipeline steps:
    1. Initialize classifier
    2. Load and prepare data
    3. Train all models
    4. Evaluate and compare models
    5. Visualize results
    6. Save best model
    """
    print("="*60)
    print("AutoJudge Classification Model Training")
    print("="*60)
    
    # Initialize classifier
    classifier = DifficultyClassifier()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = classifier.load_and_prepare_data()
    
    # Train and evaluate all models
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot model comparison
    classifier.plot_model_comparison(results)
    
    # Save the best model
    classifier.save_model()
    
    print("\n✓ Classification training complete!")


if __name__ == "__main__":
    main()