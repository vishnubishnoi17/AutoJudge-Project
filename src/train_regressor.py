"""
Regression Model Training Module
Trains multiple machine learning models to predict problem difficulty scores (0-100).

This module supports:
- Traditional ML models: Linear Regression, Random Forest
- Gradient Boosting models:  Gradient Boosting, LightGBM, XGBoost
- Score range: 0-100 (scaled from original 0-10)

The best performing model is automatically selected based on MAE (Mean Absolute Error).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import FeatureExtractor


class DifficultyRegressor:
    """
    Comprehensive regressor for predicting programming problem difficulty scores.
    
    This class:
    - Trains multiple regression models
    - Evaluates and compares model performance
    - Selects the best model automatically based on MAE
    - Provides visualization of results
    - Saves the best model for deployment
    """
    
    def __init__(self):
        """
        Initialize the DifficultyRegressor with multiple model architectures.
        
        Models included:
        1. Linear Regression: Fast linear baseline
        2. Random Forest: Ensemble of decision trees for regression
        3. Gradient Boosting: Sequential boosting regressor
        4. LightGBM: Fast gradient boosting framework optimized for large datasets
        5. XGBoost: Powerful gradient boosting implementation
        """
        self.feature_extractor = FeatureExtractor()
        
        # Dictionary of all regression models to train
        self.models = {
            'linear_regression':  LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=20,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42,
                learning_rate=0.1,
                max_depth=10
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=31,
                verbose=-1  # Suppress warnings
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=10,
                objective='reg:squarederror'  # Regression objective
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
        - Splits data into training and testing sets
        - Displays score distribution statistics
        
        Args:
            data_path (str): Path to preprocessed CSV file
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading data for regression...")
        df = pd.read_csv(data_path)
        
        # Extract features from text data
        X, feature_names = self.feature_extractor.extract_all_features(df, fit_tfidf=True)
        y = df['problem_score'].values
        
        # Split data (no stratification needed for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"✓ Training set size: {len(X_train)}")
        print(f"✓ Test set size: {len(X_test)}")
        print(f"✓ Number of features: {X_train. shape[1]}")
        
        # Display score statistics
        print(f"\nScore range (0-100 scale):")
        print(f"  Minimum: {y.min():.2f}")
        print(f"  Maximum: {y.max():.2f}")
        print(f"  Mean: {y. mean():.2f}")
        print(f"  Median:  {np.median(y):.2f}")
        print(f"  Std Dev: {y.std():.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all regression models. 
        
        For each model: 
        1. Train on training data
        2. Make predictions on test data
        3. Calculate MAE, RMSE, and R² metrics
        4. Display performance metrics
        5. Track best model based on MAE
        
        Metrics explained:
        - MAE (Mean Absolute Error): Average prediction error (lower is better)
        - RMSE (Root Mean Squared Error): Penalizes large errors (lower is better)
        - R² (R-squared): Proportion of variance explained (higher is better, max 1.0)
        
        Args:
            X_train, X_test: Feature matrices for training and testing
            y_train, y_test: Target scores for training and testing
            
        Returns:
            dict: Dictionary of model names and their performance metrics
        """
        results = {}
        best_mae = float('inf')  # Initialize with infinity
        
        for name, model in self. models.items():
            print(f"\n{'='*60}")
            print(f"Training {name.upper().replace('_', ' ')}...")
            print(f"{'='*60}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Ensure predictions are within valid range [0, 100]
            y_pred = np.clip(y_pred, 0, 100)
            
            # Calculate regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            # Display metrics
            print(f"\n✓ Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  (On average, predictions are off by {mae:.2f} points on 0-100 scale)")
            print(f"\n✓ Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"  (Standard deviation of prediction errors)")
            print(f"\n✓ R² Score: {r2:.4f}")
            print(f"  ({r2*100:.2f}% of variance explained)")
            
            # Track the best model (lowest MAE)
            if mae < best_mae:
                best_mae = mae
                self. best_model = model
                self.best_model_name = name
        
        # Display final results
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name.upper().replace('_', ' ')}")
        print(f"BEST MAE: {best_mae:.4f}")
        print(f"{'='*60}")
        
        return results
    
    def plot_model_comparison(self, results, output_dir='models'):
        """
        Create and save visualizations comparing model performance.
        
        Creates two plots:
        1. Bar plot of MAE comparison (lower is better)
        2. Bar plot of R² comparison (higher is better)
        
        Args:
            results (dict): Dictionary of model names and their metrics
            output_dir (str): Directory to save plots
        """
        models = list(results.keys())
        mae_values = [results[m]['mae'] for m in models]
        r2_values = [results[m]['r2'] for m in models]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: MAE comparison (lower is better)
        ax1 = axes[0]
        bars1 = ax1.bar(models, mae_values, color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        # Highlight best model (lowest MAE)
        best_mae_idx = mae_values.index(min(mae_values))
        bars1[best_mae_idx].set_color('gold')
        bars1[best_mae_idx].set_edgecolor('darkgoldenrod')
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        ax1.set_title('Model Performance - MAE (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, mae in enumerate(mae_values):
            ax1.text(i, mae + 0.1, f'{mae:.3f}', ha='center', fontsize=10)
        
        # Plot 2: R² comparison (higher is better)
        ax2 = axes[1]
        bars2 = ax2.bar(models, r2_values, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        
        # Highlight best model (highest R²)
        best_r2_idx = r2_values.index(max(r2_values))
        bars2[best_r2_idx].set_color('gold')
        bars2[best_r2_idx].set_edgecolor('darkgoldenrod')
        
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('Model Performance - R² (Higher is Better)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, r2 in enumerate(r2_values):
            ax2.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / 'regressor_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {plot_path}")
        
        plt.close()
    
    def plot_predictions(self, X_test, y_test, output_dir='models'):
        """
        Create scatter plot of predicted vs actual scores.
        
        This visualization helps understand:
        - How well predictions match actual values
        - Whether model tends to over/under-predict
        - Distribution of prediction errors
        
        Args: 
            X_test:  Test feature matrix
            y_test:  Actual test scores
            output_dir (str): Directory to save plot
        """
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 100)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line (diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Score (0-100)', fontsize=12)
        plt.ylabel('Predicted Score (0-100)', fontsize=12)
        plt.title(f'Predicted vs Actual Scores - {self.best_model_name. upper().replace("_", " ")}', 
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add metrics text
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nR²: {r2:.3f}', 
                 transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / 'prediction_scatter.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction scatter plot saved to {plot_path}")
        
        plt.close()
    
    def save_model(self, output_dir='models'):
        """
        Save the best regression model and feature extractor to disk.
        
        Saves:
        - Best regressor model (joblib format)
        - Feature extractor with fitted TF-IDF vectorizer
        - Model metadata (name, type, scale info)
        
        Args: 
            output_dir (str): Directory to save model files
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the best model
        model_path = output_path / 'regressor.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"✓ Best regressor saved to {model_path}")
        
        # Save feature extractor
        extractor_path = output_path / 'feature_extractor_regressor.pkl'
        joblib.dump(self.feature_extractor, extractor_path)
        print(f"✓ Feature extractor saved to {extractor_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': 'regressor',
            'scale':  '0-100',
            'score_range': {'min': 0, 'max': 100}
        }
        metadata_path = output_path / 'regressor_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        print(f"✓ Model metadata saved to {metadata_path}")


def main():
    """
    Main function to run the complete regression training pipeline.
    
    Pipeline steps:
    1. Initialize regressor
    2. Load and prepare data
    3. Train all models
    4. Evaluate and compare models
    5. Visualize results
    6. Save best model
    """
    print("="*60)
    print("AutoJudge Regression Model Training")
    print("="*60)
    
    # Initialize regressor
    regressor = DifficultyRegressor()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = regressor.load_and_prepare_data()
    
    # Train and evaluate all models
    results = regressor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot model comparisons
    regressor.plot_model_comparison(results)
    
    # Plot predictions vs actual
    regressor.plot_predictions(X_test, y_test)
    
    # Save the best model
    regressor.save_model()
    
    print("\n✓ Regression training complete!")


if __name__ == "__main__":
    main()