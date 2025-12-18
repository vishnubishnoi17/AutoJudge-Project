"""
Regression Model Training Module
Trains models to predict problem difficulty score
"""

import pandas as pd
import numpy as np
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from feature_engineering import FeatureExtractor


class DifficultyRegressor:
    def __init__(self):
        """
        Initialize the DifficultyRegressor
        """
        self.feature_extractor = FeatureExtractor()
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
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
        y = df['problem_score'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size:  {len(X_test)}")
        print(f"Score range: {y. min():.2f} - {y.max():.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train, X_test, y_train, y_test: Train and test data
        """
        results = {}
        best_mae = float('inf')
        
        for name, model in self. models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\nMean Absolute Error (MAE): {mae:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Save best model (based on MAE)
            if mae < best_mae:
                best_mae = mae
                self. best_model = model
                self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best MAE: {best_mae:.4f}")
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
        model_path = f"{output_dir}/regressor_{self.best_model_name}.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save feature extractor
        extractor_path = f"{output_dir}/feature_extractor_regressor.pkl"
        joblib.dump(self.feature_extractor, extractor_path)
        print(f"Feature extractor saved to {extractor_path}")
    
    def plot_results(self, results):
        """
        Plot model comparison results
        
        Args:
            results (dict): Dictionary of model results
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        models = list(results.keys())
        mae_values = [results[m]['mae'] for m in models]
        rmse_values = [results[m]['rmse'] for m in models]
        
        # MAE comparison
        axes[0]. bar(models, mae_values, color=['blue', 'green', 'orange'])
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Mean Absolute Error Comparison')
        
        for i, v in enumerate(mae_values):
            axes[0]. text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # RMSE comparison
        axes[1].bar(models, rmse_values, color=['blue', 'green', 'orange'])
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Squared Error Comparison')
        
        for i, v in enumerate(rmse_values):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('models/regressor_comparison.png')
        print("\nComparison plot saved to models/regressor_comparison.png")


def main():
    """
    Main execution function
    """
    # Initialize regressor
    regressor = DifficultyRegressor()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = regressor.load_and_prepare_data()
    
    # Train and evaluate models
    results = regressor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save best model
    regressor.save_model()
    
    # Plot results
    regressor.plot_results(results)
    
    print("\nRegression model training completed!")


if __name__ == "__main__":
    main()