"""
AutoJudge:  Programming Problem Difficulty Prediction System

This package provides tools for predicting the difficulty of programming problems
using machine learning models including LightGBM and XGBoost. 

Modules:
- data_preprocessing: Data loading and preprocessing
- feature_engineering: Feature extraction from problem text
- train_classifier: Training classification models
- train_regressor:  Training regression models
- predict: Making predictions on new problems

Version 2.0 Updates:
- Difficulty scale changed from 0-10 to 0-100
- Added LightGBM and XGBoost models
- Enhanced code documentation
- Improved feature engineering
"""

__version__ = "2.0.0"
__author__ = "AutoJudge Team"
__description__ = "Programming Problem Difficulty Predictor with ML"

# Import main classes for easy access
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureExtractor
from .train_classifier import DifficultyClassifier
from . train_regressor import DifficultyRegressor
from .predict import ProblemDifficultyPredictor, load_predictor

__all__ = [
    'DataPreprocessor',
    'FeatureExtractor',
    'DifficultyClassifier',
    'DifficultyRegressor',
    'ProblemDifficultyPredictor',
    'load_predictor'
]