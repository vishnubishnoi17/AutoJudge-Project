"""
Data Preprocessing Module
Handles data loading, cleaning, and initial preparation for the AutoJudge system. 
This module processes programming problem data and prepares it for machine learning models. 
"""

import pandas as pd
import jsonlines
import re
from pathlib import Path


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for handling programming problem datasets.
    
    This class handles:
    - Loading data from JSONL files
    - Cleaning and normalizing text data
    - Handling missing values
    - Scaling difficulty scores from 0-10 to 0-100
    - Combining multiple text fields
    """
    
    def __init__(self, data_path='data/problems_data.jsonl'):
        """
        Initialize the DataPreprocessor with the path to the dataset.
        
        Args:
            data_path (str): Path to the JSONL dataset file containing problem data
        """
        self. data_path = data_path
        self.df = None  # Will hold the loaded DataFrame
    
    def load_data(self):
        """
        Load data from JSONL file into a pandas DataFrame.
        
        JSONL (JSON Lines) format stores each JSON object on a separate line,
        which is efficient for streaming and processing large datasets.
        
        Returns:
            pd.DataFrame: Loaded dataframe with all problem records
            
        Raises:
            FileNotFoundError: If the specified data file doesn't exist
        """
        print("Loading data from JSONL file...")
        data_list = []
        
        try: 
            # Read JSONL file line by line
            with jsonlines. open(self.data_path) as reader:
                for obj in reader:
                    data_list.append(obj)
            
            # Convert list of dictionaries to DataFrame
            self.df = pd.DataFrame(data_list)
            print(f"✓ Successfully loaded {len(self.df)} records")
            return self.df
        
        except FileNotFoundError:  
            print(f"✗ Error: File not found at {self.data_path}")
            return None
    
    def clean_text(self, text):
        """
        Clean and normalize text data by removing extra whitespace and special characters. 
        
        This function: 
        - Handles NaN values gracefully
        - Removes excessive whitespace
        - Preserves mathematical symbols important for problem descriptions
        
        Args:
            text (str): Input text to be cleaned
            
        Returns: 
            str: Cleaned and normalized text
        """
        # Handle missing values
        if pd.isna(text):
            return ""
        
        # Convert to string to handle mixed types
        text = str(text)
        
        # Replace multiple whitespace characters with a single space
        text = re. sub(r'\s+', ' ', text)
        
        # Note: We preserve mathematical symbols like +, -, *, /, =, <, >, (, ), [, ], {, }
        # as they are important for understanding problem complexity
        
        return text.strip()
    
    def combine_text_fields(self):
        """
        Combine multiple text fields into a single 'combined_text' column. 
        
        This creates a unified text representation by concatenating:
        - title:  Problem title
        - description: Main problem description
        - input_description: Input format description
        - output_description:  Expected output description
        
        The combined text is used for feature extraction and analysis.
        """
        print("Combining text fields...")
        
        # Define all text columns in the dataset
        text_columns = ['title', 'description', 'input_description', 'output_description']
        
        # Clean each text column individually
        for col in text_columns:
            if col in self. df.columns:
                self. df[col] = self.df[col].apply(self.clean_text)
        
        # Combine all text fields with spaces
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['input_description'].fillna('') + ' ' +
            self.df['output_description'].fillna('')
        )
        
        print("✓ Text fields combined successfully")
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset using appropriate strategies.
        
        Strategy:
        - Text columns:  Fill with empty strings (meaningful default)
        - Target columns (problem_class, problem_score): Drop rows (cannot train without labels)
        
        This ensures data quality while preserving maximum information.
        """
        print("Handling missing values...")
        
        # Identify missing values per column
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values per column:")
            print(missing_counts[missing_counts > 0])
        
        # Fill missing text values with empty strings
        text_columns = ['title', 'description', 'input_description', 
                       'output_description', 'combined_text']
        
        for col in text_columns:  
            if col in self. df.columns:
                self. df[col].fillna('', inplace=True)
        
        # Drop rows with missing target values (cannot use for training)
        initial_size = len(self.df)
        
        if 'problem_class' in self. df. columns:
            self.df = self.df.dropna(subset=['problem_class'])
        
        if 'problem_score' in self.df.columns:
            self.df = self.df.dropna(subset=['problem_score'])
        
        dropped_rows = initial_size - len(self.df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing target values")
        
        print(f"✓ Cleaned dataset size: {len(self.df)} records")
    
    def scale_scores(self):
        """
        Scale problem difficulty scores from 0-10 range to 0-100 range. 
        
        This transformation: 
        - Makes scores more intuitive (percentage-like)
        - Provides finer granularity for difficulty assessment
        - Maintains relative relationships between scores
        
        Formula: new_score = old_score * 10
        """
        if 'problem_score' in self. df.columns:
            print("Scaling scores from 0-10 to 0-100...")
            
            # Store original score statistics
            original_min = self.df['problem_score'].min()
            original_max = self.df['problem_score'].max()
            original_mean = self.df['problem_score'].mean()
            
            # Scale scores by multiplying by 10
            self. df['problem_score'] = self.df['problem_score'] * 10
            
            # Ensure scores are within valid range [0, 100]
            self.df['problem_score'] = self.df['problem_score'].clip(0, 100)
            
            # Display transformation results
            new_min = self.df['problem_score'].min()
            new_max = self.df['problem_score'].max()
            new_mean = self.df['problem_score'].mean()
            
            print(f"✓ Score scaling complete:")
            print(f"  Original range: [{original_min:.2f}, {original_max:.2f}] (mean: {original_mean:.2f})")
            print(f"  New range: [{new_min:.2f}, {new_max:.2f}] (mean: {new_mean:.2f})")
    
    def preprocess(self):
        """
        Execute the complete preprocessing pipeline.
        
        Pipeline stages:
        1. Load data from JSONL file
        2. Check if data loaded successfully
        3. Combine text fields into unified representation
        4. Handle missing values appropriately
        5. Scale difficulty scores to 0-100 range
        6. Save preprocessed data to CSV
        
        Returns:
            pd.DataFrame: Fully preprocessed dataframe ready for feature extraction
        """
        # Stage 1: Load raw data
        self.load_data()
        
        if self.df is None:
            print("✗ Preprocessing failed: Could not load data")
            return None
        
        # Stage 2:  Combine text fields
        self.combine_text_fields()
        
        # Stage 3: Handle missing values
        self.handle_missing_values()
        
        # Stage 4: Scale scores from 0-10 to 0-100
        self.scale_scores()
        
        # Stage 5: Save preprocessed data
        output_path = Path('data/processed_data.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✓ Preprocessed data saved to {output_path}")
        
        return self. df


def main():
    """
    Main function to run the preprocessing pipeline.
    
    This function serves as an entry point for standalone execution
    of the preprocessing module.
    """
    print("="*60)
    print("AutoJudge Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    df = preprocessor.preprocess()
    
    if df is not None:
        print("\n" + "="*60)
        print("Preprocessing Summary")
        print("="*60)
        print(f"Total records: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"\nColumns:  {list(df.columns)}")
        
        # Display sample statistics
        if 'problem_score' in df. columns:
            print(f"\nScore statistics:")
            print(df['problem_score'].describe())
        
        if 'problem_class' in df.columns:
            print(f"\nClass distribution:")
            print(df['problem_class'].value_counts())


if __name__ == "__main__":
    main()