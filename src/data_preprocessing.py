"""
Data Preprocessing Module
Handles data loading, cleaning, and initial preparation
"""

import pandas as pd
import jsonlines
import re
from pathlib import Path


class DataPreprocessor:
    def __init__(self, data_path='data/problems_data.jsonl'):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_path (str): Path to the JSONL dataset file
        """
        self. data_path = data_path
        self.df = None
    
    def load_data(self):
        """
        Load data from JSONL file
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("Loading data from JSONL file...")
        data_list = []
        
        try:
            with jsonlines.open(self.data_path) as reader:
                for obj in reader:
                    data_list.append(obj)
            
            self.df = pd. DataFrame(data_list)
            print(f"Loaded {len(self.df)} records")
            return self.df
        
        except FileNotFoundError: 
            print(f"Error: File not found at {self.data_path}")
            return None
    
    def clean_text(self, text):
        """
        Clean text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re. sub(r'\s+', ' ', text)
        
        # Remove special characters but keep mathematical symbols
        # text = re.sub(r'[^\w\s\+\-\*\/\=\<\>\(\)\[\]\{\}]', '', text)
        
        return text.strip()
    
    def combine_text_fields(self):
        """
        Combine all text fields into a single column
        """
        print("Combining text fields...")
        
        # Define text columns
        text_columns = ['title', 'description', 'input_description', 'output_description']
        
        # Clean each column
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self.clean_text)
        
        # Combine all text fields
        self.df['combined_text'] = (
            self.df['title']. fillna('') + ' ' +
            self.df['description']. fillna('') + ' ' +
            self.df['input_description'].fillna('') + ' ' +
            self.df['output_description'].fillna('')
        )
        
        print("Text fields combined successfully")
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        # Fill missing text with empty strings
        text_columns = ['title', 'description', 'input_description', 
                       'output_description', 'combined_text']
        
        for col in text_columns: 
            if col in self.df.columns:
                self.df[col]. fillna('', inplace=True)
        
        # Drop rows with missing target values
        if 'problem_class' in self.df. columns:
            self.df = self.df.dropna(subset=['problem_class'])
        
        if 'problem_score' in self.df.columns:
            self.df = self.df.dropna(subset=['problem_score'])
        
        print(f"Cleaned dataset size: {len(self.df)} records")
    
    def preprocess(self):
        """
        Run the complete preprocessing pipeline
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        self.load_data()
        
        if self.df is None:
            return None
        
        self.combine_text_fields()
        self.handle_missing_values()
        
        # Display basic statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {len(self.df)}")
        
        if 'problem_class' in self. df.columns:
            print("\nClass distribution:")
            print(self. df['problem_class'].value_counts())
        
        if 'problem_score' in self. df.columns:
            print(f"\nScore range: {self.df['problem_score'].min()} - {self.df['problem_score'].max()}")
            print(f"Mean score: {self.df['problem_score'].mean():.2f}")
        
        return self.df
    
    def save_processed_data(self, output_path='data/processed_data.csv'):
        """
        Save processed data to CSV
        
        Args:
            output_path (str): Output file path
        """
        if self.df is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"\nProcessed data saved to {output_path}")


def main():
    """
    Main execution function
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/problems_data.jsonl')
    
    # Run preprocessing
    df = preprocessor.preprocess()
    
    # Save processed data
    if df is not None:
        preprocessor.save_processed_data()
        print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()