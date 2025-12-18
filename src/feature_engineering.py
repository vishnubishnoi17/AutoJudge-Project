"""
Feature Engineering Module
Extracts features from text data for model training
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor: 
    def __init__(self):
        """
        Initialize the FeatureExtractor
        """
        self. tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.keywords = [
            'graph', 'tree', 'dynamic', 'dp', 'recursion', 'backtrack',
            'greedy', 'sort', 'search', 'binary', 'array', 'string',
            'matrix', 'linked', 'list', 'stack', 'queue', 'hash',
            'dfs', 'bfs', 'dijkstra', 'shortest', 'path', 'optimize',
            'maximum', 'minimum', 'subsequence', 'substring'
        ]
    
    def extract_text_length_features(self, df):
        """
        Extract text length-based features
        
        Args: 
            df (pd.DataFrame): Input dataframe
            
        Returns: 
            pd.DataFrame: Dataframe with length features
        """
        features = pd.DataFrame()
        
        # Character count
        features['char_count'] = df['combined_text']. apply(len)
        
        # Word count
        features['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
        
        # Average word length
        features['avg_word_length'] = df['combined_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Sentence count (approximate)
        features['sentence_count'] = df['combined_text'].apply(
            lambda x: len(re.findall(r'[.!? ]+', x))
        )
        
        # Description length
        features['description_length'] = df['description']. apply(len)
        
        # Input description length
        features['input_desc_length'] = df['input_description'].apply(len)
        
        # Output description length
        features['output_desc_length'] = df['output_description'].apply(len)
        
        return features
    
    def extract_mathematical_features(self, df):
        """
        Extract features related to mathematical symbols
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns: 
            pd.DataFrame: Dataframe with mathematical features
        """
        features = pd.DataFrame()
        
        # Count mathematical operators
        features['math_operators'] = df['combined_text']. apply(
            lambda x: len(re.findall(r'[\+\-\*\/\=\<\>]', x))
        )
        
        # Count parentheses
        features['parentheses_count'] = df['combined_text'].apply(
            lambda x: len(re.findall(r'[\(\)\[\]\{\}]', x))
        )
        
        # Count numbers
        features['number_count'] = df['combined_text'].apply(
            lambda x: len(re.findall(r'\b\d+\b', x))
        )
        
        # Check for formulas (contains ^ or subscript patterns)
        features['has_formula'] = df['combined_text'].apply(
            lambda x: 1 if re.search(r'\^|_\d+', x) else 0
        )
        
        return features
    
    def extract_keyword_features(self, df):
        """
        Extract keyword frequency features
        
        Args: 
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with keyword features
        """
        features = pd.DataFrame()
        
        # Count occurrences of each keyword
        for keyword in self.keywords:
            features[f'keyword_{keyword}'] = df['combined_text'].apply(
                lambda x: len(re.findall(r'\b' + keyword + r'\b', x. lower()))
            )
        
        # Total keyword count
        features['total_keywords'] = features. sum(axis=1)
        
        return features
    
    def extract_tfidf_features(self, df, fit=True):
        """
        Extract TF-IDF features
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_text'])
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(df['combined_text'])
        
        return tfidf_matrix. toarray()
    
    def extract_all_features(self, df, fit_tfidf=True):
        """
        Extract all features
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit_tfidf (bool): Whether to fit TF-IDF vectorizer
            
        Returns:
            tuple: (feature_matrix, feature_names)
        """
        print("Extracting features...")
        
        # Extract different types of features
        length_features = self.extract_text_length_features(df)
        math_features = self.extract_mathematical_features(df)
        keyword_features = self.extract_keyword_features(df)
        tfidf_features = self.extract_tfidf_features(df, fit=fit_tfidf)
        
        # Combine all features
        combined_features = pd.concat([
            length_features,
            math_features,
            keyword_features
        ], axis=1)
        
        # Add TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        combined_features = pd.concat([combined_features, tfidf_df], axis=1)
        
        print(f"Extracted {combined_features.shape[1]} features")
        
        return combined_features. values, combined_features.columns. tolist()


def main():
    """
    Main execution function for testing
    """
    # Load processed data
    df = pd.read_csv('data/processed_data.csv')
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    features, feature_names = extractor. extract_all_features(df)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("\nSample feature names:")
    print(feature_names[:10])


if __name__ == "__main__":
    main()