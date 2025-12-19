"""
Feature Engineering Module
Extracts and engineers features from programming problem text data. 

This module converts raw text into numerical features that machine learning
models can process, including:
- Text length features (character count, word count, etc.)
- Mathematical features (operators, formulas, numbers)
- Keyword features (algorithm and data structure terms)
- TF-IDF features (term frequency-inverse document frequency)
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    Comprehensive feature extraction class for programming problem analysis.
    
    This class extracts multiple types of features: 
    1. Text-based features: Length, word count, sentence structure
    2. Mathematical features: Operators, parentheses, numbers, formulas
    3. Keyword features: Algorithm/data structure-specific terms
    4. TF-IDF features: Statistical text representation
    """
    
    def __init__(self):
        """
        Initialize the FeatureExtractor with TF-IDF vectorizer and keyword list.
        
        TF-IDF Parameters:
        - max_features=500:  Limit vocabulary to top 500 terms (reduces dimensionality)
        - ngram_range=(1,2): Use unigrams and bigrams (captures phrases)
        - min_df=2: Ignore terms appearing in < 2 documents (removes rare words)
        - max_df=0.95: Ignore terms appearing in > 95% of docs (removes common words)
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Keywords commonly associated with different difficulty levels
        # These terms indicate algorithmic complexity and data structure usage
        self.keywords = [
            'graph', 'tree', 'dynamic', 'dp', 'recursion', 'backtrack',
            'greedy', 'sort', 'search', 'binary', 'array', 'string',
            'matrix', 'linked', 'list', 'stack', 'queue', 'hash',
            'dfs', 'bfs', 'dijkstra', 'shortest', 'path', 'optimize',
            'maximum', 'minimum', 'subsequence', 'substring', 'segment',
            'interval', 'partition', 'permutation', 'combination', 'bitwise'
        ]
    
    def extract_text_length_features(self, df):
        """
        Extract features related to text length and structure.
        
        These features capture the complexity and detail level of problem descriptions: 
        - Longer descriptions often indicate more complex problems
        - More sentences suggest detailed problem specifications
        - Average word length can indicate technical terminology usage
        
        Args:
            df (pd.DataFrame): Input dataframe with text columns
            
        Returns:
            pd.DataFrame: DataFrame with text length features
        """
        features = pd.DataFrame()
        
        def safe_len(text):
            """Count characters, handling NaN values"""
            return 0 if pd.isna(text) else len(str(text))
        
        def safe_word_count(text):
            """Count words by splitting on whitespace"""
            return 0 if pd.isna(text) else len(str(text).split())
        
        def safe_avg_word_len(text):
            """Calculate average word length"""
            if pd.isna(text):
                return 0
            words = str(text).split()
            return 0 if not words else sum(len(w) for w in words) / len(words)
        
        # Overall text metrics
        features['char_count'] = df['combined_text'].apply(safe_len)
        features['word_count'] = df['combined_text']. apply(safe_word_count)
        features['avg_word_length'] = df['combined_text']. apply(safe_avg_word_len)
        
        # Sentence count (using punctuation marks as delimiters)
        features['sentence_count'] = df['combined_text'].apply(
            lambda x: len(re.findall(r'[.!? ]+', str(x))) if pd.notna(x) else 0
        )
        
        # Individual section lengths
        features['description_length'] = df['description']. apply(safe_len)
        features['input_desc_length'] = df['input_description'].apply(safe_len)
        features['output_desc_length'] = df['output_description'].apply(safe_len)
        
        return features
    
    def extract_mathematical_features(self, df):
        """
        Extract features related to mathematical notation and complexity.
        
        Mathematical symbols often indicate:
        - Algorithmic operations (complexity)
        - Formula-based problems (requiring mathematical reasoning)
        - Computational requirements
        
        Args:
            df (pd.DataFrame): Input dataframe with text columns
            
        Returns:
            pd.DataFrame: DataFrame with mathematical features
        """
        features = pd.DataFrame()
        
        def safe_count(text, pattern):
            """Count pattern occurrences, handling NaN values"""
            return 0 if pd.isna(text) else len(re.findall(pattern, str(text)))
        
        # Count mathematical operators:  +, -, *, /, =, <, >
        features['math_operators'] = df['combined_text'].apply(
            lambda x: safe_count(x, r'[\+\-\*\/\=\<\>]')
        )
        
        # Count parentheses and brackets (often used in formulas)
        features['parentheses_count'] = df['combined_text']. apply(
            lambda x: safe_count(x, r'[\(\)\[\]\{\}]')
        )
        
        # Count numbers (constraint values, test cases, etc.)
        features['number_count'] = df['combined_text'].apply(
            lambda x: safe_count(x, r'\b\d+\b')
        )
        
        # Detect mathematical formulas (exponents, subscripts)
        features['has_formula'] = df['combined_text']. apply(
            lambda x: 1 if (pd.notna(x) and re.search(r'\^|_\d+|\\(sum|prod|int)', str(x))) else 0
        )
        
        return features
    
    def extract_keyword_features(self, df):
        """
        Extract features based on algorithm and data structure keywords.
        
        Keywords indicate:
        - Required algorithmic techniques (DFS, BFS, DP, etc.)
        - Data structures needed (tree, graph, heap, etc.)
        - Problem type and complexity
        
        Args: 
            df (pd.DataFrame): Input dataframe with text columns
            
        Returns:
            pd.DataFrame: DataFrame with keyword-based features
        """
        features = pd.DataFrame()
        
        # Count occurrences of each keyword
        for keyword in self.keywords:
            features[f'keyword_{keyword}'] = df['combined_text'].apply(
                lambda x: len(re.findall(r'\b' + keyword + r'\b', str(x).lower())) if pd.notna(x) else 0
            )
        
        # Total keyword count (overall algorithmic complexity indicator)
        features['total_keywords'] = features. sum(axis=1)
        
        return features
    
    def extract_tfidf_features(self, df, fit=True):
        """
        Extract TF-IDF (Term Frequency-Inverse Document Frequency) features.
        
        TF-IDF captures:
        - Important terms specific to each problem
        - Statistical word importance across the dataset
        - Semantic similarity between problems
        
        Args: 
            df (pd.DataFrame): Input dataframe with combined_text column
            fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of TF-IDF features
        """
        if fit:
            # Fit and transform (for training data)
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['combined_text'])
        else:
            # Transform only (for test/prediction data)
            tfidf_features = self.tfidf_vectorizer.transform(df['combined_text'])
        
        return tfidf_features
    
    def extract_all_features(self, df, fit_tfidf=True):
        """
        Extract all feature types and combine them into a single feature matrix.
        
        This method orchestrates the complete feature extraction pipeline:
        1. Text length features
        2. Mathematical features
        3. Keyword features
        4. TF-IDF features
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit_tfidf (bool): Whether to fit TF-IDF vectorizer
            
        Returns:
            tuple: (feature_matrix, feature_names)
                - feature_matrix: numpy array of all features
                - feature_names: list of feature column names
        """
        print("Extracting features...")
        
        # Extract all feature types
        text_features = self.extract_text_length_features(df)
        math_features = self.extract_mathematical_features(df)
        keyword_features = self.extract_keyword_features(df)
        tfidf_features = self.extract_tfidf_features(df, fit=fit_tfidf)
        
        # Combine all features
        # Convert TF-IDF sparse matrix to dense array
        tfidf_dense = tfidf_features.toarray()
        
        # Concatenate all feature types
        all_features = np.hstack([
            text_features.values,
            math_features.values,
            keyword_features.values,
            tfidf_dense
        ])
        
        # Create feature names list
        feature_names = (
            list(text_features.columns) +
            list(math_features. columns) +
            list(keyword_features.columns) +
            [f'tfidf_{i}' for i in range(tfidf_dense.shape[1])]
        )
        
        print(f"✓ Extracted {all_features.shape[1]} features from {all_features.shape[0]} samples")
        
        return all_features, feature_names


def main():
    """
    Main function to demonstrate feature extraction.
    
    This serves as a test/demo function showing how to use the FeatureExtractor. 
    """
    print("="*60)
    print("AutoJudge Feature Engineering Demo")
    print("="*60)
    
    # Load preprocessed data
    df = pd. read_csv('data/processed_data.csv')
    print(f"Loaded {len(df)} records")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    X, feature_names = extractor. extract_all_features(df, fit_tfidf=True)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names (first 20): {feature_names[:20]}")


if __name__ == "__main__":
    main()