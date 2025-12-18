# AutoJudge:  Predicting Programming Problem Difficulty

An intelligent system that automatically predicts programming problem difficulty based on textual descriptions.

## Features

- **Classification Model**: Predicts problem difficulty class (Easy/Medium/Hard)
- **Regression Model**: Predicts numerical difficulty score
- **Web Interface**:  Simple UI for real-time predictions

## Project Structure

```
AutoJudge/
├── data/                          # Dataset directory
├── models/                        # Saved trained models
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature extraction
│   ├── train_classifier.py       # Classification model training
│   ├── train_regressor.py        # Regression model training
│   └── predict.py                # Prediction utilities
├── app/                          # Web application
│   ├── app.py                    # Flask application
│   ├── templates/                # HTML templates
│   └── static/                   # CSS and static files
└── requirements.txt              # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AREEG94FAHAD/TaskComplexityEval-24.git
cd TaskComplexityEval-24
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python src/data_preprocessing.py
```

### 2. Train Models

Train the classification model:
```bash
python src/train_classifier.py
```

Train the regression model:
```bash
python src/train_regressor.py
```

### 3. Run Web Application

```bash
python app/app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Model Performance

### Classification Model
- **Accuracy**: ~XX%
- **Models Tested**:  Logistic Regression, Random Forest, SVM

### Regression Model
- **MAE**: ~XX
- **RMSE**: ~XX
- **Models Tested**: Linear Regression, Random Forest, Gradient Boosting

## Dataset

The dataset (`problems_data.jsonl`) contains:
- `title`: Problem title
- `description`: Problem description
- `input_description`: Input format description
- `output_description`: Output format description
- `problem_class`: Difficulty class (Easy/Medium/Hard)
- `problem_score`: Numerical difficulty score

## Features Used

1. **Text Length Features**
   - Total character count
   - Word count
   - Average word length

2. **Mathematical Symbols**
   - Count of mathematical operators
   - Presence of equations

3. **Keyword Frequency**
   - Keywords: graph, dynamic programming, recursion, etc. 

4. **TF-IDF Vectors**
   - Text vectorization for semantic understanding

## Technology Stack

- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, pandas, numpy
- **NLP**: TfidfVectorizer
- **Frontend**: HTML, CSS, JavaScript

## Contributors

- AREEG94FAHAD

## License

This project is for educational purposes. 
```