# AutoJudge - Programming Problem Difficulty Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange)](https://github.com/vishnubishnoi17/AutoJudge-Project)

## 📖 Overview

**AutoJudge** is an intelligent machine learning system that automatically predicts the difficulty level of programming problems based on their textual description. The system provides:

- **Classification**: Easy / Medium / Hard labels
- **Regression**: Numerical difficulty scores (0-100 scale)

The predictions are based solely on problem text, making it useful for coding platforms, educators, and content creators who need to assess problem difficulty quickly.

---

## ✨ Key Features

- 🎯 **Dual Prediction Models**: Both classification and regression models for comprehensive difficulty assessment
- 🚀 **Multiple ML Algorithms**: Includes Logistic Regression, Random Forest, SVM, LightGBM, and XGBoost
- 📊 **Enhanced Scoring System**: Difficulty scores scaled from 0-100 with interpretable ranges
- 🌐 **Web Interface**: User-friendly Flask web application for easy predictions
- 📈 **Performance Visualization**: Automatic generation of model comparison and evaluation plots
- 💻 **Well-Documented Code**: Comprehensive comments and modular design

---

## 🏗️ Project Structure

```
AutoJudge-Project/
│
├── data/
│   ├── problems_data.jsonl       # Raw problem dataset
│   └── processed_data.csv        # Preprocessed and cleaned data
│
├── models/
│   ├── classifier.pkl            # Trained classification model
│   ├── regressor.pkl             # Trained regression model
│   ├── feature_extractor_classifier.pkl
│   ├── feature_extractor_regressor.pkl
│   └── *.png                     # Performance visualization plots
│
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── feature_engineering.py    # Feature extraction logic
│   ├── train_classifier.py       # Classification model training
│   ├── train_regressor.py        # Regression model training
│   └── predict.py                # Prediction utilities
│
├── app/
│   ├── app.py                    # Flask web application
│   ├── templates/
│   │   └── index.html            # Web interface template
│   └── static/
│       ├── css/                  # Stylesheets
│       └── js/                   # JavaScript files
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vishnubishnoi17/AutoJudge-Project.git
cd AutoJudge-Project
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Usage Guide

### Step 1: Preprocess the Data

Clean and prepare the dataset for training:

```bash
cd src
python data_preprocessing.py
```

**What this does:**
- Loads raw problem data from `problems_data.jsonl`
- Cleans text fields (removes special characters, handles missing values)
- Scales difficulty scores from 0-10 to 0-100 range
- Saves processed data to `data/processed_data.csv`

### Step 2: Train Classification Model

Train the difficulty class predictor (Easy/Medium/Hard):

```bash
python train_classifier.py
```

**Models trained:**
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- LightGBM Classifier
- XGBoost Classifier

**Output:**
- Best performing model saved to `models/classifier.pkl`
- Feature extractor saved to `models/feature_extractor_classifier.pkl`
- Comparison plots saved to `models/classifier_comparison.png`

### Step 3: Train Regression Model

Train the difficulty score predictor (0-100):

```bash
python train_regressor.py
```

**Models trained:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM Regressor
- XGBoost Regressor

**Output:**
- Best performing model saved to `models/regressor.pkl`
- Feature extractor saved to `models/feature_extractor_regressor.pkl`
- Comparison plots saved to `models/regressor_comparison.png`

### Step 4: Make Predictions

#### Option A: Web Application

```bash
cd app
python app.py
```

Then open your browser and navigate to: `http://localhost:5000`

**Web Interface Features:**
- Input problem title, description, input format, and output format
- Get instant predictions with confidence scores
- View difficulty class and numerical score
- Clean and intuitive user interface

#### Option B: Python Script

```python
from src.predict import load_predictor

# Initialize predictor
predictor = load_predictor('models')

# Make prediction
result = predictor.predict(
    title="Two Sum Problem",
    description="Given an array of integers and a target, find two numbers that add up to the target.",
    input_description="An array of integers and a target integer.",
    output_description="Indices of the two numbers."
)

# Display results
print(f"Predicted Class: {result['predicted_class']}")
print(f"Predicted Score: {result['predicted_score']:.2f}/100")
print(f"Interpretation: {result['score_interpretation']}")
print(f"\nClass Probabilities:")
for cls, prob in result['probabilities'].items():
    print(f"  {cls}: {prob:.2%}")
```

#### Option C: Command Line Interface

```bash
cd src
python predict.py
```

Follow the interactive prompts to enter problem details.

---

## 🎯 Difficulty Score Interpretation

The system uses a 0-100 scale with the following interpretations:

| Score Range | Interpretation | Typical Class |
|-------------|----------------|---------------|
| 0-20        | Very Easy      | Easy          |
| 20-40       | Easy           | Easy          |
| 40-60       | Medium         | Medium        |
| 60-80       | Hard           | Hard          |
| 80-100      | Very Hard      | Hard          |

---

## 🔧 Feature Engineering

The system extracts various features from problem text:

### Text-Based Features
- **Character count**: Total number of characters
- **Word count**: Total number of words
- **Sentence count**: Number of sentences in description
- **Average word length**: Indicates complexity of vocabulary

### Mathematical Features
- **Operator count**: Presence of mathematical operators (+, -, *, /, etc.)
- **Formula detection**: Identifies mathematical formulas and equations

### Keyword Features
Detects mentions of:
- **Algorithms**: sorting, searching, recursion, dynamic programming, greedy, etc.
- **Data Structures**: array, tree, graph, heap, stack, queue, hash table, etc.
- **Complexity**: time complexity, space complexity, O(n), O(log n), etc.

### Statistical Features
- **TF-IDF vectors**: Statistical representation of text importance
- Captures distinctive terms that indicate difficulty

---

## 📈 Model Performance

After training, the system generates performance visualizations:

- `classifier_comparison.png` - Accuracy comparison across different classification models
- `regressor_comparison.png` - MAE and R² comparison for regression models
- `prediction_scatter.png` - Predicted vs. actual scores scatter plot
- `confusion_matrix.png` - Classification confusion matrix

**Typical Performance Metrics:**
- Classification Accuracy: 75-85%
- Regression MAE: 5-10 points (on 0-100 scale)
- Regression R²: 0.70-0.85

---

## 🌐 Web API

### Endpoints

#### `POST /predict`
Make a difficulty prediction.

**Request Body:**
```json
{
    "title": "Problem Title",
    "description": "Problem description text...",
    "input_description": "Input format description...",
    "output_description": "Output format description..."
}
```

**Response:**
```json
{
    "success": true,
    "predicted_class": "Medium",
    "predicted_score": 55.23,
    "score_interpretation": "Medium",
    "probabilities": {
        "Easy": 0.15,
        "Medium": 0.70,
        "Hard": 0.15
    }
}
```

#### `GET /health`
Check application health status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

#### `GET /about`
Get information about the models.

**Response:**
```json
{
    "version": "2.0.0",
    "models": {
        "classifier": "LightGBM/XGBoost/RandomForest",
        "regressor": "LightGBM/XGBoost/GradientBoosting"
    },
    "score_range": "0-100"
}
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Libraries** | scikit-learn, LightGBM, XGBoost |
| **Data Processing** | pandas, numpy |
| **NLP** | NLTK, TF-IDF Vectorizer |
| **Web Framework** | Flask |
| **Visualization** | matplotlib, seaborn |
| **Serialization** | joblib |

---

## 📦 Dependencies

Main dependencies (see `requirements.txt` for full list):

```
flask>=3.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
xgboost>=2.0.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- [ ] Add more feature engineering techniques
- [ ] Implement deep learning models (BERT, transformers)
- [ ] Add more datasets from different platforms
- [ ] Improve web UI design
- [ ] Add user authentication
- [ ] Create mobile app version
- [ ] Add multi-language support

---

## 👤 Author

**Vishnu Bishnoi**
- GitHub: [@vishnubishnoi17](https://github.com/vishnubishnoi17)
- Project Link: [https://github.com/vishnubishnoi17/AutoJudge-Project](https://github.com/vishnubishnoi17/AutoJudge-Project)

---

## 🙏 Acknowledgments

- Programming problem datasets from various coding platforms
- Open-source machine learning community
- Contributors and testers
- Coding platforms: Codeforces, LeetCode, CodeChef, Kattis

---

## 📚 Dataset Information

The project uses a dataset containing:
- **title**: Problem title
- **description**: Full problem description
- **input_description**: Input format specification
- **output_description**: Expected output format
- **problem_class**: Difficulty label (Easy/Medium/Hard)
- **problem_score**: Original difficulty score (0-10, scaled to 0-100)

You can use your own dataset by following the same format and placing it in `data/problems_data.jsonl`.

---

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Implement BERT/transformers for better text understanding
- [ ] **Real-time Learning**: Update models based on user feedback
- [ ] **Multi-label Classification**: Predict problem topics and techniques
- [ ] **Personalized Difficulty**: Adjust predictions based on user skill level
- [ ] **Integration APIs**: Connect with Codeforces, LeetCode, etc.
- [ ] **Batch Processing**: Handle multiple problems simultaneously
- [ ] **Confidence Intervals**: Provide uncertainty estimates for predictions
- [ ] **Explainable AI**: Add SHAP values to explain predictions

---

## ❓ Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Issue: Models not found**
```bash
# Solution: Train the models first
cd src
python train_classifier.py
python train_regressor.py
```

**Issue: Web app not starting**
```bash
# Solution: Check if port 5000 is available
# Or specify a different port
cd app
flask run --port 8000
```

**Issue: Low prediction accuracy**
```bash
# Solution: Retrain with more data or adjust hyperparameters
# Edit train_classifier.py or train_regressor.py
```

---

## 📧 Contact & Support

For questions, issues, or suggestions:

1. **Open an Issue**: [GitHub Issues](https://github.com/vishnubishnoi17/AutoJudge-Project/issues)
2. **Discussions**: Use GitHub Discussions for general questions
3. **Email**: Contact through GitHub profile

---

## ⭐ Show Your Support

If you find this project helpful, please consider:
- ⭐ **Starring the repository**
- 🍴 **Forking for your own use**
- 📢 **Sharing with others**
- 🐛 **Reporting bugs**
- 💡 **Suggesting features**

---

**Made with ❤️ by Vishnu Bishnoi**
