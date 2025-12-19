# 🎯 AutoJudge:  Programming Problem Difficulty Predictor

An intelligent machine learning system that automatically predicts programming problem difficulty based on textual descriptions. Uses both classification (Easy/Medium/Hard) and regression (numeric score) approaches with an interactive web interface.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green. svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- **🎲 Classification Model**: Predicts problem difficulty class (Easy/Medium/Hard) with probability distribution
- **📊 Regression Model**: Predicts numerical difficulty score (0-10 scale)
- **🌐 Web Interface**: Beautiful, responsive UI for real-time predictions
- **🌓 Dark Mode**: Toggle between light and dark themes with persistent storage
- **🤖 Machine Learning**: Multiple models (Random Forest, Logistic Regression, SVM, Gradient Boosting)
- **📈 Feature Engineering**: 540+ features including TF-IDF, keyword analysis, and text statistics

## 🚀 Live Demo

> **📸 Add your screenshots here after running the app!**

### Light Mode Interface
![AutoJudge Light Mode](https://via.placeholder.com/800x400/667eea/ffffff?text=Add+Light+Mode+Screenshot)
*Replace with:  Screenshot of the main interface in light mode*

### Dark Mode Interface
![AutoJudge Dark Mode](https://via.placeholder.com/800x400/1a202c/ffffff?text=Add+Dark+Mode+Screenshot)
*Replace with: Screenshot of the main interface in dark mode*

### Prediction Results
![Prediction Results](https://via.placeholder.com/800x300/48bb78/ffffff?text=Add+Prediction+Results+Screenshot)
*Replace with: Screenshot showing a successful prediction with probability bars*

## 📁 Project Structure

```
AutoJudge-Project/
├── data/
│   ├── problems_data.jsonl           # Raw dataset (JSONL format)
│   └── processed_data.csv            # Preprocessed dataset
│
├── models/
│   ├── classifier_random_forest.pkl  # Trained classification model
│   ├── regressor_random_forest. pkl   # Trained regression model
│   ├── feature_extractor_classifier.pkl
│   └── feature_extractor_regressor.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data cleaning & preparation
│   ├── feature_engineering.py        # Feature extraction (540+ features)
│   ├── train_classifier.py           # Classification model training
│   ├── train_regressor.py            # Regression model training
│   └── predict.py                    # Prediction utilities
│
├── app/
│   ├── app. py                        # Flask web application
│   ├── templates/
│   │   └── index.html                # Web interface
│   └── static/
│       ├── style.css                 # Styling with dark mode support
│       └── main.js                   # Frontend JavaScript logic
│
├── requirements.txt                   # Python dependencies
├── . gitignore
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishnubishnoi17/AutoJudge-Project.git
   cd AutoJudge-Project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements. txt
   ```

4. **Verify installation:**
   ```bash
   python --version  # Should be 3.8+
   pip list          # Check all packages installed
   ```

## 📖 Usage

### Option 1: Use Pre-trained Models (Quickstart)

If models are already trained, just run the web app: 

```bash
python app/app.py
```

Then open your browser and navigate to:  **http://localhost:5000**

### Option 2: Train Models from Scratch

#### Step 1: Preprocess Data
```bash
python src/data_preprocessing.py
```
**Output:** Creates `data/processed_data.csv` with cleaned text and combined features. 

#### Step 2: Train Classification Model
```bash
python src/train_classifier.py
```
**Output:** 
- `models/classifier_random_forest. pkl`
- `models/feature_extractor_classifier.pkl`
- Training accuracy and classification report

#### Step 3: Train Regression Model
```bash
python src/train_regressor.py
```
**Output:**
- `models/regressor_random_forest.pkl`
- `models/feature_extractor_regressor. pkl`
- MAE, RMSE, and R² scores

#### Step 4: Test Predictions
```bash
python src/predict.py
```
**Output:** Test prediction on "Two Sum" problem example.

#### Step 5: Launch Web App
```bash
python app/app.py
```

## 📊 Model Performance

### Classification Model (Random Forest)
- **Accuracy**: 54-60% (3-class problem)
- **Best Model**: Random Forest Classifier
- **Features**:  540 combined features
- **Classes**: Easy (54%), Medium (24%), Hard (22%)

### Regression Model (Random Forest)
- **MAE**: ~1.2 (Mean Absolute Error)
- **RMSE**: ~1.5 (Root Mean Squared Error)
- **R² Score**: ~0.65
- **Score Range**: 0-10

### Models Tested
| Model | Type | Performance |
|-------|------|-------------|
| Random Forest | Classification | ⭐ Best Accuracy |
| Logistic Regression | Classification | Good baseline |
| SVM | Classification | Moderate |
| Random Forest | Regression | ⭐ Best MAE |
| Gradient Boosting | Regression | Close second |
| Linear Regression | Regression | Baseline |

## 🧠 Features Engineered

### 1. **Text Length Features** (7 features)
- Character count
- Word count
- Average word length
- Sentence count
- Description length
- Input description length
- Output description length

### 2. **Mathematical Features** (4 features)
- Math operators count (+, -, *, /, =, <, >)
- Parentheses/brackets count
- Number count
- Formula presence

### 3. **Keyword Features** (29 features)
Algorithm keywords detected:
- `graph`, `tree`, `dynamic`, `dp`, `recursion`, `backtrack`
- `greedy`, `sort`, `search`, `binary`, `array`, `string`
- `matrix`, `linked`, `list`, `stack`, `queue`, `hash`
- `dfs`, `bfs`, `dijkstra`, `shortest`, `path`, `optimize`
- `maximum`, `minimum`, `subsequence`, `substring`

### 4. **TF-IDF Features** (500 features)
- Bi-gram text vectorization
- Semantic understanding of problem descriptions
- Captures unique terminology

**Total: 540 features**

## 🗂️ Dataset

The dataset (`problems_data.jsonl`) contains programming problems with:

| Field | Description | Example |
|-------|-------------|---------|
| `title` | Problem name | "Two Sum" |
| `description` | Full problem description | "Given an array of integers..." |
| `input_description` | Input format | "Array of integers and target" |
| `output_description` | Expected output | "Two indices" |
| `problem_class` | Difficulty label | "easy" / "medium" / "hard" |
| `problem_score` | Numerical difficulty | 3.24 (scale 0-10) |

**Format:** JSONL (JSON Lines) - one JSON object per line

## 💻 Technology Stack

### Backend
- **Python 3.8+**
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning
- **pandas 2.1.4** - Data manipulation
- **numpy 1.26.2** - Numerical computing
- **joblib 1.3.2** - Model serialization

### Machine Learning
- **TfidfVectorizer** - Text vectorization
- **RandomForestClassifier** - Classification
- **RandomForestRegressor** - Regression
- **LogisticRegression, SVM, GradientBoosting** - Alternative models

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with dark mode support
- **JavaScript (ES6+)** - Interactive UI, AJAX requests, dark mode toggle
- **Responsive Design** - Mobile-friendly

## 🎨 Web Interface Features

- ✅ **Real-time predictions** via AJAX
- ✅ **Dark mode toggle** with localStorage persistence
- ✅ **Loading animations** during prediction
- ✅ **Error handling** with user-friendly messages
- ✅ **Probability visualization** with colored progress bars
- ✅ **Responsive design** for all screen sizes
- ✅ **Smooth scrolling** to results
- ✅ **Form validation** (description required)
- ✅ **Clear button** to reset form

## 🔧 API Endpoints

### `GET /`
Returns the main web interface.

### `POST /predict`
Accepts JSON with problem description, returns prediction.

**Request:**
```json
{
  "title": "Two Sum",
  "description": "Given an array of integers.. .",
  "input_description":  "Array of integers and target",
  "output_description": "Two indices"
}
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "easy",
  "predicted_score": 3.24,
  "probabilities": {
    "Easy":  0.54,
    "Medium": 0.24,
    "Hard": 0.22
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## 🧪 Testing

### Test Prediction Module
```bash
python src/predict.py
```

### Test with cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Two Sum",
    "description": "Given an array of integers, return indices of two numbers that add up to target",
    "input_description": "Array and target integer",
    "output_description":  "Two indices"
  }'
```

## 🐛 Troubleshooting

### Models not loading
```bash
# Check if model files exist
ls -la models/*. pkl

# Should show 4 files:
# - classifier_random_forest.pkl
# - regressor_random_forest.pkl
# - feature_extractor_classifier.pkl
# - feature_extractor_regressor.pkl
```

### Port 5000 already in use
```bash
# Find process on port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

### Virtual environment issues
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📈 Future Enhancements

- [ ] Add more algorithms (Neural Networks, XGBoost)
- [ ] Implement user authentication
- [ ] Store prediction history in database
- [ ] Add batch prediction upload (CSV)
- [ ] Deploy to cloud (Heroku/AWS/GCP)
- [ ] Add model retraining API
- [ ] Implement A/B testing for models
- [ ] Add explainability (SHAP values)
- [ ] Export predictions to PDF/Excel

## 👥 Contributors

- **[vishnubishnoi17](https://github.com/vishnubishnoi17)** - Project Lead & Developer

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset inspired by competitive programming platforms
- Built with guidance from machine learning best practices
- UI design inspired by modern web applications

## 📞 Contact

- **GitHub**: [@vishnubishnoi17](https://github.com/vishnubishnoi17)
- **Repository**: [AutoJudge-Project](https://github.com/vishnubishnoi17/AutoJudge-Project)

---

**⭐ Star this repository if you find it helpful!**

Built with ❤️ for automatic programming problem difficulty assessment