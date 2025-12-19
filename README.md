# AutoJudge - Programming Problem Difficulty Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-2.0.0-orange)

## 🎯 Overview

**AutoJudge** is an intelligent system that predicts the difficulty level of programming problems using advanced machine learning techniques. The system provides both classification (Easy/Medium/Hard) and regression (0-100 score) predictions.

### ✨ Version 2.0 Features

- 🎚️ **Enhanced Scale**: Difficulty scores now range from 0-100 (upgraded from 0-10)
- 🚀 **Advanced Models**: Added LightGBM and XGBoost for improved accuracy
- 📝 **Better Documentation**: Comprehensive code comments for readability
- 🎯 **Improved Accuracy**: Multiple model architectures for best performance

---

## 🏗️ Project Structure

```
AutoJudge-Project/
├── data/
│   ├── problems_data.jsonl       # Raw problem data
│   └── processed_data.csv        # Preprocessed data
├── models/
│   ├── classifier.pkl            # Trained classification model
│   ├── regressor.pkl             # Trained regression model
│   ├── feature_extractor_classifier.pkl
│   ├── feature_extractor_regressor.pkl
│   └── *. png                     # Performance visualization plots
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── feature_engineering.py   # Feature extraction
│   ├── train_classifier.py      # Classification model training
│   ├── train_regressor.py       # Regression model training
│   └── predict.py                # Prediction utilities
├── app/
│   ├── app.py                    # Flask web application
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/
│       ├── css/
│       └── js/
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vishnubishnoi17/AutoJudge-Project.git
cd AutoJudge-Project
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Usage

### Step 1: Data Preprocessing

Preprocess the raw data and scale scores from 0-10 to 0-100:

```bash
cd src
python data_preprocessing.py
```

**Output**:  `data/processed_data.csv`

### Step 2: Train Classification Model

Train multiple classifiers including LightGBM and XGBoost:

```bash
python train_classifier.py
```

**Models Trained**:
- Logistic Regression
- Random Forest
- SVM
- LightGBM ✨
- XGBoost ✨

**Output**: Best model saved to `models/classifier.pkl`

### Step 3: Train Regression Model

Train multiple regressors for score prediction (0-100):

```bash
python train_regressor.py
```

**Models Trained**:
- Linear Regression
- Random Forest
- Gradient Boosting
- LightGBM ✨
- XGBoost ✨

**Output**: Best model saved to `models/regressor.pkl`

### Step 4: Make Predictions

#### Option A: Command Line

```bash
python predict.py
```

#### Option B: Python API

```python
from predict import load_predictor

# Load predictor
predictor = load_predictor('models')

# Make prediction
results = predictor.predict(
    title="Two Sum Problem",
    description="Given an array of integers and a target, find two numbers that add up to the target.",
    input_description="An array of integers and a target integer.",
    output_description="Indices of the two numbers."
)

print(f"Class: {results['predicted_class']}")
print(f"Score: {results['predicted_score']}/100")
print(f"Interpretation: {results['score_interpretation']}")
```

#### Option C: Web Application

```bash
cd app
python app.py
```

Visit:  `http://localhost:5000`

---

## 🎯 Features

### 1. Multi-Model Architecture

The system trains and compares multiple models:

| Model Type | Classification | Regression |
|------------|---------------|------------|
| Linear Models | Logistic Regression | Linear Regression |
| Tree Ensembles | Random Forest | Random Forest |
| SVM | RBF Kernel SVM | - |
| Gradient Boosting | LightGBM, XGBoost | LightGBM, XGBoost, GB |

### 2. Comprehensive Feature Engineering

- **Text Features**: Character count, word count, sentence count
- **Mathematical Features**: Operator count, formula detection
- **Keyword Features**: Algorithm and data structure terms
- **TF-IDF Features**: Statistical text representation

### 3. 0-100 Difficulty Scale

Scores are automatically scaled and interpreted:

| Score Range | Interpretation |
|-------------|----------------|
| 0-20 | Very Easy |
| 20-40 | Easy |
| 40-60 | Medium |
| 60-80 | Hard |
| 80-100 | Very Hard |

---

## 📈 Model Performance

After training, performance visualizations are automatically generated:

- `models/classifier_comparison.png` - Classification accuracy comparison
- `models/regressor_comparison.png` - Regression metrics (MAE, R²)
- `models/prediction_scatter.png` - Predicted vs actual scores

---

## 🌐 Web API Endpoints

### `POST /predict`

Make a difficulty prediction. 

**Request**:
```json
{
    "title": "Problem Title",
    "description": "Problem description.. .",
    "input_description":  "Input format.. .",
    "output_description":  "Output format..."
}
```

**Response**:
```json
{
    "success": true,
    "predicted_class":  "Medium",
    "predicted_score": 55.23,
    "score_interpretation": "Medium",
    "probabilities": {
        "Easy": 0.15,
        "Medium": 0.70,
        "Hard": 0.15
    }
}
```

### `GET /health`

Check application health status.

### `GET /about`

Get information about models and features.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Data Processing**: pandas, numpy
- **Web Framework**:  Flask
- **Visualization**: matplotlib, seaborn
- **Serialization**: joblib

---

## 📝 Code Quality

### Version 2.0 Improvements

✅ **Comprehensive Comments**: Every function and class thoroughly documented  
✅ **Type Hints**: Better code clarity and IDE support  
✅ **Error Handling**: Robust error checking and informative messages  
✅ **Modular Design**: Clean separation of concerns  
✅ **Logging**: Detailed progress and status messages  

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Vishnu Bishnoi** - [@vishnubishnoi17](https://github.com/vishnubishnoi17)

---

## 🙏 Acknowledgments

- Programming problem datasets
- Open-source ML community
- Contributors and testers

---

## 📞 Contact

For questions or suggestions, please open an issue or contact: 
- GitHub: [@vishnubishnoi17](https://github.com/vishnubishnoi17)

---

## 🔮 Future Enhancements

- [ ] Deep learning models (BERT, transformers)
- [ ] Multi-language support
- [ ] Real-time difficulty adjustment
- [ ] Integration with online judges
- [ ] User feedback loop for model improvement

---

**⭐ If you find this project helpful, please star the repository! **
