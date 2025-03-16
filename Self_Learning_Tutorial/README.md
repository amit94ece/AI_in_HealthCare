# Healthcare Length of Stay Prediction Analysis

## Overview
This script performs comprehensive exploratory data analysis (EDA) and implements machine learning models to predict hospital patient length of stay. It includes detailed data visualization, statistical analysis, feature engineering, model training, and performance evaluation.

## Prerequisites
- Python 3.11 or above
- pip (Python package installer)
- Git

## Setup and Installation

### 1. Clone the Repository
```
git clone <repository-url>
cd AI_in_HealthCare
```
### 2. Create and Activate Virtual Environment
For macOS/Linux:

# Create virtual environment
```python3 -m venv sltvenv```

# Activate virtual environment
```source sltvenv/bin/activate```

For Windows:
# Create virtual environment
```python -m venv sltvenv```

# Activate virtual environment
```.\sltvenv\Scripts\activate```


### 3. Install Dependencies
```pip install -r requirements.txt```


# Project Structure
AI_in_HealthCare/
│
├── Self_Learning_Tutorial/
│   ├── data/
│   │   └── healthcare_dataset.csv
│   │
│   ├── scripts/
│       └── expda.py
├── requirements.txt
└── README.md


#Features
## 1. Exploratory Data Analysis (EDA)
Distribution Analysis

Length of Stay distribution

Age distribution

Medical condition distribution

Correlation Analysis

Age vs Length of Stay

Billing Amount vs Length of Stay

Feature correlations

Demographic Analysis

Gender-based analysis

Age group analysis

Medical condition patterns

Temporal Analysis

Admission patterns

Seasonal trends

Day-of-week effects

## 2. Feature Engineering
Log transformation of billing amount

Age-related features

Time-based features

Interaction features

Medical condition encoding

## 3. Model Implementation
Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

Model pipeline with preprocessing

## 4. Model Evaluation
R-squared (R²) score

Root Mean Square Error (RMSE)

Mean Absolute Error (MAE)

Cross-validation scores

Prediction intervals

## Usage
### Running the Analysis
# Ensure you're in the project directory and virtual environment is activated
```cd Self_Learning_Tutorial/scripts
python3 expda.py
```

# Key Visualizations
The script generates multiple visualizations:

Distribution Plots

Length of Stay distribution

Age distribution by medical condition

Billing amount distribution

Correlation Plots

Feature correlation heatmap

Age vs Length of Stay scatter plots

Billing vs Length of Stay analysis

Model Performance Plots

Actual vs Predicted values

Residuals analysis

Feature importance

Model comparison charts

Prediction intervals (Random Forest)

Statistical Analysis

Box plots by medical condition

Percentile analysis

Gender-based comparisons

Age group analysis

## Model Performance
Metrics Format
Random Forest:
RMSE: X.XX
MAE: X.XX
R2: X.XX

Gradient Boosting:
RMSE: X.XX
MAE: X.XX
R2: X.XX

XGBoost:
RMSE: X.XX
MAE: X.XX
R2: X.XX


# Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost


# If you encounter version conflicts, try
pip install -r requirements.txt --no-cache-dir


# Memory Issues

Reduce sample size for analysis

Use batch processing for large datasets

Close other applications when running heavy computations


# Contributing
Fork the repository

Create your feature branch ( git checkout -b feature/AmazingFeature)

Commit your changes ( git commit -m 'Add some AmazingFeature')

Push to the branch ( git push origin feature/AmazingFeature)

Open a Pull Request

# License
This project is licensed under the MIT License - see the LICENSE file for details