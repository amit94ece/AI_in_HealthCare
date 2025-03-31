# AI in Healthcare: MIMIC-IV Data Analysis

## Project Overview

This project implements machine learning and deep learning models for analyzing MIMIC-IV healthcare data, with a specific focus on Congestive Heart Failure (CHF) prediction. The implementation includes data preprocessing, feature engineering, and neural network modeling using PyTorch.

## Data Preprocessing

The `data_preprocessing.py` script handles the following tasks:
- Reads and processes MIMIC-III CSV compressed files
- Filters data for CHF patients using ICD-9 codes
- Merges relevant tables (diagnoses, admissions, patients, ICU stays)
- Processes lab events data for key indicators:
  - Brain Natriuretic Peptide (BNP)
  - Creatinine
  - Sodium
  - Potassium
- Handles missing values and performs feature engineering

### Key Features

- Age calculation from admission and birth dates
- Length of stay computation
- Race/ethnicity standardization
- Lab values normalization
- Comprehensive error handling and logging

### Traditional ML Models
- Random Forest Classifier
  - Hyperparameter tuning via GridSearchCV
  - Feature importance analysis
  - Handles class imbalance through class weights

- XGBoost
  - Gradient boosting implementation
  - Early stopping to prevent overfitting
  - Built-in feature importance metrics

- Logistic Regression
  - Baseline model for comparison
  - L2 regularization
  - Scaled features using StandardScaler

## Neural Network Model

The implementation includes a custom neural network with:
- Multiple dropout layers for regularization
- LeakyReLU activation functions
- Binary classification output (Sigmoid activation)
- Early stopping mechanism
- Learning rate scheduling
- Gradient clipping for training stability

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```
AI_in_HealthCare/
├── MIMIC_ML_DL/
│ ├── data_preprocessing.py
│ └── train_models.py
├── data/
│ ├── hosp/
│ └── icu/
└── README.md
```

## Usage

Ensure MIMIC-IV data is in the correct directory structure

## Environment Setup

Create and Activate Virtual Environment
For macOS/Linux:

### Create virtual environment

```python3 -m venv myvenv```

### Activate virtual environment

```source myvenv/bin/activate```

For Windows:
### Create virtual environment

```python3 -m venv myvenv```

### Activate virtual environment

```.\myvenv\Scripts\activate```


## Install Dependencies

```pip install -r requirements.txt```


## Perform Data Processing

```python3 data_preprocessing.py```

## Train and evaluate ML models

```python3 train_models.py```

## Train and evaluate DL models

```python3 train_neural.py```

## Output

Processed dataset (CSV)
Training curves visualization
ROC curve plots
Model performance metrics
Trained model checkpoints

## Metrics Tracked

AUC-ROC
F1 Score
Precision
Recall
Accuracy


## Note

Access to MIMIC-III data requires appropriate credentialing through PhysioNet.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

MIMIC-III database
PhysioNet for providing access to healthcare data