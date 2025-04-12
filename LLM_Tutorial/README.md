# Healthcare Data Preprocessing Pipeline

This project implements a data preprocessing pipeline for analyzing medication adherence patterns in healthcare data. The pipeline processes synthetic patient data to extract meaningful features about medication adherence and patient demographics.

## Prerequisites

- Python 3.8 or higher
- AWS Account (for AWS credentials setup)
- Git

## Project Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

# Create virtual environment
```bash
python -m venv venv
```

## Activate virtual environment
### On Windows
```bash
venv\Scripts\activate
```
### On macOS/Linux
```bash
source venv/bin/activate
```

### Install the requirements file:
```bash
pip install -r requirements.txt
```

## Create an AWS User
Sign in to AWS Management Console

Navigate to IAM (Identity and Access Management)

Click "Users" → "Add user"

Set username and enable "Access key - Programmatic access"

Attach necessary permissions (e.g., AWSBedrock)

Complete user creation and save the credentials securely


## Configure AWS credentials:
```bash
aws configure --profile healthcare-project
```


## Enter the following when prompted:

AWS Access Key ID - Acess Key ID from "Create an AWS User Step"

AWS Secret Access Key - Secret Access Key from "Create an AWS User Step"

Default region - us-east-1

Default output format (json) - Just press enter

5. Project Structure
.
├── data_preprocessing.py
├── claude.py
├── requirements.txt
├── README.md
├── synthea_data/
│   ├── patients.csv
│   ├── medications.csv
│   └── encounters.csv
└── processed_data/
    └── adherence_features.csv


## Usage

### Data Preprocessing Pipeline

The data_preprocessing.py script processes patient data to analyze medication adherence:
```bash
python3 -m data_preprocessing
```

This will:

Load patient, medication, and encounter data

Process chronic conditions

Calculate medication adherence metrics

Generate demographic features

Save processed data to processed_data/adherence_features.csv

## Output Features
The pipeline generates the following features:

ADHERENCE_RATE: Average medication adherence rate

AVG_GAP: Average gap between medication refills

MAX_GAP: Maximum gap between refills

REFILL_COUNT: Total number of medication refills

MEDICATION_COUNT: Number of unique medications

Demographic information (age, gender, income)

Chronic condition flags


## Data Requirements
Input CSV files should be placed in the synthea_data/ directory:

patients.csv: Patient demographic information

medications.csv: Medication prescriptions and refills

encounters.csv: Patient encounters and conditions

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.

## License
This project is open-source and available under the MIT License.


