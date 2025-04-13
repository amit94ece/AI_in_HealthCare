# Healthcare Data Preprocessing Pipeline

This project combines data preprocessing with advanced AI analysis using Amazon Bedrock's Claude model to provide comprehensive medication adherence insights and personalized healthcare recommendations. This might cost money depending on how many times the model is invoked. Pricing details for invoking Claude 3.7 via bedrock is in the given link (https://aws.amazon.com/bedrock/pricing/). For this example and running the scripts shouldn't cost more than a few cents.

## Overview

The system consists of two main components:
1. Data Preprocessing Pipeline (data_preprocessing.py)
2. AI-Powered Analysis Engine (claude.py)

## Prerequisites

- Python 3.8 or higher
- AWS Account (for AWS credentials setup)
- Git
- DataSource - (https://synthea.mitre.org/downloads). Download 100 Sample Synthetic Patient Records, CSV: 7 MB and move the required files to the project sturcture below.

## Project Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

# Create virtual environment
```bash
python -m venv llmvenv
```

## Activate virtual environment
### On Windows
```bash
llmvenv\Scripts\activate
```
### On macOS/Linux
```bash
source llmvenv/bin/activate
```

### Install the requirements file:
```bash
pip install -r requirements.txt
```

### Create an AWS User
Sign in to AWS Management Console

Navigate to IAM (Identity and Access Management)

Click "Users" → "Add user"

Set username and enable "Access key - Programmatic access"

Attach necessary permissions (e.g., AWSBedrock)

Complete user creation and save the credentials securely


### Configure AWS credentials:
```bash
aws configure --profile healthcare-project
```


### Enter the following when prompted:

AWS Access Key ID - Acess Key ID from "Create an AWS User Step"

AWS Secret Access Key - Secret Access Key from "Create an AWS User Step"

Default region - us-east-1

Default output format (json) - Just press enter

### Export AWS Profile:
#### On Linux/macOS
```bash
export AWS_PROFILE=healthcare-project
```
#### On Windows (Command Prompt)
```bash
set AWS_PROFILE=healthcare-project
```

#### On Windows (PowerShell)
```bash
$env:AWS_PROFILE="healthcare-project"
```


### Enable Model Access in AWS Account
Refer link for details - https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html

You need to enable access for Claude 3.7

### Model Configuration

#### Important Note About Model ID
The model_id in the configuration must be updated with the corresponding Amazon Bedrock model inference ARN. This is crucial for successful model invocation.

Example format:
```bash
arn:aws:bedrock:us-east-1:<aws account id>:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0
```


## Project Structure
```bash
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
```

## Usage

### 1. Data Preprocessing Pipeline

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

### 2. AI Analysis Engine (claude.py)

The Claude 3.7 module leverages Amazon Bedrock's Claude model for sophisticated healthcare analysis through advanced prompt engineering techniques:


### Prompt Engineering Techniques

The `claude.py` implementation utilizes four specific prompt engineering techniques for analyzing medication adherence patterns:

1. **Zero-Shot Prompting**
   - Direct analysis without providing examples or prior context
   - Model analyzes patient data and generates insights based purely on the given information
   - Useful for straightforward adherence pattern analysis where the task is clearly defined
   - Best suited for cases where the model can directly infer patterns from raw data

2. **Few-Shot Prompting**
   - Provides the model with a few examples before analyzing new cases
   - Helps establish clear patterns and expected output format through examples
   - Improves consistency and accuracy of adherence analysis
   - Particularly effective when analyzing complex adherence patterns that benefit from example-based learning

3. **Chain-of-Thought Prompting**
   - Breaks down complex adherence analysis into logical, sequential steps
   - Enables step-by-step reasoning about adherence patterns and causes
   - Provides transparent decision-making process for recommendations
   - Helps in understanding the relationship between different adherence factors
   - Results in more detailed and well-reasoned analysis

4. **Tree-of-Thought Prompting**
   - Explores multiple parallel reasoning paths simultaneously
   - Considers different possible explanations for adherence patterns
   - Evaluates multiple solution paths before arriving at conclusions
   - Particularly useful for complex cases with multiple contributing factors
   - Enables comprehensive analysis of interrelated adherence issues

These techniques are used individually or in combination depending on the complexity of the analysis required and the specific characteristics of each patient case. The selection of technique is based on factors such as:
- Complexity of adherence patterns
- Amount of available patient data
- Type of insights needed
- Specificity of recommendations required

Each technique contributes to a comprehensive understanding of medication adherence patterns and helps generate appropriate, patient-specific recommendations for improving medication adherence.


## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.

## License
This project is open-source and available under the MIT License.


