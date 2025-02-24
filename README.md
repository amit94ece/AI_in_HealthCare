# AI in Healthcare: Patient Data Analysis and NLP
This repository contains Python scripts for analyzing patient data and performing Natural Language Processing (NLP) on medical notes using various techniques and libraries.

## Project Structure
```
AI_in_Healthcare/
│
├── MIMIC_VISUALIZATION/
│   ├── data/                          # Data directory
│   │   ├── PATIENTS.csv
│   │   ├── ADMISSIONS.csv
│   │   ├── MICROBIOLOGYEVENTS.csv
│   │   ├── PRESCRIPTIONS.csv
│   │   └── ICUSTAYS.csv
│   │                 
│   ├── patient_microbiology_events.py
│   ├── patient_drugs_icu_stays.py
│   └── patient_admission_analysis.py
│
├── MIMIC_NLP/
│   ├── data/
│   │   ├── NOTEEVENTS.csv.gz
│   │   └── DIAGNOSES_ICD.csv.gz
│   │
│   ├── nlp_spacymodels.py
│   └── nlp_medspacy.py
│
├── README.md
└── .gitignore
```

## MIMIC_VISUALIZATION

This folder MIMIC_VISUALIZATION contains three Python scripts that create interactive dashboards for analyzing patient data using Dash and Plotly. Each script focuses on different aspects of patient information and medical events.

### Files

1. `patient_microbiology_events.py`
2. `patient_drugs_icu_stays.py`
3. `patient_admission_analysis.py`

### Features

#### Patient Microbiology Events Analysis

- Weekly distribution of microbiology specimens
- Organism vs Specimen Type heatmap
- Resistance Pattern heatmap
- Distribution of Specimen Types (Treemap)
- Age Distribution by Specimen Type (Violin Plot)
- Distribution of Test Timing by Organism (Box Plot)

#### Patient Drugs & ICU Stays Analysis

- Top 10 Most Prescribed Medications
- ICU Length of Stay Distribution
- Distribution of Drug Types
- ICU Length of Stay vs Number of Medications
- Age vs Number of Medications
- Distribution of ICU Types

#### Patient Admission Analysis

- Gender Distribution
- Age Distribution at Admission
- Admission Locations Distribution
- Length of Stay vs Age
- Top 10 Diagnoses Distribution
- Age Distribution: Admission vs Death (Butterfly Chart)

## MIMIC_NLP
This folder contains Python scripts for performing Natural Language Processing (NLP) on medical notes, focusing on diabetes-related text.

### Files
1. `nlp_spacymodels.py`
2. `nlp_medspacy.py`

### Features
- Entity extraction using SpaCy models (en_core_sci_lg and en_core_web_sm)
- Entity extraction using MedSpaCy
- Word embeddings generation using Word2Vec
- Visualization of entity relationships and word similarities
- t-SNE visualization of word embeddings

## Requirements

- Python 3.x
- pandas
- plotly
- dash
- numpy

## Usage

1. Ensure you have the required datasets in a `./data/` directory:
   - PATIENTS.csv
   - ADMISSIONS.csv
   - MICROBIOLOGYEVENTS.csv
   - PRESCRIPTIONS.csv
   - ICUSTAYS.csv

2. Install the required packages:
   ```
   pip install pandas plotly dash numpy
   ```

3. Run each script individually:
   ```
   python patient_microbiology_events.py
   python patient_drugs_icu_stays.py
   python patient_admission_analysis.py
   ```

4. Open a web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.


5. Before running a new dashboard script, ensure the previous script is killed to free up port 8050:

Check if any application is running on port 8050:
```
lsof -i :8050
```

Kill the process if found:
```
kill -9 <PID>
```

## Customization

Each dashboard includes filters that allow users to interact with the data:

- Age Range
- Specimen Type / ICU Type / Diagnosis
- Organism / Drug Type / Insurance

Adjust these filters to explore different aspects of the patient data.





## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.

## License

This project is open-source and available under the MIT License.
