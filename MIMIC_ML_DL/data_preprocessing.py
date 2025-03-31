import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def read_mimic_csv(mimic_dir, csv_file):
    """
    Read MIMIC-IV csv.gz data file into pandas dataframe.
    """
    # Remove .csv extension if present to handle both csv and csv.gz files
    csv_file = csv_file.replace('.csv', '')
    
    hosp_path = os.path.join(mimic_dir, 'hosp', f'{csv_file}.csv.gz')
    icu_path = os.path.join(mimic_dir, 'icu', f'{csv_file}.csv.gz')
    
    if os.path.exists(hosp_path):
        return pd.read_csv(hosp_path, compression='gzip')
    elif os.path.exists(icu_path):
        return pd.read_csv(icu_path, compression='gzip')
    else:
        raise FileNotFoundError(f"Could not find {csv_file}.csv.gz in {mimic_dir}/hosp or {mimic_dir}/icu")

def get_chf_data(mimic_dir):
    """
    Read, filter, merge, and return raw MIMIC-IV data for CHF patients.
    """
    # Read ICD codes
    d_icd = read_mimic_csv(mimic_dir, 'd_icd_diagnoses.csv')
    logging.info("D_ICD columns: %s", d_icd.columns.tolist())
    
    # Read diagnoses
    diagnoses = read_mimic_csv(mimic_dir, 'diagnoses_icd.csv')
    logging.info("Diagnoses columns: %s", diagnoses.columns.tolist())
    
    # Filter for CHF diagnoses (ICD-9: 4280)
    chf_icd9 = d_icd[d_icd['ICD9_CODE'] == '4280']
    logging.info(f"Found {len(chf_icd9)} CHF diagnosis codes")
    
    # Get all diagnoses with CHF
    chf_diagnoses = diagnoses[diagnoses['ICD9_CODE'].isin(chf_icd9['ICD9_CODE'])]
    logging.info(f"Found {len(chf_diagnoses)} CHF diagnoses")
    
    # Read admissions data
    admissions = read_mimic_csv(mimic_dir, 'admissions.csv')
    logging.info("Admissions columns: %s", admissions.columns.tolist())
    logging.info(f"Read {len(admissions)} admission records")
    
    # Merge diagnoses with admissions
    chf_admissions = pd.merge(chf_diagnoses, admissions, on=['HADM_ID', 'SUBJECT_ID'])
    
    # Clean up duplicate columns
    columns_to_drop = [col for col in chf_admissions.columns if '_x' in col or '_y' in col]
    chf_admissions = chf_admissions.drop(columns=columns_to_drop)
    
    logging.info("CHF Admissions columns after cleanup: %s", chf_admissions.columns.tolist())
    logging.info(f"Merged data has {len(chf_admissions)} records")
    
    # Read patients data
    patients = read_mimic_csv(mimic_dir, 'patients.csv')
    logging.info("Patients columns: %s", patients.columns.tolist())
    
    # Merge with patients data
    chf_patients = pd.merge(chf_admissions, patients, on='SUBJECT_ID')
    
    # Clean up duplicate columns again
    columns_to_drop = [col for col in chf_patients.columns if '_x' in col or '_y' in col]
    chf_patients = chf_patients.drop(columns=columns_to_drop)
    
    # Read ICU stays
    icustays = read_mimic_csv(mimic_dir, 'icustays.csv')
    
    # Merge with ICU stays
    chf_icu = pd.merge(chf_patients, icustays, on=['SUBJECT_ID', 'HADM_ID'])
    
    # Clean up duplicate columns one more time
    columns_to_drop = [col for col in chf_icu.columns if '_x' in col or '_y' in col]
    chf_icu = chf_icu.drop(columns=columns_to_drop)
    
    logging.info(f"After merging with ICU stays: {len(chf_icu)} records")
    
    # Read lab events (in chunks due to large size)
    lab_events_path = os.path.join(mimic_dir, 'hosp', 'labevents.csv.gz')
    lab_events_chunks = pd.read_csv(lab_events_path, 
                                  compression='gzip',
                                  chunksize=1000000)
    
    # Filter for relevant lab tests for CHF
    bnp_itemid = 51003      # Brain Natriuretic Peptide (BNP)
    creatinine_itemid = 50912  # Creatinine
    sodium_itemid = 50983      # Sodium
    potassium_itemid = 50971   # Potassium
    
    lab_events_filtered = []
    chunk_count = 0
    
    for chunk in lab_events_chunks:
        chunk_count += 1
        logging.info(f"Processing lab events chunk {chunk_count}")
        
        # Filter for relevant lab tests and hadm_ids in chf_icu
        chunk_filtered = chunk[
            (chunk['ITEMID'].isin([bnp_itemid, creatinine_itemid, sodium_itemid, potassium_itemid])) & 
            (chunk['HADM_ID'].isin(chf_icu['HADM_ID']))
        ]
        if not chunk_filtered.empty:
            lab_events_filtered.append(chunk_filtered)
            logging.info(f"Found {len(chunk_filtered)} relevant lab events in chunk {chunk_count}")
    
    # Combine filtered chunks
    if lab_events_filtered:
        lab_events = pd.concat(lab_events_filtered)
        logging.info(f"Total lab events after filtering: {len(lab_events)}")
        
        # Pivot lab events to get one row per admission with lab values as columns
        lab_pivot = lab_events.pivot_table(
            index='HADM_ID', 
            columns='ITEMID', 
            values='VALUENUM', 
            aggfunc='mean'
        )
        
        # Rename columns
        lab_pivot.columns = [
            'bnp' if col == bnp_itemid else
            'creatinine' if col == creatinine_itemid else
            'sodium' if col == sodium_itemid else
            'potassium' if col == potassium_itemid else
            f'lab_{col}' for col in lab_pivot.columns
        ]
        
        # Merge lab values with CHF data
        chf_data = pd.merge(chf_icu, lab_pivot, on='HADM_ID', how='left')
        logging.info(f"Final dataset shape after merging lab values: {chf_data.shape}")
    else:
        chf_data = chf_icu
        logging.warning("No lab events found matching the criteria")
    
    return chf_data

def convert_date_format(date_str):
    try:
        date_obj = pd.to_datetime(date_str)
        return date_obj
    except:
        return pd.NaT

def calculate_age(admit_date, birth_date):
    try:
        if pd.isna(admit_date) or pd.isna(birth_date):
            return np.nan
        years = admit_date.year - birth_date.year
        if (admit_date.month, admit_date.day) < (birth_date.month, birth_date.day):
            years -= 1
        return years
    except:
        return np.nan

def process_chf_data(data):
    """
    Process the CHF data to create features for modeling.
    """
    logging.info("Processing CHF data...")
    logging.info("Available columns: %s", data.columns.tolist())
    
    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()
    
    # Convert dates
    processed_data['DOB'] = processed_data['DOB'].apply(convert_date_format)
    processed_data['ADMITTIME'] = processed_data['ADMITTIME'].apply(convert_date_format)
    processed_data['DISCHTIME'] = pd.to_datetime(processed_data['DISCHTIME'])
    
    # Calculate age at admission
    processed_data['AGE'] = processed_data.apply(
        lambda row: calculate_age(row['ADMITTIME'], row['DOB']), axis=1
    )
    
    # Filter for valid ages between 1 and 120
    processed_data = processed_data[
        (processed_data['AGE'] >= 1) & 
        (processed_data['AGE'] <= 120)
    ].copy()
    
    # Calculate length of stay if not already present
    if 'LOS' not in processed_data.columns:
        processed_data['LOS'] = (
            processed_data['DISCHTIME'] - processed_data['ADMITTIME']
        ).dt.total_seconds() / (24 * 60 * 60)
    
    # Define race mapping
    race_mapping = {
        'WHITE': 'WHITE',
        'BLACK/AFRICAN AMERICAN': 'BLACK',
        'HISPANIC/LATINO': 'HISPANIC',
        'ASIAN': 'ASIAN',
        'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER',
        'OTHER': 'OTHER',
        'UNKNOWN': 'UNKNOWN'
    }
    
    # Map race categories
    processed_data['RACE'] = processed_data['ETHNICITY'].apply(
        lambda x: next((v for k, v in race_mapping.items() if k in str(x).upper()), 'OTHER')
    )
    
    # Select relevant features
    features = [
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'GENDER', 'AGE', 'RACE',
        'HOSPITAL_EXPIRE_FLAG', 'FIRST_CAREUNIT', 'LOS',
        'bnp', 'creatinine', 'sodium', 'potassium'
    ]
    
    # Select only the columns that exist in the data
    features = [f for f in features if f in processed_data.columns]
    logging.info("Selected features: %s", features)
    
    final_data = processed_data[features].copy()
    
    # Handle missing values in lab results
    lab_cols = ['bnp', 'creatinine', 'sodium', 'potassium']
    for col in lab_cols:
        if col in final_data.columns:
            mean_value = final_data[col].mean()
            final_data[col] = final_data[col].fillna(mean_value)
            logging.info(f"Filled {final_data[col].isna().sum()} missing values in {col}")
    
    # Only keep rows with essential lab results
    final_data = final_data.dropna(subset=['creatinine', 'sodium', 'potassium'])
    
    # One-hot encode categorical variables
    categorical_cols = ['GENDER', 'RACE', 'FIRST_CAREUNIT']
    categorical_cols = [col for col in categorical_cols if col in final_data.columns]
    
    final_data_encoded = pd.get_dummies(
        final_data, 
        columns=categorical_cols, 
        drop_first=False
    )
    
    # Drop any remaining rows with missing values
    final_data_encoded = final_data_encoded.dropna()
    
    logging.info(f"Final processed data shape: {final_data_encoded.shape}")
    
    # Print summary statistics
    logging.info("\nSummary statistics for numerical features:")
    numerical_cols = final_data_encoded.select_dtypes(include=[np.number]).columns
    logging.info(final_data_encoded[numerical_cols].describe())
    
    # Print categorical feature distributions
    logging.info("\nCategorical feature distributions:")
    for col in categorical_cols:
        logging.info(f"\n{col} distribution:")
        logging.info(final_data[col].value_counts(normalize=True))
    
    logging.info("\nMortality rate:")
    logging.info(f"{final_data_encoded['HOSPITAL_EXPIRE_FLAG'].mean():.2%}")
    
    # Additional feature statistics
    logging.info("\nFeature set information:")
    logging.info(f"Total features after encoding: {len(final_data_encoded.columns)}")
    logging.info(f"Numerical features: {len(numerical_cols)}")
    logging.info(f"Categorical features encoded: {len(categorical_cols)}")
    logging.info("Encoded feature names:")
    logging.info(final_data_encoded.columns.tolist())
    
    return final_data_encoded

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set the path to your data folder
    mimic_dir = "data"
    
    try:
        logging.info("Starting data processing...")
        
        # Get raw CHF data
        chf_data = get_chf_data(mimic_dir)
        logging.info("Successfully loaded CHF data")
        
        # Process the data
        processed_data = process_chf_data(chf_data)
        logging.info(f"Processed data shape: {processed_data.shape}")
        
        # Save processed data
        output_file = "processed_chf_data.csv"
        processed_data.to_csv(output_file, index=False)
        logging.info(f"Saved processed data to {output_file}")
        
        # Print final statistics
        logging.info("\nFinal dataset statistics:")
        logging.info(f"Total patients: {processed_data['SUBJECT_ID'].nunique()}")
        logging.info(f"Total admissions: {processed_data['HADM_ID'].nunique()}")
        logging.info(f"Total features: {len(processed_data.columns)}")
        
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace:", exc_info=True)
        raise
