import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_adherence_data(patients_file, medications_file, encounters_file, output_file):
    """
    Preprocess synthetic patient data to extract medication adherence features.
    
    Args:
        patients_file: Path to patients CSV file
        medications_file: Path to medications CSV file
        encounters_file: Path to encounters CSV file
        output_file: Path to save preprocessed data
    
    Returns:
        DataFrame with adherence features
    """
    print("Loading data files...")
    patients_df = pd.read_csv(patients_file)
    medications_df = pd.read_csv(medications_file)
    encounters_df = pd.read_csv(encounters_file)
    
    print("Available columns in patients_df:")
    print(patients_df.columns.tolist())
    
    print("Available columns in encounters_df:")
    print(encounters_df.columns.tolist())
    
    print("Available columns in medications_df:")
    print(medications_df.columns.tolist())
    
    # Extract conditions from encounters and medications
    print("Extracting conditions from encounters and medications...")
    
    # Get conditions from encounters
    encounter_conditions = encounters_df[['PATIENT', 'REASONDESCRIPTION']].dropna()
    encounter_conditions = encounter_conditions.rename(columns={'PATIENT': 'PATIENT'})
    
    # Get conditions from medications
    medication_conditions = medications_df[['PATIENT', 'REASONDESCRIPTION']].dropna()
    medication_conditions = medication_conditions.rename(columns={'PATIENT': 'PATIENT'})
    
    # Combine conditions from both sources
    all_conditions = pd.concat([encounter_conditions, medication_conditions])
    
    # Group conditions by patient
    patient_conditions = all_conditions.groupby('PATIENT')['REASONDESCRIPTION'].apply(
        lambda x: ', '.join(set(x))).reset_index()
    patient_conditions.rename(columns={'REASONDESCRIPTION': 'CONDITIONS'}, inplace=True)
    
    # Merge conditions with patients
    # patients_df = pd.merge(patients_df, patient_conditions, on='PATIENT', how='left')
    patients_df = pd.merge(patients_df, patient_conditions, left_on='Id', right_on='PATIENT', how='left')

    patients_df['CONDITIONS'].fillna('', inplace=True)
    
    # Define chronic conditions to look for
    chronic_conditions = ['diabetes', 'hypertension', 'heart failure', 'chronic', 'COPD']
    
    # Create flags for chronic conditions
    for condition in chronic_conditions:
        condition_name = condition.upper().replace(' ', '_')
        patients_df[condition_name] = patients_df['CONDITIONS'].str.contains(
            condition, case=False, na=False).astype(int)
    
    # Calculate total chronic conditions
    patients_df['CHRONIC_CONDITIONS'] = patients_df[[c.upper().replace(' ', '_') for c in chronic_conditions]].sum(axis=1)
    
    # Filter for patients with chronic conditions
    chronic_patients = patients_df[patients_df['CHRONIC_CONDITIONS'] > 0]
    print(f"Found {len(chronic_patients)} patients with chronic conditions")
    
    # Process medication data
    print("Processing medication data...")
    
    # Convert date strings to datetime objects
    for col in ['START', 'STOP']:
        if col in medications_df.columns:
            medications_df[col] = pd.to_datetime(medications_df[col], errors='coerce')
    
    # Filter medications for chronic patients
    patient_meds = pd.merge(
        chronic_patients[['PATIENT']],
        medications_df,
        on='PATIENT',
        how='inner'
    )
    
    # Sort by patient and date
    patient_meds = patient_meds.sort_values(['PATIENT', 'DESCRIPTION', 'START'])
    
    # Calculate adherence metrics
    print("Calculating adherence metrics...")
    
    # Group by patient and medication
    patient_med_groups = patient_meds.groupby(['PATIENT', 'DESCRIPTION'])
    
    # Create a new DataFrame to store adherence metrics
    adherence_data = []
    
    for (patient, med), group in patient_med_groups:
        # Sort by start date
        group = group.sort_values('START')
        
        # Calculate days between refills
        group['NEXT_REFILL'] = group['START'].shift(-1)
        group['DAYS_SUPPLY_GAP'] = (group['NEXT_REFILL'] - group['STOP']).dt.days
        
        # Define adherence (refill within 7 days of previous supply ending)
        group['ADHERENT'] = group['DAYS_SUPPLY_GAP'].fillna(0) <= 7
        
        # Only include in analysis if there are multiple refills
        if len(group) > 1:
            adherence_data.append({
                'PATIENT': patient,
                'MEDICATION': med,
                'ADHERENCE_RATE': group['ADHERENT'].mean(),
                'AVG_GAP': group['DAYS_SUPPLY_GAP'].mean(),
                'MAX_GAP': group['DAYS_SUPPLY_GAP'].max(),
                'REFILL_COUNT': len(group)
            })
    
    # Convert to DataFrame
    if adherence_data:
        adherence_metrics = pd.DataFrame(adherence_data)
        
        # Aggregate to patient level
        patient_adherence = adherence_metrics.groupby('PATIENT').agg({
            'ADHERENCE_RATE': 'mean',
            'AVG_GAP': 'mean',
            'MAX_GAP': 'max',
            'REFILL_COUNT': 'sum',
            'MEDICATION': 'nunique'
        }).reset_index()
        
        patient_adherence.rename(columns={'MEDICATION': 'MEDICATION_COUNT'}, inplace=True)
        
        # Merge with patient demographics
        final_data = pd.merge(
            patient_adherence,
            chronic_patients[['PATIENT', 'BIRTHDATE', 'GENDER', 'INCOME', 'CONDITIONS', 'CHRONIC_CONDITIONS']],
            on='PATIENT',
            how='left'
        )
        
        # Calculate age
        current_date = datetime.now()
        final_data['BIRTHDATE'] = pd.to_datetime(final_data['BIRTHDATE'], errors='coerce')
        final_data['AGE'] = (current_date - final_data['BIRTHDATE']).dt.days // 365
        
        # Categorize income
        final_data['INCOME_CATEGORY'] = pd.cut(
            final_data['INCOME'],
            bins=[0, 30000, 60000, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"Generated adherence data for {len(final_data)} patients")
        
        # Save to file
        print(f"Saving preprocessed data to {output_file}")
        final_data.to_csv(output_file, index=False)
        
        return final_data
    else:
        print("No adherence data could be calculated. Check if there are patients with multiple medication refills.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    data_dir = "synthea_data/"  # Current directory where the CSV files are located
    output_dir = "processed_data/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess data
    adherence_data = preprocess_adherence_data(
        f"{data_dir}patients.csv",
        f"{data_dir}medications.csv",
        f"{data_dir}encounters.csv",
        f"{output_dir}adherence_features.csv"
    )
    
    print("Data preprocessing complete!")
    if not adherence_data.empty:
        print(f"Processed {len(adherence_data)} patient records")
        print("\nSample of processed data:")
        print(adherence_data.head())
