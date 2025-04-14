import pandas as pd
import boto3
import json
from botocore.config import Config
from datetime import datetime

def load_adherence_data(file_path):
    """
    Load and preprocess the adherence data.
    """
    df = pd.read_csv(file_path)
    df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
    return df

def format_patient_data(patient):
    """
    Format patient data for the prompt.
    """
    return f"""
        Patient ID: {patient['PATIENT']}
        Age: {patient['AGE']}
        Gender: {patient['GENDER']}
        Income Category: {patient['INCOME_CATEGORY']}
        Conditions: {patient['CONDITIONS']}
        Adherence Rate: {patient['ADHERENCE_RATE']:.2f}
        Max Gap Between Refills: {patient['MAX_GAP']} days
        Average Gap Between Refills: {patient['AVG_GAP']:.1f} days
        Number of Medications: {patient['MEDICATION_COUNT']}
        Chronic Conditions: {patient['CHRONIC_CONDITIONS']}
        Refill Count: {patient['REFILL_COUNT']}
        """

def analyze_adherence_with_claude(patient_data, prompt_type="zero-shot", region="us-east-1"):
    """
    Analyze patient medication adherence data using Claude 3.7 on AWS Bedrock.
    """
    valid_prompt_types = ["zero-shot", "few-shot", "chain-of-thought", "tree-of-thought"]
    if prompt_type not in valid_prompt_types:
        raise ValueError(f"Invalid prompt_type. Must be one of {valid_prompt_types}")
    
    required_fields = ['PATIENT', 'AGE', 'GENDER', 'INCOME_CATEGORY', 'CONDITIONS', 
                      'ADHERENCE_RATE', 'MAX_GAP', 'AVG_GAP', 'MEDICATION_COUNT']
    missing_fields = [field for field in required_fields if field not in patient_data]
    if missing_fields:
        raise ValueError(f"Missing required patient data fields: {missing_fields}")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=60, read_timeout=300)
    )
    
    # model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    model_id = "arn:aws:bedrock:us-east-1:<aws account id>:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    patient_str = format_patient_data(patient_data)
    
    prompts = {
        "zero-shot": f"""Based on the following patient data, 
        identify factors that might predict medication non-adherence and suggest potential interventions:

            {patient_str}""",
        
        "few-shot": f"""Here are two examples of patient medication adherence patterns:

            Example 1:
            Patient ID: SYN12345
            Age: 72
            Gender: Male
            Income Category: Medium
            Conditions: Hypertension, Heart Failure
            Adherence Rate: 0.45
            Max Gap Between Refills: 35 days
            Average Gap Between Refills: 12 days
            Number of Medications: 3
            Analysis: This patient shows poor medication adherence likely due to complex medication regimen (multiple medications) 
            and possible financial constraints. Recommended interventions include medication simplification, pill organizers, and 
            financial assistance programs.

            Example 2:
            Patient ID: SYN23456
            Age: 58
            Gender: Female
            Income Category: High
            Conditions: Type 2 Diabetes
            Adherence Rate: 0.92
            Max Gap Between Refills: 8 days
            Average Gap Between Refills: 3 days
            Number of Medications: 1
            Analysis: This patient shows excellent medication adherence with minimal gaps between refills. 
            Contributing factors likely include simple medication regimen and higher income allowing for consistent access.

            Now analyze this new patient:
            {patient_str}""",
        
        "chain-of-thought": f"""Let's analyze medication adherence factors for this patient by following these steps:

            1. Identify the patient's demographic and clinical characteristics
            2. Analyze their current adherence metrics
            3. Compare these metrics to typical adherence patterns
            4. Identify potential barriers to adherence
            5. Suggest targeted interventions based on the identified barriers

            Here's the patient data:
            {patient_str}""",
        
        "tree-of-thought": f"""Let's analyze medication adherence factors for this patient by exploring multiple perspectives:

            Path 1: Analyze demographic factors (age, gender, income) and their impact on adherence
            Path 2: Examine clinical factors (conditions, medication complexity)
            Path 3: Investigate behavioral patterns (refill gaps, consistency)
            Path 4: Consider social determinants of health (income, potential access issues)

            For each path, identify potential barriers to adherence, then combine insights to develop comprehensive intervention recommendations.

            Here's the patient data:
            {patient_str}"""
                }
    
    prompt = prompts[prompt_type]
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "temperature": 0.7
    }
    
    try:
        response = bedrock_runtime.invoke_model(modelId=model_id, body=json.dumps(request_body))
        response_body = json.loads(response.get('body').read())
        analysis = response_body['content'][0]['text']

        if not analysis.strip():
            raise ValueError("Empty response from Claude")

        return analysis

    except json.JSONDecodeError:
        print("Error: Invalid JSON response from Claude")
        return "Error: Invalid JSON response from Claude"

    except Exception as e:
        print(f"Error invoking Claude 3.7: {e}")
        return f"Error: {str(e)}"

def create_reasoning_prompt(patient_data, reasoning_type, examples=None):
    """
    Creates specialized prompts combining reasoning with in-context learning
    """
    patient_str = format_patient_data(patient_data)
    
    reasoning_templates = {
        "prediction": f"""
                        Using the following steps, predict this patient's future adherence behavior:
                        1. Analyze historical adherence patterns
                        2. Identify key risk factors
                        3. Compare with similar patient profiles
                        4. Generate probability of adherence decline

                        Previous examples of adherence patterns:
                        {examples if examples else ''}

                        Patient Data:
                        {patient_str}

                        Predict: What is the likelihood of adherence decline in the next 3 months?
                        Explain your reasoning step by step.
                        """,
                    "classification": f"""
            Classify this patient's adherence risk level using the following categories:
            - High Risk (adherence rate < 0.6)
            - Medium Risk (adherence rate 0.6-0.8)
            - Low Risk (adherence rate > 0.8)

            Consider these factors in your classification:
            1. Current adherence metrics
            2. Demographic risk factors
            3. Clinical complexity
            4. Social determinants

            Previous classification examples:
            {examples if examples else ''}

            Patient Data:
            {patient_str}

            Classify and explain your reasoning.
            """
    }
    
    return reasoning_templates.get(reasoning_type, "")

def analyze_with_reasoning(patient_data, reasoning_type="prediction", examples=None):
    """
    Enhanced analysis function that combines reasoning with in-context learning
    """
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 3, "mode": "standard"})
    )
    
    prompt = create_reasoning_prompt(patient_data, reasoning_type, examples)
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "temperature": 0.7
    }
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId="arn:aws:bedrock:us-east-1:<aws account id>:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            body=json.dumps(request_body)
        )
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except Exception as e:
        return f"Error in reasoning analysis: {str(e)}"

def main():
    file_path = "processed_data/adherence_features.csv"
    adherence_data = load_adherence_data(file_path)

        # Create example cases for in-context learning
    example_cases = """
        Example 1:
        Patient with 0.45 adherence rate, multiple chronic conditions
        Classification: High Risk
        Reasoning: Multiple conditions, low adherence history, complex medication regimen
        Outcome: Required intervention within 2 months

        Example 2:
        Patient with 0.85 adherence rate, single condition
        Classification: Low Risk
        Reasoning: Consistent refill history, simple medication regimen
        Outcome: Maintained high adherence for 6 months
        """
    
    # Sample a few patients for analysis
    sample_patients = adherence_data.sample(n=3)
    
    prompt_types = ["zero-shot", "few-shot", "chain-of-thought", "tree-of-thought"]
    
    for _, patient in sample_patients.iterrows():
        print(f"Analyzing patient {patient['PATIENT']}...")
        print("\nStandard Analysis:")
        for prompt_type in prompt_types:
            print(f"\nRunning {prompt_type} analysis...")
            analysis = analyze_adherence_with_claude(patient, prompt_type=prompt_type)
            print(analysis)
            print("="*50)

    # Enhanced reasoning-based prediction
    print("\nPredictive Analysis:")
    prediction = analyze_with_reasoning(
        patient, 
        reasoning_type="prediction",
        examples=example_cases
    )
    print(prediction)
    
    # Risk classification with reasoning
    print("\nRisk Classification:")
    classification = analyze_with_reasoning(
        patient,
        reasoning_type="classification",
        examples=example_cases
    )
    print(classification)
    print("="*50)

if __name__ == "__main__":
    main()
