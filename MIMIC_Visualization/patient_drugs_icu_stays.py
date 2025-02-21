import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash

# Load the datasets
patients = pd.read_csv('./data/PATIENTS.csv', dtype={'dob': str})
admissions = pd.read_csv('./data/ADMISSIONS.csv', dtype={'admittime': str})
prescriptions = pd.read_csv('./data/PRESCRIPTIONS.csv')
icustays = pd.read_csv('./data/ICUSTAYS.csv')

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

# Convert dates
patients['dob'] = patients['dob'].apply(convert_date_format)
admissions['admittime'] = admissions['admittime'].apply(convert_date_format)
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

# First merge patients and admissions
base_data = pd.merge(patients, admissions, on='subject_id', how='inner')

# Calculate age at admission
base_data['age_at_admission'] = base_data.apply(
    lambda row: calculate_age(row['admittime'], row['dob']), axis=1
)

# Filter for valid ages between 1 and 120
base_data = base_data[
    (base_data['age_at_admission'] >= 1) & 
    (base_data['age_at_admission'] <= 120)
].copy()

# Merge with prescriptions and ICU stays
analysis_data = base_data.merge(
    prescriptions[['subject_id', 'hadm_id', 'drug', 'drug_type', 'formulary_drug_cd']], 
    on=['subject_id', 'hadm_id'], 
    how='inner'
)

analysis_data = analysis_data.merge(
    icustays[['subject_id', 'hadm_id', 'first_careunit', 'last_careunit', 'intime', 'outtime']], 
    on=['subject_id', 'hadm_id'], 
    how='inner'
)

analysis_data['icu_length_of_stay'] = icustays['los']

# Get medication counts per patient
med_counts = analysis_data.groupby('subject_id')['drug'].nunique().reset_index(name='unique_drugs')
analysis_data = analysis_data.merge(med_counts, on='subject_id')

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    # Main Title
    html.H1("Patient Drugs & ICU Stays Analysis", 
            style={
                'text-align': 'center',
                'margin-bottom': '20px',
                'padding': '20px',
                'background-color': '#f8f9fa',
                'color': '#2c3e50',
                'font-family': 'Arial, sans-serif'
            }),
    
    # Filters Container
    html.Div([
        html.H3("Filters", style={'margin-bottom': '20px'}),
        
        # Age Range Filter
        html.Label('Age Range:', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
        dcc.RangeSlider(
            id='age-filter',
            min=int(analysis_data['age_at_admission'].min()),
            max=int(analysis_data['age_at_admission'].max()),
            value=[1, 120],
            marks={i: str(i) for i in range(0, 121, 20)},
            step=1
        ),
        
        # ICU Type Filter
        html.Label('ICU Type:', style={'font-weight': 'bold', 'margin-top': '20px'}),
        dcc.Dropdown(
            id='icu-filter',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': x, 'value': x} for x in sorted(analysis_data['first_careunit'].unique())],
            value='All',
            style={'margin-bottom': '20px'}
        ),
        
        # Drug Type Filter
        html.Label('Drug Type:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='drug-type-filter',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': x, 'value': x} for x in sorted(analysis_data['drug_type'].unique())],
            value='All',
            style={'margin-bottom': '20px'}
        )
    ], style={
        'width': '20%',
        'background-color': '#f8f9fa',
        'padding': '20px',
        'border-radius': '5px',
        'box-shadow': '2px 2px 2px lightgrey',
        'margin-right': '20px',
        'position': 'fixed',
        'left': '0',
        'height': 'calc(100vh - 150px)',
        'overflow-y': 'auto'
    }),
    
    # Graphs Container
    html.Div([
        # First row of graphs
        html.Div([
            dcc.Graph(id='drug-usage-chart', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='icu-los-hist', style={'width': '50%', 'display': 'inline-block'})
        ]),
        # Second row of graphs
        html.Div([
            dcc.Graph(id='drug-type-pie', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='icu-drug-correlation', style={'width': '50%', 'display': 'inline-block'})
        ]),
        # Third row of graphs
        html.Div([
            dcc.Graph(id='age-drug-scatter', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='icu-type-distribution', style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style={
        'width': '75%',
        'margin-left': '22%',
        'padding': '20px'
    })
])

@app.callback(
    [Output('drug-usage-chart', 'figure'),
     Output('icu-los-hist', 'figure'),
     Output('drug-type-pie', 'figure'),
     Output('icu-drug-correlation', 'figure'),
     Output('age-drug-scatter', 'figure'),
     Output('icu-type-distribution', 'figure')],
    [Input('age-filter', 'value'),
     Input('icu-filter', 'value'),
     Input('drug-type-filter', 'value')]
)
def update_graphs(age_range, icu_filter, drug_type_filter):
    # Filter data based on selections
    filtered_data = analysis_data.copy()
    
    # Apply filters
    if age_range:
        filtered_data = filtered_data[
            (filtered_data['age_at_admission'] >= age_range[0]) &
            (filtered_data['age_at_admission'] <= age_range[1])
        ]
    
    if icu_filter != 'All':
        filtered_data = filtered_data[filtered_data['first_careunit'] == icu_filter]
        
    if drug_type_filter != 'All':
        filtered_data = filtered_data[filtered_data['drug_type'] == drug_type_filter]

    # 1. Top 10 Most Common Drugs
    drug_counts = filtered_data['drug'].value_counts().head(10)
    fig1 = px.bar(
        x=drug_counts.index,
        y=drug_counts.values,
        title='Top 10 Most Prescribed Medications',
        labels={'x': 'Medication', 'y': 'Number of Prescriptions'}
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # 2. ICU Length of Stay Distribution
    fig2 = px.histogram(
        filtered_data,
        x='icu_length_of_stay',
        title='ICU Length of Stay Distribution',
        labels={'icu_length_of_stay': 'Length of Stay (days)'},
        nbins=50
    )

    # 3. Drug Type Distribution
    drug_type_counts = filtered_data['drug_type'].value_counts()
    fig3 = px.pie(
        values=drug_type_counts.values,
        names=drug_type_counts.index,
        title='Distribution of Drug Types',
        hole=0.4
    )

    # 4. ICU Stay vs Number of Medications
    fig4 = px.scatter(
        filtered_data,
        x='icu_length_of_stay',
        y='unique_drugs',
        title='ICU Length of Stay vs Number of Medications',
        labels={
            'icu_length_of_stay': 'ICU Length of Stay (days)',
            'unique_drugs': 'Number of Different Medications'
        }
    )

    # 5. Age vs Number of Medications
    fig5 = px.scatter(
        filtered_data,
        x='age_at_admission',
        y='unique_drugs',
        title='Age vs Number of Medications',
        labels={
            'age_at_admission': 'Age',
            'unique_drugs': 'Number of Different Medications'
        }
    )

    # 6. ICU Type Distribution
    icu_counts = filtered_data['first_careunit'].value_counts()
    fig6 = px.pie(
        values=icu_counts.values,
        names=icu_counts.index,
        title='Distribution of ICU Types',
        hole=0.4
    )

    # Update layout for all figures
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            template='plotly_white'
        )

    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run_server(debug=True)
