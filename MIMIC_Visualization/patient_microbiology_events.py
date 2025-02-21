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
microevents = pd.read_csv('./data/MICROBIOLOGYEVENTS.csv')

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
microevents['chartdate'] = pd.to_datetime(microevents['chartdate'])

# Merge datasets
base_data = pd.merge(patients, admissions, on='subject_id', how='inner')
analysis_data = base_data.merge(microevents, on=['subject_id', 'hadm_id'], how='inner')

# Calculate age at admission
analysis_data['age_at_admission'] = analysis_data.apply(
    lambda row: calculate_age(row['admittime'], row['dob']), axis=1
)

# Filter for valid ages
analysis_data = analysis_data[
    (analysis_data['age_at_admission'] >= 1) & 
    (analysis_data['age_at_admission'] <= 120)
].copy()

# Calculate some useful metrics
analysis_data['test_to_admit_days'] = (analysis_data['chartdate'] - analysis_data['admittime']).dt.total_seconds() / (24 * 60 * 60)

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    # Main Title
    html.H1("Patient Microbiology Events Analysis", 
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
        
        # Specimen Type Filter
        html.Label('Specimen Type:', style={'font-weight': 'bold', 'margin-top': '20px'}),
        dcc.Dropdown(
            id='specimen-filter',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': x, 'value': x} for x in sorted(analysis_data['spec_type_desc'].unique()) if pd.notna(x)],
            value='All',
            style={'margin-bottom': '20px'}
        ),

        # Organism Filter
        # Organism Filter
        html.Label('Organism:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='organism-filter',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': str(x), 'value': str(x)} for x in 
                    analysis_data['org_name'].dropna().sort_values().unique()
                    if str(x).strip() != '' and str(x).lower() != 'nan'
                    ],
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
            dcc.Graph(id='specimen-timeline', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='organism-heatmap', style={'width': '50%', 'display': 'inline-block'})
        ]),
        # Second row of graphs
        html.Div([
            dcc.Graph(id='resistance-pattern', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='specimen-distribution', style={'width': '50%', 'display': 'inline-block'})
        ]),
        # Third row of graphs
        html.Div([
            dcc.Graph(id='age-specimen-relation', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='test-distribution', style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style={
        'width': '75%',
        'margin-left': '22%',
        'padding': '20px'
    })
])

@app.callback(
    [Output('specimen-timeline', 'figure'),
     Output('organism-heatmap', 'figure'),
     Output('resistance-pattern', 'figure'),
     Output('specimen-distribution', 'figure'),
     Output('age-specimen-relation', 'figure'),
     Output('test-distribution', 'figure')],
    [Input('age-filter', 'value'),
     Input('specimen-filter', 'value'),
     Input('organism-filter', 'value')]
)
def update_graphs(age_range, specimen_type, organism):
    # Filter data based on selections
    filtered_data = analysis_data.copy()
    
    if age_range:
        filtered_data = filtered_data[
            (filtered_data['age_at_admission'] >= age_range[0]) &
            (filtered_data['age_at_admission'] <= age_range[1])
        ]
    
    if specimen_type != 'All':
        filtered_data = filtered_data[filtered_data['spec_type_desc'] == specimen_type]
        
    if organism != 'All':
        filtered_data = filtered_data[filtered_data['org_name'] == organism]


    # 1. Specimen Timeline (Line Chart)
    timeline_data = filtered_data.groupby(pd.Grouper(key='chartdate', freq='W'))['spec_type_desc'].count().reset_index()
    fig1 = px.line(
        timeline_data,
        x='chartdate',
        y='spec_type_desc',
        title='Weekly Distribution of Microbiology Specimens',
        labels={
            'chartdate': 'Date',
            'spec_type_desc': 'Number of Specimens'
        }
    )
    fig1.update_traces(mode='lines+markers')  # Add markers to the line


    # 2. Organism Heatmap
    org_spec_pivot = pd.crosstab(filtered_data['org_name'], filtered_data['spec_type_desc'])
    fig2 = px.imshow(
        org_spec_pivot,
        title='Organism vs Specimen Type Heatmap',
        labels=dict(x='Specimen Type', y='Organism', color='Count'),
        aspect='auto'
    )

    # 3. Heatmap for resistance pattern
    resistance_pivot = pd.pivot_table(
        filtered_data,
        values='row_id',
        index='spec_type_desc',
        columns='interpretation',
        aggfunc='count',
        fill_value=0
    )

    fig3 = px.imshow(
        resistance_pivot,
        title='Resistance Pattern Heatmap',
        labels=dict(x='Interpretation', y='Specimen Type', color='Count'),
        aspect='auto'
    )


    # 4. Specimen Distribution (Treemap)
    specimen_counts = filtered_data['spec_type_desc'].value_counts()
    fig4 = px.treemap(
        names=specimen_counts.index,
        parents=['Specimens'] * len(specimen_counts),
        values=specimen_counts.values,
        title='Distribution of Specimen Types'
    )

    # 5. Age vs Specimen Types (Violin Plot)
    fig5 = px.violin(
        filtered_data,
        y='age_at_admission',
        x='spec_type_desc',
        title='Age Distribution by Specimen Type',
        labels={'age_at_admission': 'Age', 'spec_type_desc': 'Specimen Type'}
    )

    # Box Plot
    fig6 = px.box(
        filtered_data,
        x='org_name',
        y='test_to_admit_days',
        title='Distribution of Test Timing by Organism',
        labels={
            'org_name': 'Organism',
            'test_to_admit_days': 'Days from Admission to Test'
        }
    )
    fig6.update_xaxes(tickangle=45)

    # Update layout for all figures
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            template='plotly_white'
        )

    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run_server(debug=True)