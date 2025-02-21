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

# Merge datasets
merged_data = pd.merge(patients, admissions, on='subject_id', how='inner')

# Calculate age at admission
merged_data['age_at_admission'] = merged_data.apply(
    lambda row: calculate_age(row['admittime'], row['dob']), axis=1
)

# Filter for valid ages between 1 and 120
merged_data = merged_data[
    (merged_data['age_at_admission'] >= 1) & 
    (merged_data['age_at_admission'] <= 120)
].copy()

# Calculate length of stay
merged_data['length_of_stay'] = (merged_data['dischtime'] - merged_data['admittime']).dt.total_seconds() / (24 * 60 * 60)

# Create the dashboard with filters
def create_filtered_data(data, diagnosis_filter='All', age_range=None, insurance_filter='All'):
    filtered_data = data.copy()
    
    # Apply diagnosis filter
    if diagnosis_filter != 'All':
        filtered_data = filtered_data[filtered_data['diagnosis'] == diagnosis_filter]
    
    # Apply age range filter
    if age_range is not None:
        filtered_data = filtered_data[
            (filtered_data['age_at_admission'] >= age_range[0]) &
            (filtered_data['age_at_admission'] <= age_range[1])
        ]
    
    # Apply insurance filter
    if insurance_filter != 'All':
        filtered_data = filtered_data[filtered_data['insurance'] == insurance_filter]
    
    return filtered_data

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Patient Admission Analysis Dashboard", 
            style={
                'text-align': 'center',
                'margin-bottom': '20px',
                'padding': '20px',
                'background-color': '#f8f9fa'
            }),
    
    # Main container
    html.Div([
        # Left panel for filters
        html.Div([
            html.H3("Filters", style={'margin-bottom': '20px'}),
            
            html.Div([
                # Diagnosis Filter
                html.Label('Diagnosis:', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='diagnosis-filter',
                    options=[{'label': 'All', 'value': 'All'}] +
                            [{'label': x, 'value': x} 
                             for x in sorted(merged_data['diagnosis'].unique())
                             if pd.notna(x)],
                    value='All',
                    style={
                        'margin-bottom': '20px',
                        'width': '100%'
                    },
                    searchable=True,
                    placeholder="Select a diagnosis..."
                ),
                
                # Insurance Filter
                html.Label('Insurance:', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='insurance-filter',
                    options=[{'label': 'All', 'value': 'All'}] +
                            [{'label': x, 'value': x} 
                             for x in sorted(merged_data['insurance'].unique())
                             if pd.notna(x)],
                    value='All',
                    style={
                        'margin-bottom': '20px',
                        'width': '100%'
                    }
                ),
                
                # Age Range Filter
                html.Label('Age Range:', style={'font-weight': 'bold', 'margin-bottom': '10px'}),
                dcc.RangeSlider(
                    id='age-filter',
                    min=int(merged_data['age_at_admission'].min()),
                    max=int(merged_data['age_at_admission'].max()),
                    value=[1, 120],
                    marks={i: str(i) for i in range(0, 121, 20)},
                    step=1
                ),
            ], style={'padding': '20px'})
        ], style={
            'width': '20%',
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-radius': '5px',
            'box-shadow': '2px 2px 2px lightgrey',
            'margin-right': '20px',
            'height': 'calc(100vh - 100px)',
            'position': 'fixed',
            'left': '0',
            'top': '100px',
            'overflow-y': 'auto'
        }),
        
        # Right panel for graphs
        html.Div([
            html.Div([
                dcc.Graph(id='gender-pie', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='age-hist', style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div([
                dcc.Graph(id='admission-bar', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='los-box', style={'width': '50%', 'display': 'inline-block'}),
            ]),
                html.Div([  # Add this new div for the additional graphs
                dcc.Graph(id='diagnosis-donut', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='age-butterfly', style={'width': '50%', 'display': 'inline-block'}),
            ])
        ], style={
            'width': '75%',
            'margin-left': '22%',
            'padding': '20px'
        })
    ], style={
        'display': 'flex',
        'margin-top': '20px'
    })
])

@app.callback(
    [Output('gender-pie', 'figure'),
     Output('age-hist', 'figure'),
     Output('admission-bar', 'figure'),
     Output('los-box', 'figure'),
     Output('diagnosis-donut', 'figure'),
     Output('age-butterfly', 'figure')],
    [Input('diagnosis-filter', 'value'),
     Input('age-filter', 'value'),
     Input('insurance-filter', 'value')]
)
def update_graphs(diagnosis_filter, age_range, insurance_filter):
    filtered_data = create_filtered_data(merged_data, diagnosis_filter, age_range, insurance_filter)
    
    # 1. Gender Distribution (Pie Chart)
    gender_counts = filtered_data['gender'].value_counts()
    
    fig1 = go.Figure(data=[go.Pie(
        labels=gender_counts.index,
        values=gender_counts.values,
        textinfo='label+percent',
        marker_colors=px.colors.qualitative.Set3,
        hovertemplate="Gender: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig1.update_layout(
        title={
            'text': 'Gender Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 2. Age Distribution at Admission (Bar Chart)
    fig2 = go.Figure()
    
    fig2.add_trace(go.Histogram(
        x=filtered_data['age_at_admission'],
        nbinsx=30,
        name='Age Distribution',
        marker_color='lightblue',
        hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"
    ))
    
    fig2.update_layout(
        title={
            'text': 'Age Distribution at Admission',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Age',
        yaxis_title='Number of Patients',
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    
    # 3. Admission Locations (Donut Chart)
    location_counts = filtered_data['admission_location'].value_counts()
    
    fig3 = go.Figure(data=[go.Pie(
        labels=location_counts.index,
        values=location_counts.values,
        hole=0.6,  # This creates the donut hole
        textposition='outside',
        textinfo='label+percent',
        marker_colors=px.colors.qualitative.Set3,
        hovertemplate="Location: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig3.update_layout(
        title={
            'text': 'Admission Locations Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        annotations=[{
            'text': f'Total: {sum(location_counts)}',
            'x': 0.5,
            'y': 0.5,
            'font_size': 12,
            'showarrow': False
        }],
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 4. Length of Stay (Scatter Plot)
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=filtered_data['age_at_admission'],
        y=filtered_data['length_of_stay'],
        mode='markers',
        marker=dict(
            size=8,
            color='coral',
            opacity=0.6
        ),
        hovertemplate="Age: %{x}<br>Length of Stay: %{y:.1f} days<extra></extra>"
    ))
    
    fig4.update_layout(
        title={
            'text': 'Length of Stay vs Age',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Age at Admission',
        yaxis_title='Length of Stay (Days)',
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    
        # 5. Diagnosis Distribution (Donut Chart)
    diagnosis_counts = filtered_data['diagnosis'].value_counts().head(10)
    
    fig5 = go.Figure(data=[go.Pie(
        labels=diagnosis_counts.index,
        values=diagnosis_counts.values,
        hole=0.6,
        textposition='outside',
        textinfo='label+percent',
        marker_colors=px.colors.qualitative.Set3,
        hovertemplate="Diagnosis: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig5.update_layout(
        title={
            'text': 'Top 10 Diagnoses Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=100, b=50),
        annotations=[{
            'text': f'Total: {sum(diagnosis_counts)}',
            'x': 0.5,
            'y': 0.5,
            'font_size': 12,
            'showarrow': False
        }]
    )

    # 6. Age Distribution (Butterfly Chart)
    age_bins = list(range(0, 121, 5))
    admission_hist = np.histogram(filtered_data['age_at_admission'], bins=age_bins)
    death_hist = np.histogram(filtered_data[filtered_data['deathtime'].notna()]['age_at_admission'], bins=age_bins)
    
    bin_labels = [f'{age_bins[i]}-{age_bins[i+1]}' for i in range(len(age_bins)-1)]
    
    fig6 = go.Figure()
    
    # Add admission age bars (left side)
    fig6.add_trace(go.Bar(
        y=bin_labels,
        x=-admission_hist[0],
        name='Age at Admission',
        orientation='h',
        marker_color='lightblue',
        hovertemplate='Count: %{x}<br>Age Group: %{y}<extra></extra>'
    ))
    
    # Add death age bars (right side)
    fig6.add_trace(go.Bar(
        y=bin_labels,
        x=death_hist[0],
        name='Age at Death',
        orientation='h',
        marker_color='salmon',
        hovertemplate='Count: %{x}<br>Age Group: %{y}<extra></extra>'
    ))
    
    fig6.update_layout(
        title={
            'text': 'Age Distribution: Admission vs Death',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        barmode='overlay',
        bargap=0.1,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        shapes=[dict(
            type='line',
            x0=0, x1=0,
            y0=-0.5, y1=len(bin_labels)-0.5,
            line=dict(color='black', width=1)
        )],
        xaxis=dict(
            title='Count',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
        ),
        yaxis=dict(
            title='Age Groups',
            zeroline=False,
            categoryorder='category ascending'
        )
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


if __name__ == '__main__':
    app.run_server(debug=True)
