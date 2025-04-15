import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from astros_analysis import load_data

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Load the data
df = load_data()

# Modern color palette
colors = {
    'background': '#1A1A1A',  # Dark background
    'card_background': '#2A2A2A',  # Slightly lighter dark for cards
    'text': '#FFFFFF',  # White text
    'primary': '#002D62',  # Astros Navy Blue
    'secondary': '#EB6E1F',  # Astros Orange
    'accent': '#FFFFFF',  # White accent
    'plot_background': '#2A2A2A',  # Dark plot background
    'grid': '#3A3A3A',  # Slightly lighter grid lines
    'dropdown_text': '#FFFFFF',  # White text for dropdowns
    'dropdown_bg': '#3A3A3A',  # Lighter background for dropdowns
    'radio_text': '#FFFFFF'  # White text for radio buttons
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Houston Astros Analytics Dashboard',
                style={'color': colors['text'], 'textAlign': 'center', 'padding': '20px', 'fontWeight': 'bold', 'fontSize': '2.5em'}),
        html.P('Interactive visualization of Houston Astros roster data and performance metrics',
               style={'color': colors['secondary'], 'textAlign': 'center', 'fontSize': '1.2em'})
    ]),
    
    # Main content
    html.Div([
        # First row
        html.Div([
            # WAR Analysis Card
            html.Div([
                html.H3('WAR Analysis', style={'color': colors['text'], 'fontSize': '1.8em', 'marginBottom': '15px'}),
                dcc.Graph(id='war-by-age'),
                dcc.RadioItems(
                    id='war-metric',
                    options=[
                        {'label': ' By Age', 'value': 'age'},
                        {'label': ' By Position', 'value': 'position'},
                        {'label': ' By Season', 'value': 'season'}
                    ],
                    value='age',
                    style={'margin': '10px', 'color': colors['radio_text']},
                    className='radio-items',
                    labelStyle={'display': 'block', 'margin': '10px 0', 'fontSize': '1.1em'}
                )
            ], className='card', style={'flex': 1, 'margin': '10px', 'padding': '20px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),
            
            # Career Longevity Card
            html.Div([
                html.H3('Career Longevity', style={'color': colors['text'], 'fontSize': '1.8em', 'marginBottom': '15px'}),
                dcc.Graph(id='career-analysis'),
                dcc.Dropdown(
                    id='career-metric',
                    options=[
                        {'label': 'Years Distribution', 'value': 'years'},
                        {'label': 'Retention Rate', 'value': 'retention'},
                        {'label': 'WAR Progression', 'value': 'war_prog'}
                    ],
                    value='years',
                    style={
                        'margin': '10px',
                        'color': colors['dropdown_text'],
                        'backgroundColor': colors['dropdown_bg'],
                        'fontSize': '1.1em'
                    }
                )
            ], className='card', style={'flex': 1, 'margin': '10px', 'padding': '20px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'})
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Second row
        html.Div([
            # Position Analysis Card
            html.Div([
                html.H3('Position Analysis', style={'color': colors['text'], 'fontSize': '1.8em', 'marginBottom': '15px'}),
                dcc.Graph(id='position-analysis'),
                dcc.Dropdown(
                    id='position-metric',
                    options=[
                        {'label': 'Position Distribution', 'value': 'distribution'},
                        {'label': 'Position WAR', 'value': 'war'},
                        {'label': 'Versatility', 'value': 'versatility'}
                    ],
                    value='distribution',
                    style={
                        'margin': '10px',
                        'color': colors['dropdown_text'],
                        'backgroundColor': colors['dropdown_bg'],
                        'fontSize': '1.1em'
                    }
                )
            ], className='card', style={'flex': 1, 'margin': '10px', 'padding': '20px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),
            
            # Player Demographics Card
            html.Div([
                html.H3('Player Demographics', style={'color': colors['text'], 'fontSize': '1.8em', 'marginBottom': '15px'}),
                dcc.Graph(id='demographics-analysis'),
                dcc.Dropdown(
                    id='demographics-metric',
                    options=[
                        {'label': 'Age Distribution', 'value': 'age'},
                        {'label': 'Origin', 'value': 'origin'},
                        {'label': 'Physical Attributes', 'value': 'physical'}
                    ],
                    value='age',
                    style={
                        'margin': '10px',
                        'color': colors['dropdown_text'],
                        'backgroundColor': colors['dropdown_bg'],
                        'fontSize': '1.1em'
                    }
                )
            ], className='card', style={'flex': 1, 'margin': '10px', 'padding': '20px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'})
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ], style={'padding': '20px'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})

# Callbacks
@app.callback(
    Output('war-by-age', 'figure'),
    Input('war-metric', 'value')
)
def update_war_graph(metric):
    if metric == 'age':
        fig = px.scatter(df, x='Age', y='WAR', 
                        trendline="lowess",
                        title='WAR vs Age',
                        color='All-Star',
                        color_discrete_sequence=[colors['secondary'], colors['primary']],
                        hover_data=['Name', 'Season'])
    elif metric == 'position':
        position_war = []
        position_labels = []
        for pos in ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']:
            pos_data = df[df[pos] > 0]['WAR']
            if len(pos_data) > 0:
                position_war.append(pos_data)
                position_labels.append(pos)
        
        fig = go.Figure()
        for i in range(len(position_war)):
            fig.add_trace(go.Box(y=position_war[i], name=position_labels[i],
                                line_color=colors['secondary'],
                                fillcolor=colors['primary']))
        fig.update_layout(title='WAR Distribution by Position')
    else:  # season
        season_war = df.groupby('Season')['WAR'].mean().reset_index()
        fig = px.line(season_war, x='Season', y='WAR',
                      title='Average WAR by Season',
                      color_discrete_sequence=[colors['secondary']])
    
    fig.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5,
        xaxis=dict(gridcolor=colors['grid'], showgrid=True),
        yaxis=dict(gridcolor=colors['grid'], showgrid=True)
    )
    return fig

@app.callback(
    Output('career-analysis', 'figure'),
    Input('career-metric', 'value')
)
def update_career_graph(metric):
    if metric == 'years':
        fig = px.histogram(df, x='Yrs',
                          title='Distribution of Career Lengths',
                          nbins=int(df['Yrs'].max()),
                          color_discrete_sequence=[colors['secondary']])
    elif metric == 'retention':
        max_years = int(df['Yrs'].max())
        retention = [(df['Yrs'] >= yr).mean() for yr in range(1, max_years + 1)]
        # Create a DataFrame for the line plot
        retention_df = pd.DataFrame({
            'Years': range(1, max_years + 1),
            'Retention': retention
        })
        fig = px.line(retention_df, x='Years', y='Retention',
                      title='Player Retention Rate',
                      labels={'Years': 'Years in League', 'Retention': 'Proportion of Players'},
                      color_discrete_sequence=[colors['secondary']])
    else:  # war_prog
        age_war = df.groupby('Age')['WAR'].mean().reset_index()
        fig = px.line(age_war, x='Age', y='WAR',
                      title='WAR Progression by Age',
                      color_discrete_sequence=[colors['secondary']])
    
    fig.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5,
        xaxis=dict(gridcolor=colors['grid'], showgrid=True),
        yaxis=dict(gridcolor=colors['grid'], showgrid=True)
    )
    return fig

@app.callback(
    Output('position-analysis', 'figure'),
    Input('position-metric', 'value')
)
def update_position_graph(metric):
    if metric == 'distribution':
        fig = px.histogram(df, x='positions_played',
                          title='Distribution of Positions Played',
                          nbins=int(df['positions_played'].max()),
                          color_discrete_sequence=[colors['secondary']])
    elif metric == 'war':
        position_cols = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        position_war = pd.DataFrame([
            {'Position': pos, 'WAR': df[df[pos] > 0]['WAR'].mean()}
            for pos in position_cols
        ])
        fig = px.bar(position_war, x='Position', y='WAR',
                     title='Average WAR by Position',
                     color_discrete_sequence=[colors['secondary']])
    else:  # versatility
        position_cols = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        position_counts = df[position_cols].apply(lambda x: x > 0).sum()
        fig = px.bar(x=position_counts.index, y=position_counts.values,
                     title='Number of Players by Position',
                     color_discrete_sequence=[colors['secondary']])
    
    fig.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5,
        xaxis=dict(gridcolor=colors['grid'], showgrid=True),
        yaxis=dict(gridcolor=colors['grid'], showgrid=True)
    )
    return fig

@app.callback(
    Output('demographics-analysis', 'figure'),
    Input('demographics-metric', 'value')
)
def update_demographics_graph(metric):
    if metric == 'age':
        fig = px.histogram(df, x='Age',
                          title='Age Distribution',
                          nbins=20,
                          color_discrete_sequence=[colors['secondary']])
    elif metric == 'origin':
        country_counts = df['Born'].value_counts().head(10)
        fig = px.bar(x=country_counts.index, y=country_counts.values,
                     title='Top 10 Player Origins',
                     color_discrete_sequence=[colors['secondary']])
    else:  # physical
        fig = px.scatter(df, x='Height_inches', y='Wt',
                         title='Height vs Weight Distribution',
                         color='positions_played',
                         color_discrete_sequence=[colors['secondary'], colors['primary']],
                         hover_data=['Name', 'Position'])
    
    fig.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5,
        xaxis=dict(gridcolor=colors['grid'], showgrid=True),
        yaxis=dict(gridcolor=colors['grid'], showgrid=True)
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 