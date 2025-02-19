import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data = pd.read_csv('census_tracts.csv')


data['median_income'] = data['median_income'].apply(lambda x: int(x) if x >= 0 else None)
data['median_home_value'] = data['median_home_value'].apply(lambda x: int(x) if x >= 0 else None)
data = data.dropna(subset=['median_income', 'median_home_value'])


np.random.seed(42)
data['Percent_Change_Income'] = np.random.normal(0, 1, data.shape[0])
data['Percent_Change_Home_Value'] = np.random.normal(0, 1, data.shape[0])
data['Percent_Change_Education'] = np.random.normal(0, 1, data.shape[0])


features = ['Percent_Change_Income', 'Percent_Change_Home_Value', 'Percent_Change_Education']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


data['Gentrification_Index'] = data[features].mean(axis=1)

def classify_gentrification(index):
    if index > 0.5:
        return 'High Gentrification'
    elif 0 < index <= 0.5:
        return 'Moderate Gentrification'
    elif -0.5 < index <= 0:
        return 'Low Gentrification'
    else:
        return 'No Gentrification'

data['Gentrification_Level'] = data['Gentrification_Index'].apply(classify_gentrification)


if 'latitude' not in data.columns or 'longitude' not in data.columns:
    data['latitude'] = np.random.uniform(38.5, 39.5, size=data.shape[0])
    data['longitude'] = np.random.uniform(-77.5, -76.5, size=data.shape[0])

center_lat = data['latitude'].mean()
center_lon = data['longitude'].mean()


def create_map_fig(zoom_level):
    map_fig = px.scatter_map(
        data,
        lat='latitude',
        lon='longitude',
        color='Gentrification_Level',
        hover_name='name',
        hover_data={
            'median_income': True,
            'median_home_value': True,
            'Gentrification_Index': True,
            'Cluster': True
        },
        zoom=zoom_level,
        center={"lat": center_lat, "lon": center_lon},
        height=600,
        title='Geographic Distribution of Census Tracts & Gentrification Levels'
    )
    map_fig.update_layout(mapbox_style="open-street-map")
    map_fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
    return map_fig

initial_zoom = 10
map_fig = create_map_fig(initial_zoom)

cluster_fig = px.scatter(
    data,
    x='median_income',
    y='median_home_value',
    color='Cluster',
    hover_name='name',
    title='Median Income vs. Home Value by Cluster'
)

gentrification_hist = px.histogram(
    data,
    x='Gentrification_Index',
    color='Gentrification_Level',
    nbins=50,
    title='Distribution of Gentrification Index'
)


navbar = dbc.NavbarSimple(
    brand="Gentrification Analysis Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)


footer = dbc.Container(
    html.Footer(
        html.P("© 2025 Your Organization | Data Science & Urban Studies", className="text-center mb-0"),
        className="py-3"
    ),
    fluid=True,
    className="bg-light mt-4"
)


def create_data_table(df):
    header = [html.Thead(html.Tr([html.Th(col) for col in df.columns]))]
    body = [html.Tbody([
        html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
        for i in range(len(df))
    ])]
    return dbc.Table(header + body, bordered=True, hover=True, responsive=True, striped=True)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    navbar,
    
    dbc.Row(
        dbc.Col(html.H2("Key Insights", className="text-center my-4"))
    ),
    
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Total Census Tracts", className="card-title text-center"),
                    html.H3(f"{data.shape[0]:,}", className="card-text text-center")
                ]),
                className="mb-4", color="info", inverse=True
            ),
            md=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Average Median Income", className="card-title text-center"),
                    html.H3(f"${data['median_income'].mean():,.0f}", className="card-text text-center")
                ]),
                className="mb-4", color="success", inverse=True
            ),
            md=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Average Home Value", className="card-title text-center"),
                    html.H3(f"${data['median_home_value'].mean():,.0f}", className="card-text text-center")
                ]),
                className="mb-4", color="warning", inverse=True
            ),
            md=4
        ),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='map-graph', figure=map_fig), md=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.Label("Map Zoom Level:"), md=2, className="d-flex align-items-center"),
        dbc.Col(
            dcc.Slider(
                id='zoom-slider',
                min=5, max=15, step=0.5, value=initial_zoom,
                marks={i: str(i) for i in range(5, 16)}
            ),
            md=10
        )
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='cluster-graph', figure=cluster_fig), md=6),
        dbc.Col(dcc.Graph(id='gentrification-hist', figure=gentrification_hist), md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Cluster:", className="h5 text-center mb-2"),
            dcc.Dropdown(
                id='cluster-dropdown',
                options=[{'label': f'Cluster {i}', 'value': i} for i in sorted(data['Cluster'].unique())],
                value=sorted(data['Cluster'].unique()),
                multi=True,
                placeholder="Select clusters..."
            )
        ], md=6, className="mx-auto")
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.H4("Detailed Statistics", className="text-center mb-3")),
        dbc.Col(html.Div(id='stats-panel'), md=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.H4("Sample Data Table", className="text-center mb-3")),
        dbc.Col(id='data-table', md=12)
    ], className="mb-4"),
    
    footer
], fluid=True)

@app.callback(
    [Output('map-graph', 'figure'),
     Output('cluster-graph', 'figure'),
     Output('gentrification-hist', 'figure'),
     Output('stats-panel', 'children'),
     Output('data-table', 'children')],
    [Input('cluster-dropdown', 'value'),
     Input('zoom-slider', 'value')]
)
def update_dashboard(selected_clusters, zoom_value):
 
    filtered = data[data['Cluster'].isin(selected_clusters)]
    
    
    updated_map = px.scatter_map(
        filtered,
        lat='latitude',
        lon='longitude',
        color='Gentrification_Level',
        hover_name='name',
        hover_data={
            'median_income': True,
            'median_home_value': True,
            'Gentrification_Index': True,
            'Cluster': True
        },
        zoom=zoom_value,
        center={"lat": center_lat, "lon": center_lon},
        height=600,
        title='Geographic Distribution of Census Tracts & Gentrification Levels'
    )
    updated_map.update_layout(mapbox_style="open-street-map")
    updated_map.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
    
    
    updated_cluster = px.scatter(
        filtered,
        x='median_income',
        y='median_home_value',
        color='Cluster',
        hover_name='name',
        title='Median Income vs. Home Value by Cluster'
    )
    
    
    updated_hist = px.histogram(
        filtered,
        x='Gentrification_Index',
        color='Gentrification_Level',
        nbins=50,
        title='Distribution of Gentrification Index'
    )
    
    
    stats_html = html.Div([
        html.P(f"Number of Tracts: {filtered.shape[0]}", className="lead text-center"),
        html.P(f"Mean Median Income: ${filtered['median_income'].mean():,.0f}", className="lead text-center"),
        html.P(f"Mean Median Home Value: ${filtered['median_home_value'].mean():,.0f}", className="lead text-center"),
        html.P(f"Mean Gentrification Index: {filtered['Gentrification_Index'].mean():.2f}", className="lead text-center")
    ], className="p-3 border rounded bg-light")
    
    
    table_cols = ['geoid', 'name', 'median_income', 'median_home_value', 'educational_attainment',
                  'Gentrification_Index', 'Gentrification_Level', 'Cluster']
    table_header = html.Thead(html.Tr([html.Th(col, className="text-center") for col in table_cols]))
    table_body = html.Tbody([
        html.Tr([html.Td(filtered.iloc[i][col], className="text-center") for col in table_cols])
        for i in range(min(10, filtered.shape[0]))
    ])
    table_component = dbc.Table([table_header, table_body], bordered=True, striped=True, hover=True, responsive=True)
    
    return updated_map, updated_cluster, updated_hist, stats_html, table_component


if __name__ == '__main__':
    app.run_server(debug=True)
