import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Load the transformed data
df = pd.read_csv('flow_transformed.csv')

# Get list of portfolios
portfolios = df['PORT_ID'].unique()

# Step 1: Create a nested dictionary to organize data by portfolio and country
portfolio_data = {}

for _, row in df.iterrows():
    portfolio = row['PORT_ID']
    country_code = row['CountryCode_3']
    if portfolio not in portfolio_data:
        portfolio_data[portfolio] = {}
    # Assuming one entry per country per portfolio
    portfolio_data[portfolio][country_code] = {
        "Destination_Country": row['Destination_Country'],
        "Origin_Latitude": row['Origin_Latitude'],
        "Origin_Longitude": row['Origin_Longitude'],
        "Dest_Latitude": row['Dest_Latitude'],
        "Dest_Longitude": row['Dest_Longitude'],
        "Investment_Percentage": row['Investment_Percentage']
    }

# Initialize the main figure
fig = go.Figure()

# Initialize lists to track trace indices
choropleth_indices = []      # Indices of choropleth traces
flow_traces_indices = {}     # Dictionary to map portfolio to its flow traces

# Determine color scaling for each portfolio
# Store zmin and zmax for each portfolio to set individual color scales
color_scales = {}

for portfolio in portfolios:
    portfolio_max = df[df['PORT_ID'] == portfolio]['Investment_Percentage'].max()
    color_scales[portfolio] = {
        'zmin': 0,
        'zmax': portfolio_max if portfolio_max > 0 else 1  # Avoid zmax=0
    }

# Step 2: Add choropleth and flow line traces for each portfolio
current_trace = 0  # To keep track of the current trace index

for idx, (portfolio, countries) in enumerate(portfolio_data.items()):
    # Extract locations and investment values
    locations = list(countries.keys())  # ISO-3 country codes
    investment_values = [data['Investment_Percentage'] for data in countries.values()]
    
    # Get color scale parameters for the current portfolio
    zmin = color_scales[portfolio]['zmin']
    zmax = color_scales[portfolio]['zmax']
    
    # Add Choropleth Trace
    choropleth_trace = go.Choropleth(
        locations=locations,
        z=investment_values,
        zmin=zmin,
        zmax=zmax,
        colorscale=[[0, '#f1eef6'], [0.1, '#bdc9e1'], [0.4, '#74a9cf'], [0.7, '#2b8cbe'], [1, '#045a8d']],  # Light grey at 0, blue at max
        showscale=True,  # Show color scale for all choropleths
        colorbar=dict(
            title=dict(
                text="Investment <br>Percentage (%)",
                side="top"   # Position the title at the top of the color bar
            ),
            thickness=0.05,    # Increased thickness for larger colorbar
            len=1.11,         # Increased length for larger colorbar
            y=0.527,           # Center the colorbar vertically
            x=1.01,           # Position to the right of the map
            lenmode='fraction',
            thicknessmode='fraction'
        ),
        hoverinfo='location+z',
        name='Affected Countries',
        marker_line_color='white',
        visible=(idx == 0)  # Only the first portfolio is visible initially
    )
    fig.add_trace(choropleth_trace)
    choropleth_indices.append(current_trace)
    current_trace += 1
    
    # Add Flow Line Traces
    flow_traces = []
    for country_code, data in countries.items():
        # Get color based on investment percentage
        intensity = max(0.3, data['Investment_Percentage'] / zmax)
        flow_trace = go.Scattergeo(
            locationmode='ISO-3',
            lon=[data['Origin_Longitude'], data['Dest_Longitude']],
            lat=[data['Origin_Latitude'], data['Dest_Latitude']],
            mode='lines',
            line=dict(width=1.3, color=f'rgba(4,90,141, {intensity})'),
            hoverinfo='text',
            text=(
                f"Destination: {data['Destination_Country']}<br>"
                f"Investment Percentage: {data['Investment_Percentage']:.2f}%<br>"
                f"Portfolio: {portfolio}"
            ),
            showlegend=False,
            visible=(idx == 0)  # Only the first portfolio's flow lines are visible initially
        )
        fig.add_trace(flow_trace)
        flow_traces.append(current_trace)
        current_trace += 1
    flow_traces_indices[portfolio] = flow_traces

# Step 3: Create Dropdown Buttons
buttons = []

for idx, portfolio in enumerate(portfolios):
    # Initialize visibility for all traces as False
    visibility = [False] * current_trace
    
    # Set the current portfolio's choropleth to True
    choropleth_idx = choropleth_indices[idx]
    visibility[choropleth_idx] = True
    
    # Set the current portfolio's flow lines to True
    for flow_idx in flow_traces_indices[portfolio]:
        visibility[flow_idx] = True
    
    # Create the button dictionary
    button = dict(
        label=portfolio,
        method="update",
        args=[
            {"visible": visibility},
            {"title": f"{portfolio.replace('_',' ')}"}
        ]
    )
    buttons.append(button)

# Step 4: Add the dropdown menu to the layout
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.88,        # Position the dropdown on the right
            xanchor="left",
            y=1.067,        # Align to the top
            yanchor="top"
        )
    ],
    geo=dict(
        projection_type="natural earth",
        showland=True,
        landcolor="#f1eef6",
        coastlinecolor="white",
        countrycolor="white",
        showcountries=True
    ),
    height=800,
    width=1500,          # Increased width to accommodate colorbar and dropdown
    title={
        'text': f'{portfolios[0].replace('_',' ')}',
        'x': 0.077,
    },
)

# Show the figure
fig.show()

# Save the figure as an HTML file
fig.write_html("charts/flow_map_chart.html")