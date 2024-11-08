import pandas as pd
import plotly.graph_objects as go
import pycountry

# Load the transformed data
df = pd.read_csv('flow_transformed.csv')

# Function to get ISO-3 codes
def get_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

# Apply ISO-3 conversion
df['Destination_ISO3'] = df['Destination_Country'].apply(get_iso3)

# Get list of portfolios
portfolios = df['PORT_ID'].unique()

# Function to get affected countries for a portfolio
def get_affected_countries(data):
    # Group by country and calculate cumulative affected intensity for choropleth
    affected_df = data.groupby('Destination_ISO3')['Investment_Percentage'].sum().reset_index()
    return affected_df

# Get all unique ISO-3 codes for the choropleth
all_countries = pycountry.countries
all_iso3 = [country.alpha_3 for country in all_countries]

# Initialize the figure
fig = go.Figure()

# Function to prepare choropleth data
def prepare_choropleth(affected_countries):
    choropleth_df = pd.DataFrame({
        'iso_alpha': all_iso3,
        'Affected': [affected_countries.get(iso, 0) for iso in all_iso3]
    })
    return choropleth_df

# Function to create flow lines with varying blue intensities based on investment percentage
def create_flow_traces(data):
    flow_traces = []
    for _, row in data.iterrows():
        # Ensure intensity is between 0.1 and 1
        intensity = max(0.3, row['Investment_Percentage'] / data['Investment_Percentage'].max())
        flow_traces.append(go.Scattergeo(
            locationmode='ISO-3',
            lon=[row['Origin_Longitude'], row['Dest_Longitude']],
            lat=[row['Origin_Latitude'], row['Dest_Latitude']],
            mode='lines',
            line=dict(width=1, color=f'rgba(0, 0, 255, {intensity})'),  # Blue with minimum transparency
            hoverinfo='text',
            text=(
                f"Destination: {row['Destination_Country']}<br>"
                f"Investment Percentage: {row['Investment_Percentage']:.2f}%"
            ),
            showlegend=False
        ))
    return flow_traces

# Prepare initial choropleth data
initial_portfolio = portfolios[0]
initial_data = df[df['PORT_ID'] == initial_portfolio]
initial_affected_countries_df = get_affected_countries(initial_data)
initial_affected_countries = dict(zip(initial_affected_countries_df['Destination_ISO3'], initial_affected_countries_df['Investment_Percentage']))
initial_choropleth = prepare_choropleth(initial_affected_countries)

# Add choropleth trace
fig.add_trace(go.Choropleth(
    locations=initial_choropleth['iso_alpha'],
    z=initial_choropleth['Affected'],
    colorscale=[[0.0, 'lightgray'], [1.0, 'blue']],
    showscale=True,
    hoverinfo='location',
    name='Affected Countries',
    marker_line_color='white'
))

# Add initial flow lines
initial_flow_traces = create_flow_traces(initial_data)
for trace in initial_flow_traces:
    fig.add_trace(trace)

# Update layout
fig.update_layout(
    title_text=f'Investment Flows from Switzerland - {initial_portfolio}',
    showlegend=False,
    geo=dict(
        projection_type='natural earth',
        showland=True,
        landcolor='lightgray',
        countrycolor='white',
        coastlinecolor='white',
    ),
    height=600,
    width=1000,
)

# Add interactivity with dropdown menu
buttons = []
for portfolio in portfolios:
    data_portfolio = df[df['PORT_ID'] == portfolio]
    affected_countries_df = get_affected_countries(data_portfolio)
    affected_countries = dict(zip(affected_countries_df['Destination_ISO3'], affected_countries_df['Investment_Percentage']))
    choropleth_portfolio = prepare_choropleth(affected_countries)
    flow_lines = create_flow_traces(data_portfolio)
    
    buttons.append(dict(
        label=portfolio,
        method='update',
        args=[{
            'data': [
                go.Choropleth(
                    locations=choropleth_portfolio['iso_alpha'],
                    z=choropleth_portfolio['Affected'],
                    colorscale=[[0.0, 'lightgray'], [1.0, 'blue']],
                    showscale=True,
                    hoverinfo='location',
                    name='Affected Countries',
                    marker_line_color='white'
                )
            ] + flow_lines,
            'layout': {
                'title': f'Investment Flows from Switzerland - {portfolio}'
            }
        }]
    ))

fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction='down',
        showactive=True,
        x=0.1,
        xanchor='left',
        y=1.15,
        yanchor='top'
    )]
)

# Show the plot
fig.show()