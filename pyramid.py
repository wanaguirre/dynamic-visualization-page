import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# Load the data
df = pd.read_csv('pyramid_transformed.csv')

# Filter relevant FCT_OUT_COL values
relevant_fct_out_col = [
    'PORT|PRD_NORM_LONG|SUM',
    'PORT|WGT_COV_LONG|SUM',
    'PORT|WGT_GR|SUM'
]
df = df[df['FCT_OUT_COL'].isin(relevant_fct_out_col)]

# Pivot the data to have one row per portfolio with columns for each scope
df_pivot = df.pivot_table(
    index='PORT_ID',
    columns='FCT',
    values='VAL',
    aggfunc='first'
).reset_index()

# Rename columns for clarity
df_pivot = df_pivot.rename(columns={
    'CARBON_EMISSIONS_EVIC_EUR_SCOPE_1': 'Scope 1',
    'CARBON_EMISSIONS_EVIC_EUR_SCOPE_2': 'Scope 2',
    'CARBON_EMISSIONS_EVIC_EUR_SCOPE_3': 'Scope 3'
})

# Calculate total emissions per portfolio
df_pivot['Total'] = df_pivot['Scope 1'] + df_pivot['Scope 2'] + df_pivot['Scope 3']

# Convert emission values to percentages scaled up to 100%
df_pivot['Scope 1 %'] = df_pivot['Scope 1'] / df_pivot['Total'] * 100
df_pivot['Scope 2 %'] = df_pivot['Scope 2'] / df_pivot['Total'] * 100
df_pivot['Scope 3 %'] = df_pivot['Scope 3'] / df_pivot['Total'] * 100

# Adjust percentages for the triangle structure
# The area under the triangle is proportional to the square of the height (A ∝ h^2)
# For each cumulative percentage p, the corresponding height h is h = sqrt(p)

# Calculate cumulative percentages from bottom to top
df_pivot['p0'] = 0
df_pivot['p1'] = df_pivot['Scope 3 %'] / 100
df_pivot['p2'] = df_pivot['p1'] + df_pivot['Scope 2 %'] / 100
df_pivot['p3'] = 1.0  # Total cumulative percentage is 100%

# Calculate heights corresponding to cumulative percentages
df_pivot['h0'] = 0
df_pivot['h1'] = np.sqrt(df_pivot['p1']) / 1.65
df_pivot['h2'] = np.sqrt(df_pivot['p2']) / 1.4
df_pivot['h3'] = 1.0  # Total height normalized to 1

# Calculate bar segment heights
df_pivot['Scope 3 height'] = df_pivot['h1'] - df_pivot['h0']
df_pivot['Scope 2 height'] = df_pivot['h2'] - df_pivot['h1']
df_pivot['Scope 1 height'] = df_pivot['h3'] - df_pivot['h2']

# List of portfolios
portfolios = df_pivot['PORT_ID'].tolist()

# Function to sort portfolios naturally (e.g., Portfolio_1, Portfolio_2, ..., Portfolio_10)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# Sort portfolios using the natural sort function
portfolios = sorted(portfolios, key=natural_sort_key)

# Colors and labels for the scopes
colors = ['#fff7bc', '#fec44f', '#d95f0e']  # Scope 3, Scope 2, Scope 1
labels = ['Scope 3', 'Scope 2', 'Scope 1']

# Initialize the figure
fig = go.Figure()

# Add bar traces for each scope
for i in range(3):
    fig.add_trace(go.Bar(
        x=[0],
        y=[0],  # Will be updated later
        marker_color=colors[i],
        name=labels[i],
        text='',  # Will be updated later
        textposition='none',
        hoverinfo='text',
        showlegend=True
    ))

# Add scatter traces for annotations
for i in range(3):
    fig.add_trace(go.Scatter(
        x=[0],  # Will be updated later
        y=[0],  # Will be updated later
        text='',  # Will be updated later
        mode="text",
        showlegend=False
    ))

# Set up the layout
fig.update_layout(
    barmode='stack',
    xaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.866, 0.866]
    ),
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1]
    ),
    plot_bgcolor='white',
    height=600,
    width=1200,
    title={
        'text': f'{portfolios[0].replace('_',' ')}',
        'x': 0.3,
    },
    shapes=[
        # Left triangle to create the pyramid effect
        dict(
            type='path',
            path='M -0.5,-0.3 L 0,1 L -0.5,1 Z',
            fillcolor='white',
            line=dict(width=0),
            layer='above'
        ),
        # Right triangle
        dict(
            type='path',
            path='M 0.5,-0.3 L 0,1 L 0.5,1 Z',
            fillcolor='white',
            line=dict(width=0),
            layer='above'
        )
    ]
)

# Prepare data for each portfolio
portfolio_data_list = []
for portfolio_id in portfolios:
    portfolio_data_row = df_pivot[df_pivot['PORT_ID'] == portfolio_id].iloc[0]
    heights = [
        portfolio_data_row['Scope 3 height'],
        portfolio_data_row['Scope 2 height'],
        portfolio_data_row['Scope 1 height']
    ]
    percentages = [
        portfolio_data_row['Scope 3 %'],
        portfolio_data_row['Scope 2 %'],
        portfolio_data_row['Scope 1 %']
    ]
    texts = [f"{percent:.2f}%" for percent in percentages]
    
    # Update bar traces data
    bar_ys = [[heights[0]], [heights[1]], [heights[2]]]
    bar_texts = [''] * 3  # No text inside bars

    # Calculate positions for annotations
    y_pos = 0
    annotation_xs = [[0.05]] * 3  # Positions on the right side
    annotation_ys = []
    annotation_texts = []
    for i in range(3):
        height = heights[i]
        y_pos += height / 2
        annotation_ys.append([y_pos])
        annotation_texts.append([f"{texts[i]}"])
        y_pos += height / 2

    # Store the data
    portfolio_data_list.append({
        'portfolio_id': portfolio_id,
        'bar_ys': bar_ys,
        'bar_texts': bar_texts,
        'annotation_xs': annotation_xs,
        'annotation_ys': annotation_ys,
        'annotation_texts': annotation_texts
    })

# Initialize the figure with the first portfolio's data
initial_data = portfolio_data_list[0]

# Update the bar traces
for i in range(3):
    fig.data[i].y = initial_data['bar_ys'][i]
    fig.data[i].text = initial_data['bar_texts'][i]

# Update the annotations (scatter traces)
for i in range(3):
    fig.data[i+3].x = initial_data['annotation_xs'][i]
    fig.data[i+3].y = initial_data['annotation_ys'][i]
    fig.data[i+3].text = initial_data['annotation_texts'][i]
    fig.data[i+3].textfont = dict(color='black', size=12)
    fig.data[i+3].textposition = 'middle left'
    fig.data[i+3].showlegend = False

# Create buttons for the dropdown menu
buttons = []
for data in portfolio_data_list:
    portfolio_id = data['portfolio_id'].replace('_', ' ')
    # Build the lists for updating traces
    y_values = data['bar_ys'] + data['annotation_ys']
    text_values = data['bar_texts'] + data['annotation_texts']
    x_values = [fig.data[i].x for i in range(3)] + data['annotation_xs']

    button = dict(
        label=portfolio_id,
        method='update',
        args=[
            {
                'y': y_values,
                'text': text_values,
                'x': x_values
            },
            {'title': f'{portfolio_id.replace('_',' ')}'}
        ]
    )
    buttons.append(button)

# Add the dropdown menu to the layout
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.8,
            xanchor='left',
            y=0.5,
            yanchor='middle'
        )
    ],
    legend=dict(
        x=0.79,
        y=0.8,
        xanchor='left',
        yanchor='middle'
    )
)

# Show the figure
fig.show()

# Save the figure as an HTML file
fig.write_html("charts/pyramid_chart.html")