import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Step 1: Load and Process the Data
df = pd.read_csv('horizon.csv')  # Ensure the correct filename is used

# Step 2: Convert 'REF_DATE' to datetime
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'], format='%m/%d/%y')

# Step 3: Extract 'Year-Month' as a string
df['Year-Month'] = df['REF_DATE'].dt.to_period('M').astype(str)

# Step 4: Ensure 'VAL' is numeric
df['VAL'] = pd.to_numeric(df['VAL'], errors='coerce').fillna(0)

# Step 5: Define a function to extract metrics based on conditions
def extract_metric(data, fct, fct_out_col, new_col_name):
    return data[
        (data['FCT'] == fct) &
        (data['FCT_OUT_COL'] == fct_out_col)
    ][['PORT_ID', 'Year-Month', 'VAL']].rename(columns={'VAL': new_col_name})

# Step 6: Extract 'Value_Portfolio' and 'Value_Benchmark'
value_portfolio = extract_metric(
    df,
    'CARBON_EMISSIONS_REV_EUR_SCOPE_12',
    'PORT|PRD_NORM_LONG|SUM',
    'Value_Portfolio'
)

value_benchmark = extract_metric(
    df,
    'CARBON_EMISSIONS_REV_EUR_SCOPE_12',
    'BENCH|PRD_NORM_LONG|SUM',
    'Value_Benchmark'
)

# Step 7: Merge the metrics on 'PORT_ID' and 'Year-Month'
merged = pd.merge(
    value_portfolio,
    value_benchmark,
    on=['PORT_ID', 'Year-Month'],
    how='outer'
)

# Step 8: Fill NaN values with 0
merged.fillna(0, inplace=True)

# Step 9: Ensure data is sorted by 'PORT_ID' and 'Year-Month'
merged = merged.sort_values(by=['PORT_ID', 'Year-Month'])

# Step 10: Pivot the data for Portfolio
pivot_value_portfolio = merged.pivot_table(
    index='Year-Month',
    columns='PORT_ID',
    values='Value_Portfolio'
).fillna(0)

# Step 11: Pivot the data for Benchmark
pivot_value_benchmark = merged.pivot_table(
    index='Year-Month',
    columns='PORT_ID',
    values='Value_Benchmark'
).fillna(0)

# Step 12: Create a complete date range from January 2022 to October 2024
all_months = pd.period_range(start='2022-12', end='2024-10', freq='M').astype(str)

# Step 13: Reindex the pivot tables to ensure all months are present
pivot_value_portfolio = pivot_value_portfolio.reindex(all_months, fill_value=0)
pivot_value_benchmark = pivot_value_benchmark.reindex(all_months, fill_value=0)

# Step 14: Reset index to have 'Year-Month' as a column
pivot_value_portfolio.reset_index(inplace=True)
pivot_value_benchmark.reset_index(inplace=True)

# rename it to 'Year-Month'
pivot_value_portfolio.rename(columns={'index': 'Year-Month'}, inplace=True)
pivot_value_benchmark.rename(columns={'index': 'Year-Month'}, inplace=True)

# Step 15: List of portfolios
portfolios = pivot_value_portfolio.columns.tolist()

# Step 16: Safely remove 'Year-Month' and 'index' if they exist
portfolios = [p for p in portfolios if p not in ['Year-Month', 'index']]

# Step 17: Print the portfolios list for debugging
print("Portfolios after removal:", portfolios)

# Step 18: Define number of bands and colors
bands = 4
colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b']  # From dark red to light green

# Step 19: Create subplots: 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Portfolio Emissions", "Benchmark Emissions")
)

# Step 20: Initialize a dictionary to track trace indices per portfolio
horizon_traces_indices = {}

current_trace = 0  # To keep track of the current trace index

# Step 21: Iterate through each portfolio to create traces
for portfolio in portfolios:
    horizon_traces_indices[portfolio] = {'portfolio': [], 'benchmark': []}
    
    # Determine the maximum value for scaling
    max_portfolio = pivot_value_portfolio[portfolio].max()
    max_benchmark = pivot_value_benchmark[portfolio].max()
    
    # Avoid division by zero
    max_portfolio = max_portfolio if max_portfolio > 0 else 1
    max_benchmark = max_benchmark if max_benchmark > 0 else 1
    
    # Calculate band size
    band_size_portfolio = max_portfolio / bands
    band_size_benchmark = max_benchmark / bands
    
    # Create traces for Portfolio
    for b in range(1, bands + 1):
        y = pivot_value_portfolio[portfolio].apply(
            lambda x: min(x, b * band_size_portfolio) - (b - 1) * band_size_portfolio
        )
        y = y.clip(lower=0)  # Ensure no negative values
        
        trace = go.Scatter(
            x=pivot_value_portfolio['Year-Month'],
            y=y,
            mode='lines',
            name=f'{portfolio} Band {b} Portfolio',
            fill='tonexty' if b > 1 else 'none',
            fillcolor=colors[b - 1],
            line=dict(width=0.5, color=colors[b - 1]),
            hoverinfo='text',
            text=pivot_value_portfolio.apply(
                lambda row: f"<b>Portfolio:</b> {portfolio}<br><b>Date:</b> {row['Year-Month']}<br><b>Emissions:</b> {row[portfolio]:.2f}",
                axis=1
            ),
            showlegend=False,
            visible=(portfolio == portfolios[0])  # Only first portfolio visible initially
        )
        fig.add_trace(trace, row=1, col=1)
        horizon_traces_indices[portfolio]['portfolio'].append(current_trace)
        current_trace += 1
    
    # Create traces for Benchmark
    for b in range(1, bands + 1):
        y = pivot_value_benchmark[portfolio].apply(
            lambda x: min(x, b * band_size_benchmark) - (b - 1) * band_size_benchmark
        )
        y = y.clip(lower=0)  # Ensure no negative values
        
        trace = go.Scatter(
            x=pivot_value_benchmark['Year-Month'],
            y=y,
            mode='lines',
            name=f'{portfolio} Band {b} Benchmark',
            fill='tonexty' if b > 1 else 'none',
            fillcolor=colors[b - 1],
            line=dict(width=0.5, color=colors[b - 1]),
            hoverinfo='text',
            text=pivot_value_benchmark.apply(
                lambda row: f"<b>Benchmark:</b> {portfolio}<br><b>Date:</b> {row['Year-Month']}<br><b>Emissions:</b> {row[portfolio]:.2f}",
                axis=1
            ),
            showlegend=False,
            visible=(portfolio == portfolios[0])  # Only first portfolio's benchmark visible initially
        )
        fig.add_trace(trace, row=1, col=2)
        horizon_traces_indices[portfolio]['benchmark'].append(current_trace)
        current_trace += 1

# Step 22: Create Dropdown Buttons
buttons = []
for portfolio in portfolios:
    # Initialize visibility for all traces as False
    visibility = [False] * current_trace
    
    # Set the current portfolio's traces to True
    for trace_idx in horizon_traces_indices[portfolio]['portfolio']:
        visibility[trace_idx] = True
    for trace_idx in horizon_traces_indices[portfolio]['benchmark']:
        visibility[trace_idx] = True
    
    # Create the button dictionary
    button = dict(
        label=portfolio,
        method="update",
        args=[
            {"visible": visibility},
            {"title": f"Carbon Emissions - {portfolio} vs Benchmark (2022 - Oct 2024)"}
        ]
    )
    buttons.append(button)

# Step 23: Update Layout with Dropdown Menu and Annotations
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=buttons,
            x=0.86,
            y=1.15,
            xanchor='left',
            yanchor='top'
        )
    ],
    title=f"Carbon Emissions - {portfolios[0]} vs Benchmark (2022 - Oct 2024)",
    xaxis=dict(title='Date', tickangle=-45),
    yaxis=dict(title='Carbon Emissions (EUR Scope 1,2)'),
    hovermode='closest',
    height=600,
    width=1900,
    margin=dict(l=50, r=250, t=100, b=150)
)

# Step 24: Add Annotations (Optional)
fig.update_layout(
    annotations=[
        dict(
            text="Select Portfolio:",
            x=0.9,
            y=1.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left"
        )
    ]
)

# Step 25: Optional Debugging Prints
print("\nPivot Value Portfolio:")
print(pivot_value_portfolio.head())

print("\nPivot Value Benchmark:")
print(pivot_value_benchmark.head())

print("\nPortfolios list:", portfolios)

# Step 26: Show the figure in the browser during development
fig.show()

# Step 27: Save the figure to an HTML file for GitHub Pages
# Uncomment the following line after successful visualization
# fig.write_html('horizon_graph.html', include_plotlyjs='cdn')