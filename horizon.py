# Import Libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the Data
df = pd.read_csv('horizon2.csv')

# Ensure 'VAL' is numeric
df['VAL'] = pd.to_numeric(df['VAL'], errors='coerce')

# Define the metrics of interest
metric_scope12 = 'CARBON_EMISSIONS_REV_EUR_SCOPE_12'
metric_com_score = 'COM_SCORE'

# Filter data for Portfolios
value_portfolios = df[
    (df['FCT'] == metric_scope12) &
    (df['FCT_OUT_COL'] == 'PORT|PRD_NORM_LONG|SUM')
].copy()

coverage_portfolios = df[
    (df['FCT'] == metric_scope12) &
    (df['FCT_OUT_COL'] == 'PORT|WGT_COV_LONG|SUM')
].copy()

eligible_assets_portfolios = df[
    (df['FCT'] == metric_com_score) &
    (df['FCT_OUT_COL'] == 'PORT|WGT_GR|SUM')
].copy()

# Filter data for Benchmarks
value_benchmarks = df[
    (df['FCT'] == metric_scope12) &
    (df['FCT_OUT_COL'] == 'BENCH|PRD_NORM_LONG|SUM')
].copy()

coverage_benchmarks = df[
    (df['FCT'] == metric_scope12) &
    (df['FCT_OUT_COL'] == 'BENCH|WGT_COV_LONG|SUM')
].copy()

eligible_assets_benchmarks = df[
    (df['FCT'] == metric_com_score) &
    (df['FCT_OUT_COL'] == 'BENCH|WGT_GR|SUM')
].copy()

# Function to remove related coverage and eligible assets if Value is NaN
def remove_related_entries(value_df, coverage_df, eligible_assets_df):
    # Get list of (PORT_ID, REF_DATE) where Value is NaN
    nan_entries = value_df[value_df['VAL'].isna()][['PORT_ID', 'REF_DATE']]
    
    for index, row in nan_entries.iterrows():
        port_id = row['PORT_ID']
        ref_date = row['REF_DATE']
        
        # Remove from Coverage DataFrame
        coverage_df.drop(
            coverage_df[
                (coverage_df['PORT_ID'] == port_id) &
                (coverage_df['REF_DATE'] == ref_date)
            ].index,
            inplace=True
        )
        
        # Remove from Eligible Assets DataFrame
        eligible_assets_df.drop(
            eligible_assets_df[
                (eligible_assets_df['PORT_ID'] == port_id) &
                (eligible_assets_df['REF_DATE'] == ref_date)
            ].index,
            inplace=True
        )
        
    return coverage_df, eligible_assets_df

# Apply the function to Portfolios
coverage_portfolios, eligible_assets_portfolios = remove_related_entries(
    value_portfolios,
    coverage_portfolios,
    eligible_assets_portfolios
)

# Apply the function to Benchmarks
coverage_benchmarks, eligible_assets_benchmarks = remove_related_entries(
    value_benchmarks,
    coverage_benchmarks,
    eligible_assets_benchmarks
)

# Pivot the data for Portfolios
pivot_value_portfolios = value_portfolios.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

pivot_coverage_portfolios = coverage_portfolios.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

pivot_eligible_assets_portfolios = eligible_assets_portfolios.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

# Pivot the data for Benchmarks
pivot_value_benchmarks = value_benchmarks.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

pivot_coverage_benchmarks = coverage_benchmarks.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

pivot_eligible_assets_benchmarks = eligible_assets_benchmarks.pivot_table(
    index='REF_DATE',
    columns='PORT_ID',
    values='VAL'
)

# Convert 'REF_DATE' to datetime
for df in [
    pivot_value_portfolios,
    pivot_coverage_portfolios,
    pivot_eligible_assets_portfolios,
    pivot_value_benchmarks,
    pivot_coverage_benchmarks,
    pivot_eligible_assets_benchmarks
]:
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

# Define the resampling frequency ('D' for daily)
resample_freq = 'D'

# Resample and interpolate the pivoted DataFrames
def resample_and_interpolate(df):
    df_resampled = df.resample(resample_freq).asfreq()
    df_interpolated = df_resampled.interpolate(method='linear')
    
    # Add randomness
    noise = np.random.uniform(-0.1, 0.1, size=df_interpolated.shape)
    df_interpolated += noise
    
    df_interpolated.ffill(inplace=True)
    return df_interpolated

pivot_value_portfolios = resample_and_interpolate(pivot_value_portfolios)
pivot_coverage_portfolios = resample_and_interpolate(pivot_coverage_portfolios)
pivot_eligible_assets_portfolios = resample_and_interpolate(pivot_eligible_assets_portfolios)

pivot_value_benchmarks = resample_and_interpolate(pivot_value_benchmarks)
pivot_coverage_benchmarks = resample_and_interpolate(pivot_coverage_benchmarks)
pivot_eligible_assets_benchmarks = resample_and_interpolate(pivot_eligible_assets_benchmarks)

# Adjust start and end dates
start_date = pivot_value_portfolios.index.min()
end_date = pivot_value_portfolios.index.max()

# Recalculate percentage changes for Portfolios
portfolio_start_values = pivot_value_portfolios.loc[start_date]
portfolio_pct_change = ((pivot_value_portfolios - portfolio_start_values) / portfolio_start_values) * 100

# Recalculate percentage changes for Benchmarks
benchmark_start_values = pivot_value_benchmarks.loc[start_date]
benchmark_pct_change = ((pivot_value_benchmarks - benchmark_start_values) / benchmark_start_values) * 100

# Replace infinite values and handle NaNs
for df in [
    portfolio_pct_change,
    benchmark_pct_change
]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)  # Forward fill NaNs after interpolation

# Set overall min and max percentage change values
overall_min = -100  # Set minimum percentage change to -100%
overall_max = 100   # Set maximum percentage change to 100%

# Define colors for positive and negative bands
num_bands = 10  # From -100% to 100% in steps of 20%
band_size = 20

# Colors for positive bands (e.g., shades of green), from light to dark
pos_colors = ['#e5f5e0', '#c7e9c0', '#74c476', '#31a354', '#006d2c']
assert len(pos_colors) == num_bands//2, "Number of positive colors must match num_bands//2."

# Colors for negative bands (e.g., shades of red), from light to dark
neg_colors = ['#fee0d2', '#fcbba1', '#fb6a4a', '#ef3b2c', '#cb181d']
assert len(neg_colors) == num_bands//2, "Number of negative colors must match num_bands//2."

# Define the Horizon Graph Function (Updated)
def create_horizon_traces(pct_change_series, coverage_series, eligible_assets_series, x, pos_colors, neg_colors):
    """
    Create horizon graph traces for a single time series, overlaying positive and negative bands onto the positive axis.

    Parameters:
    - pct_change_series: Pandas Series of percentage changes over time.
    - coverage_series: Pandas Series of coverage percentages over time.
    - eligible_assets_series: Pandas Series of eligible assets percentages over time.
    - x: List of x-axis values (dates).
    - pos_colors: List of colors for positive bands.
    - neg_colors: List of colors for negative bands.

    Returns:
    - List of Scatter traces for the horizon graph.
    """
    traces = []

    # Reverse colors for proper stacking
    pos_colors_reversed = pos_colors[::-1]
    neg_colors_reversed = neg_colors[::-1]

    # Generate hover text once
    hover_text = [
        f"Date: {date.strftime('%Y-%m-%d')}<br>Value: {v:.2f}%<br>Coverage: {c:.2f}<br>Eligible Assets: {e:.2f}"
        if not pd.isnull(v) else "Data Not Available"
        for date, v, c, e in zip(x, pct_change_series, coverage_series, eligible_assets_series)
    ]

    # For positive bands (plotting from highest to lowest band)
    for idx, i in enumerate(reversed(range(num_bands//2))):
        lower = i * band_size
        upper = (i + 1) * band_size

        # For each point in time, calculate band_data
        band_data = pct_change_series.apply(
            lambda val: band_size if val >= upper else (val - lower if val >= lower else 0)
        )

        band_data = band_data.clip(lower=0, upper=band_size)
        band_data.index = x

        # For the first band plotted (highest positive band), set fill to 'tozeroy'
        if idx == 0:
            fill_option = 'tozeroy'
        else:
            fill_option = 'tonexty'

        # Positive band trace
        traces.append(go.Scatter(
            x=x,
            y=band_data,
            mode='lines',
            line=dict(width=0.5, color=pos_colors_reversed[idx]),
            fill=fill_option,
            fillcolor=pos_colors_reversed[idx],
            hoverinfo='skip',  # Hide hoverinfo for bands
            showlegend=False
        ))

    # For negative bands (plotting from highest to lowest band)
    for idx, i in enumerate(reversed(range(num_bands//2))):
        lower = -((i + 1) * band_size)
        upper = -i * band_size

        # For each point in time, calculate band_data
        band_data = (-pct_change_series).apply(
            lambda val: band_size if val >= -lower else (val - (-upper) if val >= -upper else 0)
        )

        band_data = band_data.clip(lower=0, upper=band_size)
        band_data.index = x

        # For the first band plotted (highest negative band), set fill to 'tozeroy'
        if idx == 0:
            fill_option = 'tozeroy'
        else:
            fill_option = 'tonexty'

        # Negative band trace (reflected onto positive axis)
        traces.append(go.Scatter(
            x=x,
            y=band_data,
            mode='lines',
            line=dict(width=0.5, color=neg_colors_reversed[idx]),
            fill=fill_option,
            fillcolor=neg_colors_reversed[idx],
            hoverinfo='skip',  # Hide hoverinfo for bands
            showlegend=False
        ))

    # Add an invisible trace to capture hover events
    traces.append(go.Scatter(
        x=x,
        y=[10] * len(x),  # Place the trace at the middle of the plot
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        text=hover_text,
        hoverlabel=dict(align='left'),  # Align hover text to the left
        showlegend=False
    ))

    return traces

# Prepare the Plot
portfolios = portfolio_pct_change.columns.tolist()
benchmarks = benchmark_pct_change.columns.tolist()

# Ensure that the number of Portfolios and Benchmarks match
assert len(portfolios) == len(benchmarks), "Mismatch in number of portfolios and benchmarks."

num_portfolios = len(portfolios)

# Combine portfolio and benchmark titles in pairs
subplot_titles = []
for portfolio, benchmark in zip(portfolios, benchmarks):
    subplot_titles.extend([f"{portfolio.replace('_', ' ')}", f"{benchmark.replace('_', ' ')} (Benchmark)"])

# Create subplots with increased vertical spacing
fig = make_subplots(
    rows=num_portfolios,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.08,  # Increase vertical spacing between rows
    horizontal_spacing=0.05,
    subplot_titles=subplot_titles
)

# Plot the Horizon Graphs
for i, (portfolio, benchmark) in enumerate(zip(portfolios, benchmarks), start=1):
    x = portfolio_pct_change.index.tolist()

    # Portfolio Horizon Graph (Left)
    pct_change_portfolio = portfolio_pct_change[portfolio]
    coverage_portfolio = pivot_coverage_portfolios[portfolio]  # Use original values
    eligible_assets_portfolio = pivot_eligible_assets_portfolios[portfolio]  # Use original values

    # Reindex coverage and eligible assets to match x
    coverage_portfolio = coverage_portfolio.reindex(x).ffill()
    eligible_assets_portfolio = eligible_assets_portfolio.reindex(x).ffill()

    traces_portfolio = create_horizon_traces(
        pct_change_series=pct_change_portfolio,
        coverage_series=coverage_portfolio,
        eligible_assets_series=eligible_assets_portfolio,
        x=x,
        pos_colors=pos_colors,
        neg_colors=neg_colors
    )

    for trace in traces_portfolio:
        fig.add_trace(trace, row=i, col=1)

    # Benchmark Horizon Graph (Right)
    pct_change_benchmark = benchmark_pct_change[benchmark]
    coverage_benchmark = pivot_coverage_benchmarks[benchmark]  # Use original values
    eligible_assets_benchmark = pivot_eligible_assets_benchmarks[benchmark]  # Use original values

    # Reindex coverage and eligible assets to match x
    coverage_benchmark = coverage_benchmark.reindex(x).ffill()
    eligible_assets_benchmark = eligible_assets_benchmark.reindex(x).ffill()

    traces_benchmark = create_horizon_traces(
        pct_change_series=pct_change_benchmark,
        coverage_series=coverage_benchmark,
        eligible_assets_series=eligible_assets_benchmark,
        x=x,
        pos_colors=pos_colors,
        neg_colors=neg_colors
    )

    for trace in traces_benchmark:
        fig.add_trace(trace, row=i, col=2)

# Define y-axis tick values and corresponding text with a percentage sign
y_tick_vals = [0, 5, 10, 15, 20]
y_tick_text = [f"{val}%" for val in y_tick_vals]

# Adjust Axes with Customizations
for i in range(1, num_portfolios + 1):
    # Portfolio subplot (Left)
    fig.update_yaxes(
        tickvals=y_tick_vals,  # Set tick values
        ticktext=y_tick_text,  # Set tick text with % sign
        showticklabels=True,
        showgrid=False,
        zeroline=False,
        row=i,
        col=1,
        range=[0, 20]
    )
    # Benchmark subplot (Right)
    fig.update_yaxes(
        tickvals=y_tick_vals,  # Set tick values
        ticktext=y_tick_text,  # Set tick text with % sign
        showticklabels=True,
        showgrid=False,
        zeroline=False,
        row=i,
        col=2,
        range=[0, 20]
    )

    if i == num_portfolios:
        # Last row plots
        fig.update_xaxes(
            row=i,
            col=1,
            tickformat='%Y-%m',
            nticks=20,  # Keep the same number of ticks
            showgrid=False,
            range=[start_date, end_date],
            title_text='Date'
        )
        fig.update_xaxes(
            row=i,
            col=2,
            tickformat='%Y-%m',
            nticks=20,
            showgrid=False,
            range=[start_date, end_date],
            title_text='Date'
        )
    else:
        # Other rows: only show first and last date labels
        fig.update_xaxes(
            row=i,
            col=1,
            showgrid=False,
            range=[start_date, end_date],
            tickvals=[start_date, end_date],
            ticktext=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
            title_text=''  # Remove x-axis title
        )
        fig.update_xaxes(
            row=i,
            col=2,
            showgrid=False,
            range=[start_date, end_date],
            tickvals=[start_date, end_date],
            ticktext=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
            title_text=''
        )

# Update the layout of the figure
fig.update_layout(
    height=150 * num_portfolios,  # Adjust height based on number of portfolios
    width=1500,
    hovermode='x unified',
    template='simple_white',
    margin=dict(l=100, r=200, t=50, b=100)
)

# Create color legend manually, ordered from +100% at the top to -100% at the bottom
legend_traces = []

# Positive bands (reversed order for +100% at top)
for i in reversed(range(num_bands//2)):
    lower = i * band_size
    upper = (i + 1) * band_size
    label = f"+{lower}% to +{upper}%"
    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=pos_colors[i]),
            showlegend=True,
            name=label
        )
    )

# Negative bands (reversed order for -100% at bottom)
for i in reversed(range(num_bands//2)):
    lower_neg = -((i + 1) * band_size)
    upper_neg = -i * band_size
    label = f"{lower_neg}% to {upper_neg}%"
    legend_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=neg_colors[i]),
            showlegend=True,
            name=label
        )
    )

# Add all legend traces to the figure
for trace in legend_traces:
    fig.add_trace(trace)

# Update legend layout with title and smaller font size
fig.update_layout(
    legend=dict(
        title=dict(
            text="Percentage Change",  # Title for the legend
            font=dict(size=12)         # Slightly smaller font for the title
        ),
        orientation='v',              # Vertical legend
        yanchor='top',
        y=1.018,
        xanchor='left',
        x=1.02,                        # Position legend to the right
        font=dict(size=10),            # Smaller font for legend items
        itemwidth=30,
        itemsizing='constant',
    ),
    margin=dict(l=100, r=200, t=70, b=100)  # Adjust margins for cleaner layout
)

# Update layout to reduce subplot title font size
fig.update_layout(
    annotations=[
        dict(
            font=dict(size=15)  # Adjust font size for subplot titles
        )
        for annotation in fig['layout']['annotations']
    ]
)

# Display the figure
fig.show()

# Save the figure as an HTML file
fig.write_html("charts/horizon_chart.html")