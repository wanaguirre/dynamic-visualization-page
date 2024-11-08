import pandas as pd

# 1. Load your data
df = pd.read_csv('horizon.csv')

# 2. Convert REF_DATE to datetime
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

# 3. Filter data for Carbon Emission Intensity
# Assuming 'CARBON_EMISSIONS_REV_EUR_SCOPE_12' represents carbon emission intensity
df = df[df['FCT'] == 'CARBON_EMISSIONS_REV_EUR_SCOPE_12']

# 4. Separate Portfolio and Benchmark Data
# Portfolio Data
df_portfolio = df[df['FCT_OUT_COL'].str.contains('PORT')].copy()
df_portfolio['Type'] = 'Portfolio'

# Benchmark Data
df_benchmark = df[df['FCT_OUT_COL'].str.contains('BENCH')].copy()
df_benchmark['Type'] = 'Benchmark'

# 5. Combine Portfolio and Benchmark Data
df_combined = pd.concat([df_portfolio, df_benchmark], ignore_index=True)

# 6. Prepare Data for Visualization
# Pivot the data to have 'Portfolio' and 'Benchmark' as separate columns
df_pivot = df_combined.pivot_table(
    index=['PORT_ID', 'REF_DATE'],
    columns='Type',
    values='VAL'
).reset_index()

# 7. Handle missing values if any
df_pivot = df_pivot.fillna(0)

# 8. Save the transformed data
df_pivot.to_csv('horizon_transformed.csv', index=False)

print("Data transformation complete. Saved as 'horizon_transformed.csv'.")