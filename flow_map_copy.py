import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pycountry

# 1. Load your investment data
df = pd.read_csv('flow.csv')

# 2. Load the world map from the local shapefile
world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# 4. Remove entries with missing country codes
df = df[df['CountryCode_3'].notnull()]

# 5. Merge investment data with world geometries
world = world[['ADM0_A3', 'geometry']]
world.columns = ['iso_a3', 'geometry']

# Perform the merge
df = pd.merge(df, world, left_on='CountryCode_3', right_on='iso_a3', how='left')

# Convert df to a GeoDataFrame
df = gpd.GeoDataFrame(df, geometry='geometry', crs=world.crs)

# Remove rows with missing geometries
df = df[df['geometry'].notnull()]

# 6. Calculate representative points (instead of centroids) for destination countries
df['Dest_Point'] = df['geometry'].representative_point()

# Convert representative points to WGS84 for latitude and longitude extraction
df['Dest_Point'] = df['Dest_Point'].to_crs(epsg=4326)
df['Dest_Latitude'] = df['Dest_Point'].y
df['Dest_Longitude'] = df['Dest_Point'].x

# 7. Set Switzerland as the origin in WGS84, making sure it remains constant
switzerland_coords = {'Latitude': 46.8182, 'Longitude': 8.2275}
df['Origin_Latitude'] = switzerland_coords['Latitude']
df['Origin_Longitude'] = switzerland_coords['Longitude']

# 8. Calculate total investment per portfolio
df['Total_Investment'] = df.groupby('PORT_ID')['WEIGHT_'].transform('sum')

# 9. Calculate investment percentage per country
df['Investment_Percentage'] = (df['WEIGHT_'] / df['Total_Investment']) * 100

# 10. Select and rename necessary columns
df_final = df[[
    'PORT_ID',
    'CountryName',
    'Origin_Latitude',
    'Origin_Longitude',
    'Dest_Latitude',
    'Dest_Longitude',
    'CountryCode_3',
    'Investment_Percentage'
]].rename(columns={
    'CountryName': 'Destination_Country'
})

# Group by PORT_ID and Destination_Country, summing Investment_Percentage
df_grouped = df_final.groupby(
    ['PORT_ID', 'Destination_Country', 'Origin_Latitude', 'Origin_Longitude', 'Dest_Latitude', 'Dest_Longitude', 'CountryCode_3'],
    as_index=False
).agg({'Investment_Percentage': 'sum'})

# # Optional: Save the transformed data
df_grouped.to_csv('flow_transformed.csv', index=False)
print("Data transformation complete. Saved to 'flow_transformed.csv'.")

# save excel
df_grouped.to_excel('flow_transformed.xlsx', index=False)