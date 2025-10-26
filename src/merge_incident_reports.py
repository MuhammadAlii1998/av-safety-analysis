import pandas as pd

# Filenames
file_ads = '../data/SGO-2021-01_Incident_Reports_ADS.csv'
file_adas = '../data/SGO-2021-01_Incident_Reports_ADAS.csv'

# Read CSVs
ads_df = pd.read_csv(file_ads)
adas_df = pd.read_csv(file_adas)

# Add source column for clarity
ads_df['Source_System'] = 'ADS'
adas_df['Source_System'] = 'ADAS'

# Merge the two files
data_merged = pd.concat([ads_df, adas_df], ignore_index=True)

# Save output file
merged_filename = 'NHTSA_crash_data.csv'
data_merged.to_csv(merged_filename, index=False)

# Output first 5 rows for preview
data_merged.head().to_csv('Merged_Incident_Reports_ADS_ADAS_preview.csv', index=False)
'Files merged successfully. Preview saved as Merged_Incident_Reports_ADS_ADAS_preview.csv.'