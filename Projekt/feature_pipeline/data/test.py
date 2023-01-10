import pandas as pd
import os
PATH =  os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(f'{PATH}/features.csv', sep=';')

# Get the agencies which have sold the most apartments
topAgencies = df['agency'].value_counts().head(20).index.tolist()

for agency in topAgencies:
    # Get the data for the agency
    # Get the average price per square meter for the agency
    print(f'{agency}')