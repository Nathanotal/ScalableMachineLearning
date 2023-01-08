# Open every file in the data folder and concatenate them into one file

import os
import csv
import pandas as pd

fileNames = os.listdir('data')

allData = []
header = []
for fileName in fileNames:
    df = pd.read_csv('data/' + fileName, delimiter=';', encoding='utf-8')
    header = df.columns.tolist()
    # Remove all rows which equal: link;area;streetName;number;sqm;rooms;Slutpris;Utropspris;Prisutveckling;Såld eller borttagen;Slutpris/m²;Dagar på Booli;Avgift;Bostadstyp;Driftskostnad;Våning;Byggår;BRF;brfLink;Energiklass;agency
    df = df[df['link'] != 'link']
    
    # Add all of the data from the df to the allData list, not the header
    allData.extend(df.values.tolist())

# Write the data to a file
with open('allData.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(header)
    writer.writerows(allData)
