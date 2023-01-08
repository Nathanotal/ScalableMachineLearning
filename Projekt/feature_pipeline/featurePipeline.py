# import hopsworks
# from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from getCoords import getCoordinatesFromAddress
from tqdm import tqdm

PATH = os.path.dirname(os.path.abspath(__file__))

def main(apartmentDf):
    data = loadData(apartmentDf)

    # Save to csv
    print('Saving features to csv...')
    data.to_csv(f'{PATH}/data/features.csv', sep=';', index=False)
    


def loadData(apartmentDf):
    # Load 4 separate csv files and concatenate them into a single DatasetDict
    print('Loading financial data...')
    gdpDf = pd.read_csv(f'{PATH}/data/historicalGDP.csv', sep=';')
    unemploymentDf = pd.read_csv(f'{PATH}/data/historicalUnemployment.csv', sep=';')
    interestRateDf = pd.read_csv(f'{PATH}/data/historicalInterest.csv', sep=';')
    
    cleanApartmentDf = cleanData(apartmentDf)
    df = populateApartmentData(cleanApartmentDf, gdpDf, unemploymentDf, interestRateDf)
    df = addCoordinates(df)
    # A lot of records are dropped here, but if we don't do this our coordinate understanding will be off
    # df = dropZeroCoords(df) # Do this in the training script instead

    return df

def dropZeroCoords(df):
    return df[(df['lat'] != 0) | (df['lon'] != 0)]


def cleanData(df): # TODO: Clean data
    print('Cleaning data...')
    rowsBefore = len(df)
    # Set index to the link
    df = df.set_index('link')

    # Drop all rows where there is no streetName (equals "Adresssaknas")
    df = df[df['streetName'] != 'Adresssaknas']
    
    # Rename columns
    df = df.rename(columns={'Slutpris': 'price', 'Såld eller borttagen': 'soldDate', 'Avgift': 'monthlyFee', 'Driftskostnad': 'monthlyCost', 'Våning': 'floor', 'Byggår': 'yearBuilt', 'BRF': 'brf', 'Energiklass': 'energyClass'})
    
    # Convert the soldDate column to datetime
    df['soldDate'] = pd.to_datetime(df['soldDate'])
    
    # Drop all columns which are useless (Slutpris/m², Prisutveckling, Utropspris)
    df = df.drop(['energyClass', 'Slutpris/m²', 'Prisutveckling', 'Utropspris', 'Dagar på Booli', 'Bostadstyp'], axis=1)
    
    # Set the null monthlyCost to 0
    df['monthlyCost'] = df['monthlyCost'].fillna(0)
    df['monthlyFee'] = df['monthlyFee'].fillna(0)
    
    # Drop the brfLink until we determine if we want to use it
    df = df.drop(['brfLink'], axis=1)
    
    # Fill the null values in the brf column with 'NoBRF'
    df['brf'] = df['brf'].fillna('NoBRF')
    
    # Where number is null, set it to 1
    df['number'] = df['number'].fillna(1)
    
    # Where agency is null, set it to 'NoAgency'
    df['agency'] = df['agency'].fillna('NoAgency')
    
    # Drop rows with 2 or more null values
    df = df.drop(df[df.isnull().sum(axis=1) >= 2].index)
    
    # Drop all rows where the floor or yearBuilt is null
    df = df.drop(df[df['floor'].isnull() | df['yearBuilt'].isnull()].index)
    
    # Clean the floor column
    df['floor'] = df['floor'].apply(cleanFloor)
    
    # Drop all rows where the floor is above 36 (the highest in stockholm)
    df = df.drop(df[df['floor'] > 36].index)
    
    # Drop all rows where the yearBuilt is below 1850 or above 2021
    df = df.drop(df[df['yearBuilt'] < 1850].index)
    
    # Drop all rows where the date is before 2012-01-01
    df = df.drop(df[df['soldDate'] < '2012-01-01'].index)
    
    df['streetName'] = df['streetName'].apply(cleanAddress)

    # inspectData(df)
    percent = abs(int((len(df) - rowsBefore) / rowsBefore * 100))
    print(f'Rows removed: {percent}%')
    return df

def cleanFloor(x):
    x = x.replace('tr', '')
    if x == 'BV':
        return 0
    elif '½' in x:
        x = x.replace('½', '.5')
        return float(x)
    else:
        return float(x)

def cleanAddress(x):
    # Remove "-" from the street
    x = ''.join(x.split('-'))
    # Remove all zero width spaces, non-breaking spaces and non-breaking hyphens
    x = x.replace('\u200b', '')
    x = x.replace('\u00a0', '')
    x = x.replace('\u2011', '')
    # Remove all soft hyphens
    x = x.replace('\xad', '')
    x = x.replace('\u200c', '')

    x.strip()
    return x

def addCoordinates(df):
    print('Adding missing coordinates...')
    # Extract the rows where the coordinates are missing (equal to 1000.0)
    missingCoordsDf = df[(df['lat'] == 1000.0) | (df['lon'] == 1000.0)]
    addrToCoords = {} # Amazing
    for row in tqdm(missingCoordsDf.itertuples()):
        # Use your own Nominatim server!!! The throttling of the public one is extremely stric
        coords = addrToCoords.get(row.streetName + str(row.number))

        if coords is not None:
            df.at[row.Index, 'lat'] = coords[0]
            df.at[row.Index, 'lon'] = coords[1]
            continue

        lat, lon = getCoordinatesFromAddress(row.streetName, row.number)
        df.at[row.Index, 'lat'] = lat
        df.at[row.Index, 'lon'] = lon

        addrToCoords[row.streetName + str(row.number)] = (lat, lon)
    
    return df

def inspectData(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    # Inspect the floor column
    floorInfo = df['yearBuilt'].value_counts()
    # Sort the floor by value
    floorInfo = floorInfo.sort_index()
    print(floorInfo)
    
    # Make a histogram of the floor column using seaborn
    # sns.histplot(df['floor'], kde=True)
    # plt.show()
    
    # Make a histogram of the yearBuilt column using seaborn
    sns.histplot(df['yearBuilt'], kde=True)
    plt.show()
    
# Adds the financial data to the apartment data
def populateApartmentData(aptDf, gdpDf, unemploymentDf, interestRateDf):
    print('Populating with financial data...')
    gdpDf = interpolateTime(gdpDf)
    unemploymentDf = interpolateTime(unemploymentDf)
    interestRateDf = interpolateTime(interestRateDf)
    aptDf['gdp'] = aptDf['soldDate'].apply(getValueFromTime, args=(gdpDf,))
    aptDf['unemployment'] = aptDf['soldDate'].apply(getValueFromTime, args=(unemploymentDf,))
    aptDf['interestRate'] = aptDf['soldDate'].apply(getValueFromTime, args=(interestRateDf,))
    return aptDf
    
def interpolateTime(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.resample('MS').mean()
    df = df.interpolate(method='time')
    return fixChange(df)

def getValueFromTime(datetime, dataDf):
    # Get the value from the dataDf at the given datetime
    # If the datetime is not in the dataDf, print the datetime and return '0'
    # First, set the day of the datetime to the first day of the month
    datetime = datetime.replace(day=1)
    try:
        return dataDf.loc[datetime, 'value']
    except KeyError:
        # Try adding one month
        nextMonth = datetime.month + 1
        if nextMonth > 12:
            datetime = datetime.replace(month=1)
            datetime = datetime.replace(year=datetime.year + 1)
            
def fixChange(df):
    # Set change to be the difference between the current and previous price
    df['change'] = df['value'].diff()
    # If the change is Nan set it to 0
    df['change'] = df['change'].fillna(0)
    
    return df