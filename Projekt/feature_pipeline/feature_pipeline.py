# import hopsworks
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    # downloadData()
    data = loadData()
    dataset = prepareForWrite(data)
    
    # Inspect the dataset
    print(dataset)
    # uploadToHopsworks(project, dataset)



# Download / Load all data which we want to turn into features
def loadData(download=True):
    print('Loading data...')
    if download:
        downloadData()
    # We likely want to load 4 separate csv files and concatenate them into a single DatasetDict
    apartmentDf = pd.read_csv(f'{PATH}/data/apartmentData.csv', sep=';')
    gdpDf = pd.read_csv(f'{PATH}/data/historicalGDP.csv', sep=';')
    unemploymentDf = pd.read_csv(f'{PATH}/data/historicalUnemployment.csv', sep=';')
    interestRateDf = pd.read_csv(f'{PATH}/data/historicalInterest.csv', sep=';')
    
    cleanApartmentDf = cleanData(apartmentDf)
    
    df = populateApartmentData(cleanApartmentDf, gdpDf, unemploymentDf, interestRateDf)
    
    # dataset = load_dataset('csv', data_files='combinedV1.csv')
    return df

def cleanData(df): # TODO: Clean data
    print('Cleaning data...')
    rowsBefore = len(df)
    # Set index to the link
    df = df.set_index('link')
    
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
    
    # Where number is null, set it to 0
    df['number'] = df['number'].fillna(0)
    
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
    # Interpolate the GDP data so we have values for every month
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
            
        # Print the first and last date in the dataDf

def fixChange(df):
    # Set change to be the difference between the current and previous price
    df['change'] = df['value'].diff()
    # If the change is Nan set it to 0
    df['change'] = df['change'].fillna(0)
    return df

def downloadData():
    # 993jhbhPecCt6fS5.gvlZik4edWefbGbguZVwrES34rJrBQuaUBpHcJapmRlD6UseqKirncAUSNBOCTBq
    # project = hopsworks.login()
    pass

def prepareForWrite(df): # TODO: Likely remove
    print('Preparing for write...')
    # Convert the dataset to a DatasetDict
    train, test = train_test_split(df, test_size=0.2)
    train_dataset = Dataset.from_dict(train)
    test_dataset = Dataset.from_dict(test)
    dataset = DatasetDict({'train': train_dataset, 'val': test_dataset})
    return dataset

def uploadToHopsworks(project, dataset): # TODO: Make sthlm housing folders
    dataset.save_to_disk('dataset') # TODO: Understand how it is saved
    
    # Upload dataset to Hopsworks
    dataset_api = project.get_dataset_api()

    # Upload Dataset Dict
    path1 = dataset_api.upload(
        local_path = f'{PATH}/dataset/dataset_dict.json', 
        upload_path = '/Projects/nathanotal/sthlm_housing/', overwrite=True)

    # Upload train state
    path2 = dataset_api.upload(
        local_path = f'{PATH}/dataset/train/state.json', 
        upload_path = '/Projects/nathanotal/sthlm_housing/train/', overwrite=True)

    # Upload train info
    path3 = dataset_api.upload(
        local_path = f'{PATH}/dataset/train/dataset_info.json', 
        upload_path = '/Projects/nathanotal/sthlm_housing/train/', overwrite=True)

    # Upload test state
    path4 = dataset_api.upload(
        local_path = f'{PATH}/dataset/test/state.json', 
        upload_path = '/Projects/nathanotal/sthlm_housing/test/', overwrite=True)

    # Upload test info
    path5 = dataset_api.upload(
        local_path = f'{PATH}/dataset/test/dataset_info.json', 
        upload_path = '/Projects/nathanotal/sthlm_housing/test/', overwrite=True)

    # Upload test data
    path6 = dataset_api.upload(
        local_path = f'{PATH}/dataset/test/dataset.arrow', 
        upload_path = '/Projects/nathanotal/sthlm_housing/test/', overwrite=True)

    # # Upload train data
    path7 = dataset_api.upload(
        local_path = f'{PATH}/dataset/train/dataset.arrow', 
        upload_path = '/Projects/nathanotal/sthlm_housing/train/', overwrite=True)

    # Print the paths to the uploaded files
    print(path1, path2, path3, path4, path5, path6, path7)
    
    
if __name__ == '__main__':
    main()
