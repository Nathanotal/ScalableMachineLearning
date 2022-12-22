import hopsworks
from datasets import load_dataset, DatasetDict
import pandas as pd
import os

PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    # downloadData()
    data = loadData()
    # dataset = prepareForWrite(data)
    # uploadToHopsworks(project, dataset)



# Download / Load all data which we want to turn into features
def loadData(download=True):
    if download:
        downloadData()
    # We likely want to load 4 separate csv files and concatenate them into a single DatasetDict
    apartmentDf = pd.read_csv(f'{PATH}/data/apartmentData.csv', sep=';')
    gdpDf = pd.read_csv(f'{PATH}/data/gdpData.csv', sep=';')
    unemploymentDf = pd.read_csv(f'{PATH}/data/unemploymentData.csv', sep=';')
    interestRateDf = pd.read_csv(f'{PATH}/data/interestRateData.csv', sep=';')
    
    cleanApartmentDf = cleanData(apartmentDf)
    
    df = None # populateApartmentData(cleanApartmentDf, gdpDf, unemploymentDf, interestRateDf)
    
    # 
    # dataset = load_dataset('csv', data_files='combinedV1.csv')
    return df

def cleanData(df): # TODO: Clean data
    inspectData(df)
    return None

def inspectData(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    pass

# Adds the financial data to the apartment data
def populateApartmentData(aptDf, gdpDf, unemploymentDf, interestRateDf):
    pass

def downloadData():
    # 993jhbhPecCt6fS5.gvlZik4edWefbGbguZVwrES34rJrBQuaUBpHcJapmRlD6UseqKirncAUSNBOCTBq
    project = hopsworks.login()
    pass

def prepareForWrite(dataset): # TODO: Likely remove
    train_dataset, validation_dataset= dataset['train'].train_test_split(test_size=0.1).values() # TODO: wtf is this?
    dataset = DatasetDict({'train': train_dataset, 'val': validation_dataset})
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
