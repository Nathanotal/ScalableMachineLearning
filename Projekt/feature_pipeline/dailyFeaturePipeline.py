import utilityFunctions as uf
import featurePipeline as fp
import pandas as pd

# We overfetch a bit just to be sure we get all the new data
def main():
    print('Downloading data...')
    # uf.downloadData('apartmentData.csv')
    print('Downloading existing features...')
    # uf.downloadData('features.csv')
        
    # Load the current raw data
    print('Loading data...')
    rawData = uf.loadData()
    currentFeatures = uf.loadFeatures()
    
    # Map the data to the features
    ...

    # Fill in coordinates 
    ...

    # Clean the data
    ...

    print('Uploading features...')
    # uf.uploadData('features.csv')




if __name__ == '__main__':
    main()

