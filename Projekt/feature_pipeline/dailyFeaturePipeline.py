import utilityFunctions as uf
import featurePipeline as fp
from tqdm import tqdm
import numpy as np

# We overfetch a bit just to be sure we get all the new data
def main():
    print('Downloading data...')
    uf.downloadData('apartmentData.csv')
    print('Downloading existing features...')
    uf.downloadData('features.csv')
        
    # Load the current raw data
    print('Loading data...')
    rawData = uf.loadData()
    currentFeaturesDf = uf.loadFeatures()

    # Make a dict with address to coords
    print('Mapping coordinates...')
    addrToCoords, streetToCoords = mapCoordinates(currentFeaturesDf)
        
    # Map the data to the features
    print('Adding known coordinates...')
    rawData = addKnownCoordinates(rawData, addrToCoords, streetToCoords)

    # Clean the data
    fp.main(rawData)

    print('Uploading features...')
    uf.uploadData('features.csv')

def mapCoordinates(currentFeaturesDf):
    addrToCoords = {}
    streetToCoords = {} # For addresses without a number
    # Remove duplicates
    currentFeaturesDf = currentFeaturesDf.drop_duplicates(subset=['streetName', 'number'])
    for _,row in tqdm(currentFeaturesDf.iterrows()):
        addr = row['streetName'] + str(row['number'])
        addrToCoords[addr] = (row['lat'], row['lon'])
        streetToCoords[row['streetName']] = (row['lat'], row['lon'])
    return addrToCoords, streetToCoords

def addKnownCoordinates(rawData, addrToCoords, streetToCoords):
    addedCoords = 0
    for row in tqdm(rawData.itertuples()):
        streetName = fp.cleanAddress(str(row.streetName))
        number = row.number

        coords = addrToCoords.get(streetName + str(row.number))

        if np.isnan(number) or number < 1:
            coords = streetToCoords.get(streetName)
        
        if coords is not None:
            rawData.at[row.Index, 'lat'] = coords[0]
            rawData.at[row.Index, 'lon'] = coords[1]
            addedCoords += 1
        else: # Likely redundant
            rawData.at[row.Index, 'lat'] = 1000.0 # Not fantastic, I can't be bothered to do it properly
            rawData.at[row.Index, 'lon'] = 1000.0
    
    print(f'Of {len(rawData)} apartments, {addedCoords} had known coordinates')
    return rawData


if __name__ == '__main__':
    main()

