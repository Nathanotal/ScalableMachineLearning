import gradio as gr
import numpy as np
from PIL import Image
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import hopsworks
from tqdm import tqdm
import xgboost as xgb
from geopy.geocoders import Nominatim
from datetime import date
from datetime import timedelta
from autogluon.tabular import TabularPredictor
import shutil

# Login to hopsworks and get the feature store

# streetName;number;sqm;rooms;soldDate;monthlyFee;monthlyCost;floor;yearBuilt;brf;agency;lat;lon;gdp;unemployment;interestRate
columnHeaders = ['streetName','number','sqm','rooms','soldDate','monthlyFee','monthlyCost','floor','yearBuilt', 'brf','agency','lat','lon'] # ,'gdp','unemployment','interestRate'

featureToMinMax = {
        'sqm': (10, 800),
        'rooms': (1, 20),
        'monthlyFee': (0, 60000),
        'monthlyCost': (0, 20000),
        'floor': (-3, 35),
        'yearBuilt': (1850, 2023),
        'lat': (58.8, 60.2),
        'lon': (17.5, 19.1),
        'gdp': (505.1, 630.14),
        'unemployment': (6.36, 8.66),
        'interestRate': (-0.5, 2.64),
        'number': (0, 300),
        'soldDate': (2010, 2025)
    } # Extracted from the data

featureToName = {
    'number' : 'Street number',
     'sqm' : 'Size of the apartment in square meters',
    'rooms' : 'Number of rooms',
     'monthlyFee' : 'Monthly fee',
    'monthlyCost' : 'Monthly operating cost',
    'floor' : 'Floor',
    'yearBuilt' : 'Year built',
     'streetName' : 'Name of street',
}

topAgencies = ['Fastighetsbyrån','Notar','Svensk Fastighetsförmedling','HusmanHagberg','Länsförsäkringar Fastighetsförmedling','Erik Olsson','SkandiaMäklarna','Svenska Mäklarhuset','Bjurfors','Mäklarhuset','BOSTHLM','Innerstadsspecialisten','MOHV','Mäklarringen','Historiska Hem','Södermäklarna','Karlsson & Uddare','UNIK Fastighetsförmedling','Edward & Partners','Widerlöv']

def downloadAutogluonModel():
    # Download saved Autogluon model from Hopsworks
    project = hopsworks.login() 
    mr = project.get_model_registry()
    temp = mr.get_model("ag_model_20230109", version=5)
    temp_ag_folder_path = temp.download()
    print(temp_ag_folder_path)
    moveFolder(temp_ag_folder_path)

    ag_model = TabularPredictor.load("AutogluonModels/ag_model_20230109") # '/ag_model_20230109'

    return ag_model


def moveFolder(temp_ag_folder_path):
    # Move Autogluon model folder to the correct folder
    original = temp_ag_folder_path
    target = "AutogluonModels/"
    shutil.move(original, target)

def downloadModel():
    # Download saved Autogluon model from Hopsworks 
    project = hopsworks.login() 
    mr = project.get_model_registry()
    temp = mr.get_model("xgboost_model", version=5)
    model_path = temp.download()

    xgb_model = joblib.load(model_path + "/xgboost_model.pkl")
    return xgb_model

def getAddressInfo(streetName, number):
    streetName = cleanAddress(streetName)
    try:
        return getCoordinatesFromAddress(streetName, number)
    except AddressNotFound:
        return None, None

# Adds the financial data to the apartment data
def populateApartmentData(aptDf):
    print('Populating with financial data...')
    gdpDf = pd.read_csv(f'./data/historicalGDP.csv', sep=';')
    unemploymentDf = pd.read_csv(f'./data/historicalUnemployment.csv', sep=';')
    interestRateDf = pd.read_csv(f'./data/historicalInterest.csv', sep=';')
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
    # parse datetime to enable replacement
    datetime = pd.to_datetime(datetime)
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

class AddressNotFound(Exception):
    pass

def getCoordinatesFromAddress(streetName, number):

    HOST_ADDRESS = '165.227.162.37'
    HOST_PORT = '8080'
    EMAIL = 'nathan.allard@gmail.com'
    DOMAIN = HOST_ADDRESS + ':' + HOST_PORT
    LOCATOR = Nominatim(user_agent=EMAIL, domain=DOMAIN, scheme='http', timeout=10)

    number = str(int(float(number)))
    address = f'{streetName} {number}, Stockholm'
    
    if number == '0':
        address = f'{streetName}, Stockholm'
        
    location = LOCATOR.geocode(address)
    
    if location is None:
        raise AddressNotFound
    else:
        # Return with a precision of 6 decimals (accuracy of <1 meter)
        lat = round(location.latitude, 6)
        lon = round(location.longitude, 6)
        return lat, lon

def dateToFloat(date):
    year, month, day = str(date).split('-')
    day = day.split(' ')[0]
    return int(year) + int(month) / 12 + int(day) / 365

def normalize(x, minVal, maxVal, feature):
    # Not fantastic
    res = (float(x) - minVal) / (maxVal - minVal)
    return min(max(res, 0), 1)

def normalizeData(df):
    # Normalize select numerical values to a value between 0 and 1
    print('Normalizing data...')
    for feature, minMax in tqdm(featureToMinMax.items()):
        min = minMax[0]
        max = minMax[1]
        if feature == 'soldDate':
            df[feature] = df[feature].apply(lambda x: dateToFloat(x))

        df[feature] = df[feature].apply(lambda x: normalize(x, min, max, feature))

    return df

def parsePrice(price):
    featureToMinMaxPrice = {
    'price': (1.5e5, 7e7)
    }
    MIN = featureToMinMaxPrice['price'][0]
    MAX = featureToMinMaxPrice['price'][1]

    price = float(price)
    price = price * (MAX - MIN) + MIN
    return f'{addDotsToPrice(int(price))} SEK'

def addDotsToPrice(price):
    # Takes an int like 1000000 and returns a string like 1.000.000
    toReturn = ''
    price = str(price)
    for i, c in enumerate(price):
        toReturn += c
        if (len(price) - i) % 3 == 1 and i != len(price) - 1 and c != '-':
            toReturn += '.'
    return toReturn
        
        

def xgbFix(df):
    features_to_categorical = ["streetName", "brf", "agency"]

    features_to_float = ["number", "sqm", "rooms", "monthlyFee",
                        "monthlyCost", "floor", "yearBuilt", "gdp", "unemployment",
                        "interestRate", "lat", "lon", "soldDate"]

    df[features_to_categorical] = df[features_to_categorical].astype("category")
    df[features_to_float] = df[features_to_float].astype(float)
    return df


model = downloadModel()
autoModel = downloadAutogluonModel()

def xgboostPred(df):
    # Drop categorical features
    df = df.drop(['streetName', 'brf', 'agency'], axis=1)

    # Save first row as a numpy array

    results = []
    for _,row in df.iterrows():
        input_list = row.to_numpy()
        res = model.predict(np.asarray(input_list).reshape(1, -1))
        results.append(res[0]) # This is not done in a good way

    return results

def addExtraAgencyFun(df):
    # Make 20 copies of the first row with the 20 different top agencies in Sweden
    # Make a copy of the first row
    firstRow = df.iloc[0]
    # Make a list of the copies
    rows = [firstRow] * len(topAgencies)
    # Make a dataframe from the list
    df2 = pd.DataFrame(rows)

    # Add the top agencies to the dataframe
    for i, agency in enumerate(topAgencies):
        df2['agency'].iloc[i] = agency
    
    # Concatenate the two dataframes
    df = pd.concat([df, df2], ignore_index=True)

    return df

def autoPred(df):
    df = addExtraAgencyFun(df)
    res = autoModel.predict(df)

    # Convert to a list
    res = res.tolist()

    # Get the last 20 values
    agencyResults = res[-20:]
    res = res[:-20]

    # Get the mean of the agencies
    agencyToResult = {agency:result for agency, result in zip(topAgencies, agencyResults)}
    for agency, result in agencyToResult.items():
        print(agency, str(result))

    # Get the top and bottom 3 agencies with the highest results
    sortedAgencies = sorted(agencyToResult.items(), key=lambda x: x[1])
    meanPrice = sum(agencyResults) / len(agencyResults)
    top3 = sortedAgencies[-5:]
    top3.reverse()

    agencyString = parseAgencyResult(top3, meanPrice)

    return res, agencyString

def parseAgencyResult(top3, meanPrice):
    toReturn = 'To get the most money for your apartment, you should sell it with the help of one of these agencies:\n'
    toReturn += 'Top 5:\n'
    for agency, result in top3:
        diff = result - meanPrice
        toReturn += f'{agency}: {parsePrice(result)} ({parsePrice(diff)} above mean)\n'

    return toReturn

def isValidInput(streetName, number, sqm, rooms, monthlyFee, monthlyCost, floor, yearBuilt):
    # Street name is a string, all other values are numbers
    if streetName == '':
        return 'Street name is empty'
    # If Street name contains numbers it should fail
    if any(char.isdigit() for char in streetName):
        return 'Only letters are allowed in street name'

    toCheck = [number, sqm, rooms, monthlyFee, monthlyCost, floor, yearBuilt]
    toCheckName = ['number', 'sqm', 'rooms', 'monthlyFee', 'monthlyCost', 'floor', 'yearBuilt']
    for val, name in zip(toCheck, toCheckName):
        MIN = featureToMinMax[name][0]
        MAX = featureToMinMax[name][1]
        if val < MIN:
            return f'{featureToName.get(name)} is too low'
        if val > MAX:
            return f'{featureToName.get(name)} is too high'
    
    return None

def getDates():
    today = date.today()
    # inAMonth = today + timedelta(days=30)
    inAYear = today + timedelta(days=365)
    lastYear = today - timedelta(days=365)
    beforeUkraineWar = '2022-02-24'
    threeYearsAgo = today - timedelta(days=365*3)

    dateToExplanation = {
        today.strftime("%Y-%m-%d") : 'today',
        # inAMonth.strftime("%Y-%m-%d") : 'in a month',
        inAYear.strftime("%Y-%m-%d") : 'in a year',
        lastYear.strftime("%Y-%m-%d") : 'last year',
        threeYearsAgo.strftime("%Y-%m-%d") : 'three years ago',
        beforeUkraineWar : 'before Russia invaded Ukraine',
    }

    return dateToExplanation


def sthlm(streetName, number, sqm, rooms, monthlyFee, monthlyCost, floor, yearBuilt, agency, auto):
    inputErrors = isValidInput(streetName, number, sqm, rooms, monthlyFee, monthlyCost, floor, yearBuilt)
    if inputErrors is not None:
        return '0', '', '', inputErrors
    lat, lon = getAddressInfo(streetName, number)
    # If none
    if lat is None or lon is None:
        return '0', '', '', 'Address not found in the OpenStreetMap dataset (Nominatim), please try another address'

    brf = 'BRF Kartboken 1' # Not used
    dates = getDates()
    input_variables = pd.DataFrame(
            columns=columnHeaders)
    
    for soldDate in dates.keys():
        # Parse the input so we can run it through the model
        # Create a dataframe from the input values
        input_variables = input_variables.append(
            pd.DataFrame(
                [[streetName,number,sqm,rooms,soldDate,monthlyFee,monthlyCost,floor,yearBuilt,brf,agency,lat,lon]], columns=columnHeaders))
    
    df = populateApartmentData(input_variables)  
    df = normalizeData(df)

    pricePred = None
    agencyInfo = 'Please use AutoGluon instead of XGBoost to get information about agencies'
    if auto:
        pricePred, agencyInfo = autoPred(df)
    else:
        df = xgbFix(df)
        pricePred = xgboostPred(df)

    explanations = list(dates.values())
    result = [] #
    mainPred = None
    mainExplanation = None
    for i, pred in enumerate(pricePred):
        explanation = explanations[i]
        if i == 0:
            mainExplanation = explanation
            mainPred = pred
        else:
            diff = pred - mainPred
            if diff > 0:
                result.append(f'If sold {explanation} it would have been worth more: {parsePrice(pred)} (+{parsePrice(diff)})')
            else:
                result.append(f'If sold {explanation} it would have been worth less: {parsePrice(pred)} ({parsePrice(diff)})')

            

    return f'Predicted price of the apartment {mainExplanation}: {parsePrice(mainPred)}', '\n'.join(result), agencyInfo, ''



# All features present in the sthlm dataset
numericalInputs = ['number', 'sqm','rooms', 'monthlyFee','monthlyCost','floor','yearBuilt']
inputs = [gr.inputs.Textbox(lines=1, label='streetName')]


    
# Generate the input form
for feature in numericalInputs:
    minVal = featureToMinMax[feature][0]
    maxVal = featureToMinMax[feature][1]
    theLabel = f'{featureToName.get(feature)} (min: {minVal}, max: {maxVal})'
    inputs.append(gr.inputs.Number(default=0, label=theLabel))

# Add a switch to choose between xgboost and autogluon
inputs.append(gr.inputs.Dropdown(label='Agency', choices=topAgencies, default='Notar'))
inputs.append(gr.inputs.Checkbox( label='Use AutoGluon instead of XGBoost', default=False))
# Create the interface
resultOutputs = [gr.outputs.Label(label='Price if sold today'), gr.outputs.Textbox(label='If sold at a different time'), gr.outputs.Textbox(label='Best agencies to use'), gr.outputs.Textbox(label='Error').style(color='red')]

demo = gr.Interface(
    fn=sthlm,
    title="Stockholm Housing Valuation",
    description="Predict the price of an apartment in Stockholm. To get information about which agency to use, please select AutoGluon",
    allow_flagging="never",
    inputs=inputs,
    outputs=resultOutputs)

demo.launch()
