import csv
import os
from geopy.geocoders import Nominatim
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
HOST_ADDRESS = '165.227.162.37'
HOST_PORT = '8080'
EMAIL = ''
DOMAIN = HOST_ADDRESS + ':' + HOST_PORT
LOCATOR = Nominatim(user_agent=EMAIL, domain=DOMAIN, scheme='http', timeout=10)

def getCoordinatesFromAddress(streetName, number):
    number = str(int(float(number)))
    address = f'{streetName} {number}, Stockholm'
    
    if number == '0':
        address = f'{streetName}, Stockholm'
        
    location = LOCATOR.geocode(address)
    
    if location is None:
        return pd.Series([0, 0])
    else:
        # Return with a precision of 6 decimals (accuracy of <1 meter)
        lat = round(location.latitude, 6)
        lon = round(location.longitude, 6)
        return pd.Series([lat, lon])