from bs4 import BeautifulSoup
import requests
import csv
import os
import re
import pandas as pd
from hopsworksTransfer import download, upload

PATH = os.path.dirname(os.path.abspath(__file__))

def firstDateLarger(date1, date2):
    # Compare two dates in the format YYYY-MM-DD
    # Return True if date1 is older (or the same) than date2
    # Return False if date1 is newer than date2

    # Split the dates into lists
    date1 = date1.split('-')
    date2 = date2.split('-')

    for t1, t2 in zip(date1, date2):
        if int(t1) < int(t2):
            return False
        elif int(t1) > int(t2):
            return True

    return True

def getNoResults(stadsDel):
    # Return the number of results for a search query
    response = requests.get(
        f'https://www.booli.se/slutpriser/{stadsDel}?objectType=L%C3%A4genhet&page=1')
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get div where class is only mt-4
    div = soup.find('div', class_='mt-4')

    # Get span with class font-semibold in div
    noApts = div.find('span', class_='font-semibold')

    # Get the text from the span
    no = noApts.text

    # Parse the text to an integer
    no = no.replace('\xa0', ' ')
    noInt = int(no.replace(' ', ''))

    return noInt


def cleanAddress(address):
    # Separate the number and the street name and remove every - and all spaces
    number = getNumberFromText(address)
    address = address.replace(number, '')
    address = address.replace('-', '')
    address = address.replace(' ', '')
    return address, number


def cleanPrice(price):
    # Remove all spaces and the kr
    price = price.replace(' ', '')
    price = price.replace('kr', '')
    return price


def cleanSqm(sqm):
    # Remove all spaces and the m²
    sqm = sqm.replace(' ', '')
    sqm = sqm.replace('m²', '')
    return sqm


def cleanAvgift(avgift):
    # Remove all spaces and the kr/mån
    avgift = avgift.replace(' ', '')
    avgift = avgift.replace('kr/mån', '')
    return avgift


def cleanDiff(diff):
    # Example of a diff: ±0 kr (+/-0 %)
    # Example of a negative diff: -55 000 kr (-1.4 %)
    # Example of a positive diff: +55 000 kr (+1.4 %)
    # Remove all of the spaces, the kr and everything in the parentheses
    try:
        diff = diff.split('(')[0]
    except:
        pass
    diff = diff.replace('±', '')
    diff = diff.replace('kr', '')
    diff = diff.replace(' ', '')

    return diff


def cleanPriceSqm(priceSqm):
    # Remove all spaces and the kr/m²
    priceSqm = priceSqm.replace(' ', '')
    priceSqm = priceSqm.replace('kr/m²', '')
    return priceSqm


def getStadsDelar():
    with open(f'{PATH}/stadsdelar.txt', 'r') as f:
        stadsDelar = f.read().splitlines()
    return stadsDelar


def getNumberFromText(text):
    # Return a list of numbers from a string
    numbers = re.findall(r'\d+', text)
    if len(numbers) == 0:
        return ''
    else:
        return numbers[0]


def saveLinks(allLinks):
    with open(f'{PATH}/data/links.csv', 'w') as f:
        f.write('link\n')
        for link in allLinks:
            f.write(link + '\n')


def loadLinks(fileNo):
    with open(f'{PATH}/data/split/links_{fileNo}.csv', 'r') as f:
        allLinks = f.read().splitlines()
    return allLinks[1:]

def writeToCsv(data, writer, headers):
    row = []
    for header in headers:
        value = ''
        try:
            value = data[header]
        except KeyError:
            pass
        row.append(value)
    writer.writerow(row)


def checkDifferenceInData(data, headers):
    # If there is an attribute which is not in the dictionary, throw an error and inspect the data
    diff = set(data.keys()) - set(headers)
    if len(diff) > 0:
        print('Error, there is a new attribute: \n')
        print(diff)


def splitCSVIntoNFiles(fileName, n):
    # Split a csv file into n files
    with open(f'{PATH}/data/{fileName}.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # Calculate the number of rows in each file
    noRows = len(data)
    noRowsPerFile = noRows // n
    for i in range(n):
        # Write the header
        with open(f'{PATH}/data/split/{fileName}_{i+1}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data[0])

        # Write the rows
        with open(f'{PATH}/data/split/{fileName}_{i+1}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for j in range(noRowsPerFile):
                toWrite = data[i * noRowsPerFile + j + 1]
                writer.writerow(toWrite)


def getLinksWithoutData(links):
    # Get all links which are not in the data csv file
    linksWithoutData = []
    linksWithData = {}
    # If there is no apartmentData.csv return 
    try:
        with open(f'{PATH}/data/apartmentData.csv', 'r') as f:
            pass
    except FileNotFoundError:
        return links
    
    with open(f'{PATH}/data/apartmentData.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = row[0].split(';')
            link = row[0][2:]
            if link == 'link':
                continue
            linksWithData[link] = True

    for link in links:
        try:
            exists = linksWithData[link]
            if exists is None:
                linksWithoutData.append(link)
        except KeyError:
            linksWithoutData.append(link)

    return linksWithoutData
# splitCSVIntoNFiles('links', 13)

def loadData():
    # Load all data from the csv files
    df = pd.read_csv(f'{PATH}/data/apartmentData.csv', sep=';')
    return df

def loadFeatures():
    # Load all data from the csv files
    df = pd.read_csv(f'{PATH}/data/features.csv', sep=';')
    return df

def downloadData(fileName): # TODO: Fix
    download(fileName)

def uploadData(fileName):
    upload(fileName)

def dropApartmets(df, linksWhichAreUpdated):
    # Drop all appartments with links which are updated
    # Create a series that identifies the links after the first "-"
    links = df['link'].str.split('-').str[1]

    # Identify the indexes where the link is not equal to 'link' and is in linksWhichAreUpdated
    indexesToDrop = df.index[(df['link'] != 'link') & (links.isin(linksWhichAreUpdated))].tolist()
        
    df = df.drop(indexesToDrop)

    # Write the data to a csv file
    df.to_csv(f'{PATH}/data/apartmentData.csv', sep=';', index=False) 
