import csv
import os
import requests
from tqdm import tqdm
from concurrent import futures
from bs4 import BeautifulSoup
import utilityFunctions as uf
import time

PATH = os.path.dirname(os.path.abspath(__file__))
HEADERS = ['link', 'area', 'streetName', 'number', 'sqm', 'rooms', 'Slutpris', 'Utropspris', 'Prisutveckling', 'Såld eller borttagen',
           'Slutpris/m²', 'Dagar på Booli', 'Avgift', 'Bostadstyp', 'Driftskostnad', 'Våning', 'Byggår', 'BRF', 'brfLink', 'Energiklass', 'agency']
LOAD_DATA = True
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
LINK_SPLIT = 1

class DatePassedException(Exception):
    "Raised when the date is above the latest date"
    pass

def main():
    print('Initializing...')
    links = None
    if LOAD_DATA:
        links = uf.loadLinks(LINK_SPLIT)
    else:
        input('Are you sure you want to scrape all data? Press enter to continue...')
        links = getAllLinks()
        uf.saveLinks(links)
    
    linksLeftToScrape = uf.getLinksWithoutData(links)

    getAndWirteData(linksLeftToScrape)
    print('Done!')


def getAndWirteData(links):
    # If you want to run this in parralell, you need to run it through proxies as there is a strict rate limit
    with open(f'{PATH}/data/apartmentData.csv', 'a', newline='', encoding='utf-8') as f:
        # Make a writer which can handle nordic characters
        writer = csv.writer(f, delimiter=';', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        if not LOAD_DATA:
            writer.writerow(HEADERS) 

        # Write the data for each apratment
        print('Getting data from pages...')
        for link in tqdm(links):  # This can't be done in parallell because of 429
            data = getAppartmentsFromPage(link)
            if data is not None:
                for bostad in data:
                    uf.writeToCsv(bostad, writer, HEADERS)
                    # uf.checkDifferenceInData(data, HEADERS) # Debug


def getAllLinks(date=None):
    stadsDelar = uf.getStadsDelar()
    allLinks = []

    print('Getting links to apartments...')
    for stadsDel in tqdm(stadsDelar):
        allLinks.extend(getLinksFromSearchStadsDel(stadsDel, date))

    # Return only unique links
    return list(set(allLinks))


def getLinksFromSearchStadsDel(stadsDel, lastSoldDate):
    # Return a list of links to all pages of the search result
    noResults = uf.getNoResults(stadsDel)
    noLoops = noResults // 35 + 1
    houseLinks = []

    if lastSoldDate is None:
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for pageNo in range(noLoops):
                try:
                    link = f'https://www.booli.se/slutpriser/{stadsDel}?objectType=L%C3%A4genhet&page={pageNo+1}'
                    res = executor.submit(getLinksFromSearchPage, link).result()
                    if res is not None:
                        houseLinks.extend(res)
                    else:
                        print('Err', res)
                except:
                    print(f'Major error in {stadsDel} page {pageNo+1}')
    else:
        for pageNo in range(noLoops):
            try:
                link = f'https://www.booli.se/slutpriser/{stadsDel}?objectType=L%C3%A4genhet&page={pageNo+1}'
                res = getLinksFromSearchPageAfterDate(link, lastSoldDate)
                if res is not None:
                    houseLinks.extend(res)
                else:
                    print('Err', res)
            except DatePassedException:
                break
            # except:
            #     print(f'Major error in {stadsDel} page {pageNo+1}')


    return houseLinks

def getLinksFromSearchPageAfterDate(link, date):
    # Overfetch a bit because I am lazy
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all elements with class "fWha_"
    dateElements = soup.find_all('p', class_='fWha_')

    if response.status_code == 200:
        # If the date of the first element is larger than the latest date we have data for, stop
        dateLessThanExists = False
        for dateElement in dateElements:
            if uf.firstDateLarger(dateElement.text, date):
                dateLessThanExists = True
                break

        if not dateLessThanExists:
            raise DatePassedException
            
        return getHrefs(response)
    elif response.status_code == 429:
        # Too many requests, wait a minute and try again
        print('Too many requests, waiting 60 seconds...')
        time.sleep(60)
    else:
        print('Error: ', response.status_code)
        print('Failed to get links from search page')
        return []
    

    getHrefs(response)

def getHrefs(response):
    # Parse the HTML and save all links (href) in anchor tags with class "_3xhWw"
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links with class "_3xhWw"
    links = soup.find_all('a', class_='_3xhWw')

    # Get the href attribute from each link
    return [link.get('href') for link in links]
    

def getLinksFromSearchPage(link):
    response = requests.get(link)
    if response.status_code == 200:
        return getHrefs(response)
    elif response.status_code == 429:
        # Too many requests, wait a minute and try again
        print('Too many requests, waiting 60 seconds...')
        time.sleep(60)
    else:
        print('Error: ', response.status_code)
        print('Failed to get links from search page')
        return []


def getHeadInfo(soup):
    addressAndPriceClass = 'lzFZY _10w08'
    sqmAndMoreClass = '_1544W _10w08'

    # Get the first h1 tag with addressAndPriceClass (the price)
    address = soup.find('h1', class_=addressAndPriceClass).text

    # Get all the h4 tags with sqmAndMoreClass (the type and area)
    sqmRoomsArea = soup.find_all('h4', class_=sqmAndMoreClass)

    sqmAndRooms = ''
    area = ''
    for index, h4 in enumerate(sqmRoomsArea):
        if index == 0:
            sqmAndRooms = h4.text
        elif index == 1:
            area = h4.text.split(',')[1]
        else:
            print('Error: More than 2 h4 tags with class sqmAndMoreClass')
            print(sqmRoomsArea)

    # Get the second h4 with sqmAndMoreClass (the type and area)
    sqmRoom = sqmAndRooms.split(',')
    sqm = uf.getNumberFromText(sqmRoom[0])
    rooms = uf.getNumberFromText(sqmRoom[1])

    streetName, number = uf.cleanAddress(address)

    data = {'sqm': uf.cleanSqm(sqm), 'rooms': rooms, 'area': area.replace(
        ' ', ''), 'streetName': streetName, 'number': number}

    return data


def getGeneralInfo(soup):
    infoDivClass = '_36W0F mz1O4'
    attributeClass = 'DfWRI _1Pdm1 _2zXIc sVQc-'
    nameClass = '_2soQI'
    valueClass = '_18w8g'

    # Find the div with the info
    infoDiv = soup.find('div', class_=infoDivClass)

    # Find all attributes
    attributes = infoDiv.find_all('div', class_=attributeClass)

    # Get the name and value for each attribute
    data = {}
    dataToReturn = []
    for attribute in attributes:
        name = attribute.find('div', class_=nameClass).text
        value = attribute.find('div', class_=valueClass).text
        # Switch on the name
        match name:
            case 'BRF':
                # If the attribute is a BRF, get the link to the page
                link = attribute.find('a').get('href')
                data['brfLink'] = link
                data[name] = value
            case 'Slutpris' | 'Utropspris':
                data[name] = uf.cleanPrice(value)
            case 'Prisutveckling':
                data[name] = uf.cleanDiff(value)
            case 'Avgift' | 'Driftskostnad':
                data[name] = uf.cleanAvgift(value)
            case 'Sista bud':
                data['Slutpris'] = uf.cleanPrice(value)
            case 'Dagar på Booli' | 'Slutpris/m²' | 'Storlek' | 'Tomtstorlek' | 'Upplåtelseform' | 'Sidvisningar' | 'Dagar som snart till salu' | 'Biarea' | 'Boendekostnad' | 'Kvadratmeterpris' | 'Såld eller borttagen':
                pass
            case _:
                data[name] = value
    
    # Get all other events
    for index, event in enumerate(getHistory(soup)):
        newData = data.copy()
        newData.update(event)
        if index > 0:
            newData['Utropspris'] = ''
            newData['Prisutveckling'] = ''
        
        dataToReturn.append(newData)

    return dataToReturn

def getHistory(soup):
    historyItemClass = '_10hNQ DfWRI'
    priceClass = '-BWPP _38tTw _33jtR'
    dateClass = '-BWPP _33jtR'
    agencyClass = '_33jtR -BWPP'
    price = ''
    date = ''
    agency = ''
    events = []
    for historyItem in soup.find_all('div', class_=historyItemClass):
        # Get the price, date, agency
        try:
            price = uf.cleanPrice(historyItem.find('span', class_=priceClass).text)
            if price is None:
                price = ''
        except:
            price = ''
        try:
            date = historyItem.find('span', class_=dateClass).text
            if date is None:
                date = ''
        except:
            date = ''
        try:
            agency = historyItem.find('span', class_=agencyClass).text
            if agency is None:
                agency = ''
        except:
            agency = ''
        
        newEvent = {'Slutpris': price, 'Såld eller borttagen': date, 'agency': agency}
        events.append(newEvent)
        
    
    return events
        


def getAppartmentsFromPage(shortLink):
    # Return a dictionary with the attributes of a house
    dataToReturn = []
    data = {'link': shortLink}
    link = f'https://www.booli.se{shortLink}'
    response = requests.get(link)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            data.update(getHeadInfo(soup))
        except:
            pass
        try:
            for index, bostad in enumerate(getGeneralInfo(soup)):
                newData = data.copy()
                newData.update(bostad)
                newData['link'] = str(index) + '-' + shortLink
                dataToReturn.append(newData)
        except:
            pass
    elif response.status_code == 429:
        timeToWait = 5
        try:
            timeToWait = int(response.headers['Retry-After'])
        except:
            pass
        time.sleep(timeToWait)
        return getAppartmentsFromPage(shortLink)
    else:
        print('Error: ', response.status_code)

    return dataToReturn


if __name__ == "__main__":
    main()
