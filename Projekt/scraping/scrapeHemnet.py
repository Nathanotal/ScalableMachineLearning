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
           'Slutpris/m²', 'Dagar på Booli', 'Avgift', 'Bostadstyp', 'Driftskostnad', 'Våning', 'Byggår', 'BRF', 'brfLink', 'Energiklass']
LOAD_DATA = True
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
LINK_SPLIT = 3


def main():
    print('Initializing...')
    links = None
    if LOAD_DATA:
        links = uf.loadLinks(LINK_SPLIT)
    else:
        input('Are you sure you want to scrape all data? Press enter to continue...')
        links = getAllLinks()
        uf.saveLinks(links)

    getAndWirteData(links)
    print('Done!')


def getAndWirteData(links):
    with open(f'{PATH}/data/apartmentData.csv', 'w', newline='', encoding='utf-8') as f:
        # Make a writer which can handle nordic characters
        writer = csv.writer(f, delimiter=';', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADERS)

        # Write the data for each apratment
        print('Getting data from pages...')
        for link in tqdm(links):  # This can't be done in parallell because of 429
            data = getAttributesFromPage(link)
            if data is not None:
                # uf.checkDifferenceInData(data, HEADERS) # Debug
                uf.writeToCsv(data, writer, HEADERS)


def getAllLinks():
    stadsDelar = uf.getStadsDelar()
    allLinks = []

    print('Getting links to apartments...')
    for stadsDel in tqdm(stadsDelar):
        allLinks.extend(getLinksFromSearchStadsDel(stadsDel))

    return allLinks


def getLinksFromSearchStadsDel(stadsDel):
    # Return a list of links to all pages of the search result
    noResults = uf.getNoResults(stadsDel)
    noLoops = noResults // 35 + 1
    houseLinks = []

    # This should be done in parrallel
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
                pass

    return houseLinks


def getLinksFromSearchPage(link):
    response = requests.get(link)

    if response.status_code == 200:
        # Parse the HTML and save all links (href) in anchor tags with class "_3xhWw"
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links with class "_3xhWw"
        links = soup.find_all('a', class_='_3xhWw')

        # Get the href attribute from each link
        hrefs = [link.get('href') for link in links]

        return hrefs
    elif response.status_code == 429:
        # Too many requests, wait a minute and try again
        print('Too many requests, waiting 60 seconds...')
        time.sleep(60)
    else:
        print('Error: ', response.status_code)
        print('Failed to get links from search page: ', link)
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
            case 'Slutpris/m²':
                data[name] = uf.cleanPriceSqm(value)
            case 'Storlek' | 'Sista bud' | 'Tomtstorlek' | 'Upplåtelseform' | 'Sidvisningar' | 'Dagar som snart till salu' | 'Biarea' | 'Boendekostnad' | 'Kvadratmeterpris':
                pass
            case _:
                data[name] = value

    return data


def getAttributesFromPage(shortLink):
    # Return a dictionary with the attributes of a house
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
            data.update(getGeneralInfo(soup))
        except:
            pass
    elif response.status_code == 429:
        timeToWait = 5
        try:
            timeToWait = int(response.headers['Retry-After'])
        except:
            pass
        time.sleep(timeToWait)
        return getAttributesFromPage(shortLink)
    else:
        print('Error: ', response.status_code)

    return data


if __name__ == "__main__":
    main()
