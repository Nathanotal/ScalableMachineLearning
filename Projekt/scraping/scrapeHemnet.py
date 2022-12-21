# For data gathering
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import csv
import sys
import os
import time
import requests
import pandas as pd
import datetime
from tqdm import tqdm
from concurrent import futures
from bs4 import BeautifulSoup

PATH = os.path.dirname(os.path.abspath(__file__))
CHROMEDRIVERPATH = PATH + '/chromedriver.exe'
STADSDELAR_PATH = PATH + '/stadsdelar.txt'
chrome_options = Options()
chrome_options.add_argument("disable-popup-blocking")


class BostadsData:
    def __init__(self) -> None:
        pass
    

def main():
    LOAD = False
    links = None
    if LOAD:
        df = pd.read_csv('vasastan.csv')
        links = df['link'].tolist()
    else:
        links = getAllLinks()
        
    # Get the data for each link
    for link in tqdm(links):
        pass
        

def getStadsDelar():
    with open(STADSDELAR_PATH, 'r') as f:
        stadsDelar = f.read().splitlines()
    return stadsDelar
     
def getAllLinks():
    stadsDelar = getStadsDelar()
    allLinks = []
    
    for stadsDel in tqdm(stadsDelar):
        allLinks.extend(getLinksFromSearchPage(stadsDel))

    # Save all links as a CSV file
    with open('links.csv', 'w') as f:
        f.write('link')
        for link in allLinks:
            f.write(link + '\n')
        

def getNoResults(stadsDel):
    # Return the number of results for a search query
    response = requests.get(f'https://www.booli.se/slutpriser/{stadsDel}?objectType=L%C3%A4genhet&page=1')
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get the span with class "p3oVj"
    noApts = soup.find('span', class_='p3oVj')
    
    # Get the text from the span
    no = noApts.text
    
    # Parse the text to an integer
    noInt = int(no.replace(' ', ''))
    
    return noInt
       
def getLinksFromSearchPage(stadsDel):
    # Return a list of links to all pages of the search result
    noResults = getNoResults(stadsDel)
    noLoops = noResults // 35 + 1
    houseLinks = []
    for pageNo in range(noLoops):
        response = requests.get(f'https://www.booli.se/slutpriser/{stadsDel}?objectType=L%C3%A4genhet&page={pageNo+1}')
        
        if response.status_code != 200:
            print('Failed to get page: ', pageNo)
            print('Error: ', response.status_code)
            print(stadsDel)
        
        # Parse the HTML and save all links (href) in anchor tags with class "_3xhWw"
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links with class "_3xhWw"
        links = soup.find_all('a', class_='_3xhWw')
        
        
        # Get the href attribute from each link
        hrefs = [link.get('href') for link in links]
        
        # Add the hrefs to the list of house links
        houseLinks.extend(hrefs)
    
    return houseLinks
    
    
    
    




if __name__ == "__main__":
    main()
