import csv
import os
import requests
from tqdm import tqdm
from concurrent import futures
from bs4 import BeautifulSoup
import utilityFunctions as uf
import time
import bulkScrape as bs

def main():
    # Load the current raw data
    print('Loading data...')
    df = uf.loadData()
    
    print('Checking for new data...')
    # Check the latest sold date
    latestSoldDate = df['soldDate'].max()

    # Get all appartments that have been sold since the latest sold date
    linksToScrape = getLinksSinceDate(latestSoldDate) # TODO

    # Make sure none of the links are already in the data
    linksToScrape = uf.getLinksWithoutData(linksToScrape)

    print(f'Found {len(linksToScrape)} new appartments. Scraping...')
    # Scrape the new data
    # bs.getAndWirteData(linksToScrape)

def getLinksSinceDate(latestSoldDate):
    links = bs.getAllLinks(latestSoldDate)   
    return links

if __name__ == '__main__':
    main()

