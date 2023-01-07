import utilityFunctions as uf
import bulkScrape as bs

# We overfetch a bit just to be sure we get all the new data
def main():
    # Load the current raw data
    print('Loading data...')
    df = uf.loadData()
    
    print('Checking for new data...')
    # Check the latest sold date
    # Drop the first row since it is the header
    latestSoldDate = '2000-01-01'
    for soldDate in df['Såld eller borttagen'][1:]:
        if soldDate == 'Såld eller borttagen':
            pass # The raw raw data is a bit messy
        elif uf.firstDateLarger(soldDate, latestSoldDate):
            latestSoldDate = soldDate
        
    print(latestSoldDate)

    # Get all appartments that have been sold at and since the latest sold date (and some more)
    linksToScrape = getLinksSinceDate(latestSoldDate)

    print(f'Found {len(linksToScrape)} potential new appartments. Scraping...')

    # Drop apartments which will get updated
    uf.dropApartmets(df, linksToScrape)
    
    # Scrape the new data
    bs.getAndWirteData(linksToScrape)

def getLinksSinceDate(latestSoldDate):
    links = bs.getAllLinks(latestSoldDate)   
    return links

if __name__ == '__main__':
    main()

