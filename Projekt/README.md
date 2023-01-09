# Title
## Scraping pipeline
The scraping is really a part of the fetaure pipeline. However, as this process proved complicated we broke the scraping out to its own pipeline which generates a raw dataset. The pipeline scrapes Booli.se for apartment data. In order to do the first bulk scrape one can run bulkScrape.py. However, without several proxies this takes days because of very strict rate limiting. dailyScrape.py is designed to be run once a day as a cronjob. It loads the current raw data, then checks for new listings. If it finds any new listings which are not in the data, it adds them to a list and scrapes the new apartments. After it is done the data is uploaded to Hopsworks. Financial data regarding GPD, interest and unemployment was also scraped. However, a better alternative to gathering this data is to enter it manually as there are not a lot of datapoints.

## Feature pipeline
The feature pipeline downloads the data collected by the scraping pipeline and cleans it. The daily feature pipeline first loads the raw data and tries to download the current features if there are any. It then maps the coordinates from the current data to the new datapoints (if any new datapoints have been scraped) and then sends the raw data to the feature pipeline. A lot of operations are done on the data. For example, several outliers are excluded. Data with null values are either dropped or filled in. Several columns are parsed to make it possible for machine learning models to accept. The date is also converted to a numerical feature. Coordinates are also added to the datapoints which do not have them. The dataset is subsequently normalized and uploaded to Hopsworks.

#### Coordinates:
Importantly, the dataset from the website which was scraped did not include coordinates. As a result these had to be interpolated from the address of the apartment. This proved rather difficult. It would have cost ~$1000 to have identified the coordinates using the Google Maps API. We thus turned to OpenStreetMaps. We downloaded the dataset, however, it was really complicated to make a sufficient SQL query. Thus, we tried to use Nominatim (based on OpenStreetMaps), however they have a rate limit of 1 request/second and 2500 requests/day. Thus, we arrived at our final solution which entailed hosting our own Nominatim API. This API was first hosted on a private rpi, however, the dataset was too large and we had to migrate to a DigitalOcean droplet. With our own API we were able to bulk fetch all the data we needed!

## Training pipeline

### Models:
#### XGBoost
#### AutoGluon

## Inference pipeline (UI)
Blablabla
