# import utilityFunctions as uf
import hopsworks
import os

PATH = os.path.dirname(os.path.abspath(__file__))
HW_PATH = "/Projects/nathanotal/housingValuation"
LOCAL_PATH = os.path.join(PATH, "data")

HW = 'OWXnoeaQ1Bg6I0IE.EgaQo2HmubMIzfChCahCK6sQVLs4vyrhj2ODWHcYr0RN9f1gqac2dJjn8p2fXwcQ' # Don't do this lol
project = hopsworks.login(api_key_value=HW)
DATASET_API = project.get_dataset_api()

def download(fileName):
    # Download data from Hopsworks
    downloaded_file_path = DATASET_API.download(
        HW_PATH + '/' + fileName, 
        local_path = LOCAL_PATH, overwrite=True)
    print('The file has been succesfully downloaded from Hopsworks and is available at:' + '\n' + downloaded_file_path)


def upload(fileName):
    path1 = DATASET_API.upload(
        local_path = LOCAL_PATH + '/' + fileName, 
        upload_path = HW_PATH, overwrite=True)
    print('The file has been succesfully uploaded to Hopsworks and is available at:' + '\n' + path1)