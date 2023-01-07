# import utilityFunctions as uf
import hopsworks
import os

PATH = os.path.dirname(os.path.abspath(__file__))
FILENAME = "linksTest.csv"
FILENAME2 = "linksTest2.csv"
HW_PATH = "/Projects/nathanotal/sthlmData/"
HW_DL_PATH = f"/Projects/nathanotal/sthlmData/{FILENAME}"
LOCAL_PATH = os.path.join(PATH, f"data\\{FILENAME}")
LOCAL_PATH2 = os.path.join(PATH, "data")

HW = 'OWXnoeaQ1Bg6I0IE.EgaQo2HmubMIzfChCahCK6sQVLs4vyrhj2ODWHcYr0RN9f1gqac2dJjn8p2fXwcQ' # Don't do this lol

def main():
    project = hopsworks.login(api_key_value=HW)

    dataset_api = project.get_dataset_api()

    # upload(dataset_api)
    download(dataset_api)

def download(dataset_api):
    # Download data from Hopsworks
    downloaded_file_path = dataset_api.download(
        HW_DL_PATH, 
        local_path = LOCAL_PATH2, overwrite=True)
    print('The file has been succesfully downloaded from Hopsworks and is available at:' + '\n' + downloaded_file_path)


def upload(dataset_api):
    path1 = dataset_api.upload(
        local_path = LOCAL_PATH, 
        upload_path = HW_PATH + '/' + FILENAME, overwrite=True)
    print('The file has been succesfully uploaded to Hopsworks and is available at:' + '\n' + path1)

main()