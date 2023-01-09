# Title
## Scraping pipeline
The scraping is really a part of the fetaure pipeline. However, as this process proved complicated we broke the scraping out to its own pipeline which generates a raw dataset. The pipeline scrapes Booli.se for apartment data. In order to do the first bulk scrape one can run bulkScrape.py. However, without several proxies this takes days because of very strict rate limiting. dailyScrape.py is designed to be run once a day as a cronjob. It loads the current raw data, then checks for new listings. If it finds any new listings which are not in the data, it adds them to a list and scrapes the new apartments. After it is done the data is uploaded to Hopsworks. Financial data regarding GPD, interest and unemployment was also scraped. However, a better alternative to gathering this data is to enter it manually as there are not a lot of datapoints.

## Feature pipeline
The feature pipeline downloads the data collected by the scraping pipeline and cleans it. The daily feature pipeline first loads the raw data and tries to download the current features if there are any. It then maps the coordinates from the current data to the new datapoints (if any new datapoints have been scraped) and then sends the raw data to the feature pipeline. A lot of operations are done on the data. For example, several outliers are excluded. Data with null values are either dropped or filled in. Several columns are parsed to make it possible for machine learning models to accept. The date is also converted to a numerical feature. Coordinates are also added to the datapoints which do not have them. The dataset is subsequently normalized and uploaded to Hopsworks.

#### Coordinates:
Importantly, the dataset from the website which was scraped did not include coordinates. As a result these had to be interpolated from the address of the apartment. This proved rather difficult. It would have cost ~$1000 to have identified the coordinates using the Google Maps API. We thus turned to OpenStreetMaps. We downloaded the dataset, however, it was really complicated to make a sufficient SQL query. Thus, we tried to use Nominatim (based on OpenStreetMaps), however they have a rate limit of 1 request/second and 2500 requests/day. Thus, we arrived at our final solution which entailed hosting our own Nominatim API. This API was first hosted on a private rpi, however, the dataset was too large and we had to migrate to a DigitalOcean droplet. With our own API we were able to bulk fetch all the data we needed!

## Training pipeline
**PREPARATION**
1. Install requirements.
2. Control Colab GPU and CPU setup.
3. Login to Hopsworks.

**DATASET**
1.  Download full dataset from Hopsworks and save it to Colab.

**XGBOOST REGRESSOR**
1. Load full dataset.
2. Assign correct datatypes to the variables.
3. Separate the features (X) and the label (Y).
4. Split full dataset (*seed = 7*) into a training set (80%) and test set (20%).


5. Initialize XGBOOST Regressor model.
6. Optimize hyperparameters with GridSearch.
7. Train the XGBBOOST model on the training data.
8. Let the model predict on the testing data.
9. Calculate performance metrics: *MAE* and *MAPE*.
10. Plot model's predictions vs. original values.


11. Upload model to Hopsworks Model Registry.
12. Download model from Hopsworks Model Registry.


**AUTOGLUON TABULAR PREDICTOR**
1. Load full dataset and assign correct datatypes.
2. Separate the features (X) and the label (Y).
3. Split full dataset (*seed = 7*) into a training set (80%) and test set (20%).


4. Initialize, train, and then save Autogluon Tabular Predictor.
5. Let the model predict on the testing data.
6. Calculate performance metrics MAE and MAPE.


7. Upload model to Hopsworks Model Registry.
8. Download model from Hopsworks Model Registry.

---




### Models:
Two models were implemented in the UI; XGBOOST Regressor and Autogluon Tabular Predictor.
Both models were trained and evaluated on the same data splits (same seed) and the same performance metrics were used to evaluate their performances.
The purpose for using two models is to enable users to use whichever model they prefer: 
* _AutoGluon Tabular Predictor_, which is a model that is more accurate but requires more time to predict.
* _XGBOOST Regressor_, which is a model that is less accurate but requires less time to predict.

Moreover, we reasoned that it would be interesting to explore and use a popular AutoML-approach in the project since it is a new and exciting field of research.

**Final performance evaluation of the trained models:**

| Models         |   MAPE    | Avg. Execution Time |
|:---------------|:---------:|:-------------------:|
| XGBOOST        | 9,7680 %  |       Instant       |
| AutoGluon      | 9,5450 %  |    ~ 10 seconds     | 

**Note:** _The average execution time_ is the average time it takes for a model to perform a prediction on one (1) datapoint, as observed in the HuggingFace 🤗 UI.
#### XGBoost 🌲
XGBOOST is a library of gradient boosting algorithms. 
The XGBoostRegressor was used with the following settings
because they enable the usage of categorical features: `(tree_method="gpu_hist", enable_categorical=True)` ([reference](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)).
GridSearch was used to optimize the hyperparameters of the model.
The best model was trained on the training dataset and saved to Hopsworks Model Registry.



#### AutoGluon 🧠
Stacking, which is a type of ensemble learning, was implemented with the package AutoGluon through its TabularPredictor-module.
AutoGluon is an AutoML implementation which automates many steps of a typical model development process, such as feature engineering, hyperparameter optimization, model selection, etc.
The training and validation of the model is carried out in a single phase which consists of automatically training and validating each of the stack’s multiple learning algorithms.
After training (and validating) the model, it was saved to the Hopsworks Model Registry.


---

## Inference pipeline (UI)
Blablabla
