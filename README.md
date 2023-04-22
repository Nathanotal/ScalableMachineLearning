# ID2223 ScalableMachineLearning

# Project (Stockholm Housing Valuation) 🌆
## Scraping pipeline 🔍
The scraping is done to backfill the data for the feature pipeline. As this process proved complicated we broke the scraping out to its own pipeline which generates a raw dataset. The pipeline scrapes Booli.se for apartment data. In order to do the first "bulk" scrape one can run bulkScrape.py. However, without several proxies, this takes days because of very strict rate limiting. 


dailyScrape.py is designed to be run once a day as a cronjob. It loads the current raw data, then checks for new listings. If it finds any new listings which are not in the data, it adds them to a list and scrapes the new apartments. After it is done the data is uploaded to Hopsworks. Financial data regarding GPD, interest and unemployment was also scraped. However, a better alternative to gathering this data is to enter it manually as there are not a lot of datapoints.

## Feature pipeline 🧹
The feature pipeline downloads the data collected by the scraping pipeline and cleans it. The daily feature pipeline first loads the raw data and tries to download the current features if there are any. It then maps the coordinates from the current data to the new datapoints (if any new datapoints have been scraped) and then sends the raw data to the feature pipeline. 


A lot of operations are performed on the data. For example, several outliers are excluded. Data with null values are either dropped or filled in. Several columns are parsed to make it possible for machine learning models to accept. The date is also converted to a numerical feature. Coordinates are also added to the datapoints which do not have them. The dataset is subsequently normalized and uploaded to Hopsworks.

### Coordinates 🗺
Importantly, the dataset from the website which was scraped did not include coordinates. As a result these had to be interpolated from the address of the apartment. This proved rather difficult. It would have cost ~$1000 to have identified the coordinates using the Google Maps API. We thus turned to OpenStreetMaps. We downloaded the dataset, however, it was really complicated to make a sufficient SQL query. Thus, we tried to use Nominatim (based on OpenStreetMaps), however they have a rate limit of 1 request/second and 2500 requests/day. Finally, we arrived at our solution, which entailed hosting our own Nominatim API. This API was first hosted on a private rpi, however, the dataset was too large and we had to migrate to a DigitalOcean droplet. With our own API we were able to bulk fetch all the data we needed!

## Training pipeline 🛠

 

### Preparation 
1. Install requirements.
2. Control Colab GPU and CPU setup.
3. Login to Hopsworks.

### Dataset
1.  Download full dataset from Hopsworks and save it to Colab.

### XGBoost Regressor 🌲
1. Load full dataset.
2. Assign correct datatypes to the variables.
3. Drop categorical features (if training the non categorical version).
4. Separate the features (X) and the label (Y).
5. Split full dataset (seed = 7) into a training set (80%) and test set (20%).


6. Initialize XGBOOST Regressor model.
7. Optimize hyperparameters with GridSearch.
8. Train the XGBBOOST model on the training data.
9. Let the model predict on the testing data.
10. Calculate performance metrics: MAE and MAPE.
11. Plot model's predictions vs. original values.


12. Upload model to Hopsworks Model Registry.
13. Download model from Hopsworks Model Registry.


### AutoGluon tabular predictor 🧠
1. Load full dataset and assign correct datatypes.
2. Separate the features (X) and the label (Y).
3. Split full dataset (*seed = 7*) into a training set (80%) and test set (20%).


4. Initialize, train, and then save Autogluon Tabular Predictor.
5. Let the model predict on the testing data.
6. Calculate performance metrics MAE and MAPE.


7. Upload model to Hopsworks Model Registry.
8. Download model from Hopsworks Model Registry.




### Models:
Two models were implemented in the UI; an XGBOOST Regressor and an Autogluon Tabular Predictor.
Both models were trained and evaluated on the same data splits (same seed) and the same performance metrics were used to evaluate their performances.
The purpose for using two models is to enable users to use whichever model they prefer: 
* _AutoGluon Tabular Predictor_, which is a model that is more accurate but requires more time to predict, it also handles categorical data better.
* _XGBOOST Regressor_, which is a model that is less accurate but requires less time to predict.

Moreover, we reasoned that it would be interesting to explore and use a popular AutoML-approach in the project since it is a new and exciting field of research.

### Final performance evaluation of the trained models and their capabilities:

| Models    |   MAPE    | Avg. Execution Time | Can handle categorical features <br/> in Hugginface 🤗 UI |
|:----------|:---------:|:-------------------:|:---------------------------------------------------------:|
| XGBOOST   | 9,7680 %  |       Instant       |                             ❌                            |
| AutoGluon | 9,5450 %  |    ~ 10 seconds     |                             ✔️                            | 

#### Note 1:
The _average execution time_ is the average time it takes for a model to perform a prediction on one (1) datapoint, as observed in the HuggingFace 🤗 UI.

#### Note 2:
An XGBOOST Regressor model is typically able to handle categorical features if it has access to a GPU.
Since a free tier version of Huggingface 🤗 Space is used for this project no GPU is available – only a CPU. 
Thus, the used XGBOOST model was trained on the same data as the AutoGluon model, but without the categorical features.
On the contrary, the AutoGluon model is able to process categorical features while running on a CPU, and will thus use the features "streetName" and "agency" in its predictions, whereas the XGBOOST model will exclude these.

#### XGBoost 🌲
XGBOOST is a library of gradient boosting algorithms. 
The XGBoostRegressor was used with the following settings
because they enable the usage of categorical features: `(tree_method="gpu_hist", enable_categorical=True)` ([reference](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)).
GridSearch was used to optimize the hyperparameters of the model.
The best models were saved to Hopsworks Model Registry (one trained with and one without categorical features, see [Note 2](#Note-2) for more information regarding this).

**Note**: The UI could not run the categorical feature version as it requires a GPU. Thus, we use the non categorical model in the UI.




#### AutoGluon 🧠
Stacking, which is a type of ensemble learning, was implemented with the package AutoGluon through its TabularPredictor-module.
AutoGluon is an AutoML implementation which automates many steps of a typical model development process, such as feature engineering, hyperparameter optimization, model selection, etc.
The training and validation of the model is carried out in a single phase which consists of automatically training and validating each of the stack’s multiple learning algorithms.
After training (and validating) the model, it was saved to the Hopsworks Model Registry.



## Inference pipeline ([UI](https://huggingface.co/spaces/Nathanotal/stockholmHousingValuation)) 
The inference pipeline is a UI where you can get valuations of apartments in Stockholm by entering the features of the apartment. The UI gets the lat/lon from the address and interpolates several extra features e.g. historical or expected GDP. As a GPU was required to use categorical features for XGBoost, and our huggingface space only has a CPU, we had to make versions of the XGBoost model which did not use categorical data.

# Lab 2: Swedish Text Transcription using Transformers :dart:

## Architecture 🏗
### 1. A feature pipeline 📄
Downloads the [dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), converts it to features and labels accepted by the model and uploads it to Hopsworks.
### 2. Training pipeline :hammer:
Downloads the features from Hopsworks, downloads the model to fine tune from Huggingface and fine tunes it. Then uploads the model to Huggingface.
### 3. Inference pipeline (UI) 🕹
Downloads the finetuned model from Huggingface and lets the user use the model. The UI is a guessing game where the user gets to input a link to a youtube video. The video is then split into two parts (only the first 10 seconds are considered) and transcribed. The two players then get to guess what is said next in the video by first watching the first 5 seconds of the video and then using the microphone to input their guess of what is said in the next 5 seconds. The interface then transcribes what the players said and compares their guesses with what is actually said. The player with the closest guess wins! Try it out [here](https://huggingface.co/spaces/Nathanotal/GuessTheTranscription).

## Description of task 2 (ways of improving the model performance) 📜
#### A. Model-centric approach. 
In order to tune the model further we could have changed the optimizer (e.g. the Adam to Adagrad) or tuned the hyperparameters e.g. the learning rate. We initially fine tuned the "tiny" model using the default parameters. We examined some changes to the parameters but were not able to achieve a significant improvement. After this we fine tuned the "small" whisper model using the default parameters (500 warm up steps, 4000 training steps), which performed significantly better. In the end we were able to achieve a word error rate of 19.78. We also attempted to tune the hyperparameters when training the small model. However, as it takes more than 12 hours to train the model fully we were not able to try many configurations. Our best other configuration (where lr = 1e-6, we determined 1e-7 to be to low, but 1e-5 seems to overfit the model somewhat as the validaiton loss increased towards the later training steps) achieved a better validation loss of 0.296 (compared to 0.328), but a worse WER of 21.68 (compared to 19.78). We also considered increasing the number of epochs but given constraints in our computing power this was not feasible. In hindsight we should have fine tuned the tiny model with a little data to try out different configurations.
#### B. Data-centric approach. 
In order to improve the model using a data-centric approach we could have either added more data to the dataset from another source or completely switched dataset for the fine tuning. We attempted to integrate the dataset found on the website of the Norwegian language bank ([link](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/)). But after converting the audio files to arrays using librosa, we did not manage to import the data correctly into Google Colab. This was a shame as this dataset was of high quality and was much larger (~80GB) than the one we ended up fine tuning the model with. We could also have tampered with the train/val/test ratio as the current model uses a very large portion of data for validaiton.

## A couple of the fine tuned whisper models :books:
1. [Whisper-small lr=1e-5](https://huggingface.co/Nathanotal/whisper-small-v2) **(Used in app)**
2. [Whisper-small lr=1e-6](https://huggingface.co/Alexao/whisper-small-swe2)
3. [Tiny-swe](https://huggingface.co/Alexao/whisper-tiny-swe)


# Lab 1 (Titanic) 🚢:

## Task 1
The Feature, Training, Online/Batch Inference Pipeline was run for the iris model.
The [Interactive UI](https://huggingface.co/spaces/Nathanotal/iris) and the [Dashboard UI](https://huggingface.co/spaces/Nathanotal/irisMonitor) can be found at the links below:

## Task 2
We were tasked to build a similar serverless system using the [titanic dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

### 1. Data cleaning/engineering 🧹
1. Clean the data
    * Three columns have missing values: Age, Cabin, Embarked
    * Age: A random forest regressor will predict the age based on the other features. (see get_age_model())
    * Cabin: Too many missing values to be useful ==> drop feature
    * Embarked: Only 2 missing values ==> drop these rows
    * Columns deemed not useful for the model: Name, Ticket, Cabin, PassengerId ==> drop features
2. Train a random forest regressor to predict the age based on the other features.
3. Use the model to predict the age of the passengers with null values in the age column.
4. Convert categorical features ("sex" and "embarked") to numerical and convert all column names to lowercase.

### 2. Build a feature pipeline 📊
1. Either download the Titanic dataset from the web or generate synthetic data (depending on configuration)
2. Clean the data using the methods mentioned above.
3. Create a feature group on Hopsworks for the training dataset.

### 3. Build a training pipeline 🏋️‍♂️
1. Create/fetch a feature view from the Hopsworks feature store.
2. Split the data into training and test sets.
3. Train a K-nearest neighbour model on the train dataset.
4. Evaluate the model on the test dataset.
5. Generate a confusion matrix of the model's performance.
6. Specify the model's schema; what data represent the features, and what data represents the labels.
7. Create an entry for the model in the model registry and save the model to it.

### 4. Build a batch inference pipeline 🔍
 1. Get the latest model from Hopsworks.
 2. Get batch data from the feature view in Hopsworks.
 3. Run the batch data through the model.
 4. Check the result for the last person.
 5. Upload three images which show the results of the model and the actual label to Hopsworks
 6. Add the checked result to the history and upload it to Hopsworks (as an image and table)
 7. Generate a confusion matrix using the historic predictions and upload it as an image to Hopsworks
 * If SYNTHETIC == True, then we will use the synthetic data from the feature view generated by the feature pipeline 

### 5. Build an [Interactive UI](https://huggingface.co/spaces/Nathanotal/titanic) 🕹️
1. Gets the predictive model from Hopsworks.
2. Gets an input from the user.
3. Converts that input to features which can be used by the model.
4. Predicts whether the person survives or not.
5. Generates an image of the person and an image which indicates if they survive or not.

### 6. Build a [Dashboard UI](https://huggingface.co/spaces/Nathanotal/titanic_monitoring) 🔬
1. Gets the images (latest passenger, the prediction, label, recent history and the confusion matrix) created by the batch inference pipeline.
2. Displays these images.
* There is an option to view the result generated from predicting on synthetic or non synthetic passengers.
