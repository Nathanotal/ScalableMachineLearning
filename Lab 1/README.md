# ID 2223 Scalable machine learning and deep learning 🖥️ 
## Lab 1 (Titanic) 🚢:

### Task 1
The Feature, Training, Online/Batch Inference Pipeline was run for the iris model.
The [Interactive UI](https://huggingface.co/spaces/Nathanotal/iris) and the [Dashboard UI](https://huggingface.co/spaces/Nathanotal/irisMonitor) can be found at the links below:

### Task 2
We were tasked to build a similar serverless system using the [titanic dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

#### 1. Data cleaning/engineering 🧹
1. Clean the data
    * Three columns have missing values: Age, Cabin, Embarked
    * Age: A random forest regressor will predict the age based on the other features. (see get_age_model())
    * Cabin: Too many missing values to be useful ==> drop feature
    * Embarked: Only 2 missing values ==> drop these rows
    * Columns deemed not useful for the model: Name, Ticket, Cabin, PassengerId ==> drop features
2. Train a random forest regressor to predict the age based on the other features.
3. Use the model to predict the age of the passengers with null values in the age column.
4. Convert categorical features ("sex" and "embarked") to numerical and convert all column names to lowercase.

#### 2. Build a feature pipeline 📊
1. Either download the Titanic dataset from the web or generate synthetic data (depending on configuration)
2. Clean the data using the methods mentioned above.
3. Create a feature group on Hopsworks for the training dataset.

#### 3. Build a training pipeline 🏋️‍♂️
1. Create/fetch a feature view from the Hopsworks feature store.
2. Split the data into training and test sets.
3. Train a K-nearest neighbour model on the train dataset.
4. Evaluate the model on the test dataset.
5. Generate a confusion matrix of the model's performance.
6. Specify the model's schema; what data represent the features, and what data represents the labels.
7. Create an entry for the model in the model registry and save the model to it.

#### 4. Build a batch inference pipeline 🔍
 1. Get the latest model from Hopsworks.
 2. Get batch data from the feature view in Hopsworks.
 3. Run the batch data through the model.
 4. Check the result for the last person.
 5. Upload three images which show the results of the model and the actual label to Hopsworks
 6. Add the checked result to the history and upload it to Hopsworks (as an image and table)
 7. Generate a confusion matrix using the historic predictions and upload it as an image to Hopsworks
 * If SYNTHETIC == True, then we will use the synthetic data from the feature view generated by the feature pipeline 

#### 5. Build an [Interactive UI](https://huggingface.co/spaces/Nathanotal/titanic) 🕹️
1. Gets the predictive model from Hopsworks.
2. Gets an input from the user.
3. Converts that input to features which can be used by the model.
4. Predicts whether the person survives or not.
5. Generates an image of the person and an image which indicates if they survive or not.

#### 6. Build a [Dashboard UI](https://huggingface.co/spaces/Nathanotal/titanic_monitoring) 🔬
1. Gets the images (latest passenger, the prediction, label, recent history and the confusion matrix) created by the batch inference pipeline.
2. Displays these images.
* There is an option to view the result generated from predicting on synthetic or non synthetic passengers.
