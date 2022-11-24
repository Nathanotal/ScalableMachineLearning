# ID 2223 Scalable machine learning and deep learning 🖥️ 
## Lab 1 (Titanic) 🚢:

### Task 1
The Feature, Training, Online/Batch Inference Pipeline was run for the iris model.
The [Interactive UI](https://huggingface.co/spaces/Nathanotal/iris) and the [Dashboard UI](https://huggingface.co/spaces/Nathanotal/irisMonitor) can be found at the links below:

### Task 2
We were tasked to build a similar serverless system using the [titanic dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

#### 1. Data cleaning/engineering
1. Clean the data
    * Three columns have missing values: Age, Cabin, Embarked
    * Age: A random forest regressor will predict the age based on the other features. (see get_age_model())
    * Cabin: Too many missing values to be useful ==> drop feature
    * Embarked: Only 2 missing values ==> drop these rows
    * Columns deemed not useful for the model: Name, Ticket, Cabin, PassengerId ==> drop features
2. Train a random forest regressor to predict the age based on the other features.
3. Use the model to predict the age of the passengers with null values in the age column.
4. Convert categorical features ("sex" and "embarked") to numerical and convert all column names to lowercase.

#### 2. Build a feature pipeline
6. A feature pipeline was built which registered the dataset as a feature group was built. There is also an option to generate and make a feature group using synthetic data.

#### 3. Build a training pipeline
8. A training pipeline was built which reads the training data from a feature view and trains a machine learning model.

#### 4. Build a batch inference pipeline
10. A batch inference pipeline which reads the data and model form hopsworks, tests the model, and generates a prediction history and a confusion matrix was built. This pipeline also includes an option to run the batch inference using synthetic data.

#### 5. Build an [Interactive UI](https://huggingface.co/spaces/Nathanotal/titanic)

#### 6. Build a [Dashboard UI](https://huggingface.co/spaces/Nathanotal/titanic_monitoring)
was built which uses the model built by the training pipeline.
was built which shows the historic performance of the model and its confusion matrix. There is an option to view the result generated from predicting on synthetic or non synthetic passengers.
