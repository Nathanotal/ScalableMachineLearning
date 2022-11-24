# ID 2223 Scalable machine learning and deep learning 🖥️ 
## Lab 1 (Titanic) 🚢:

### Task 1
The Feature, Training, Online/Batch Inference Pipeline was run for the iris model.
The [Interactive UI](https://huggingface.co/spaces/Nathanotal/iris) and the [Dashboard UI](https://huggingface.co/spaces/Nathanotal/irisMonitor) can be found at the links below:

### Task 2
We were tasked to build a similar serverless system using the [titanic dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

1. The data was cleaned and feature engineered. 
2. A feature pipeline was built which registered the dataset as a feature group was built. There is also an option to generate and make a feature group using synthetic data.
3. A training pipeline was built which reads the training data from a feature view and trains a machine learning model.
4. A batch inference pipeline which reads the data and model form hopsworks, tests the model, and generates a prediction history and a confusion matrix was built. This pipeline also includes an option to run the batch inference using synthetic data.
5. An [Interactive UI](https://huggingface.co/spaces/Nathanotal/titanic) was built which uses the model built by the training pipeline.
6. A [Dashboard UI](https://huggingface.co/spaces/Nathanotal/titanic_monitoring) was built which shows the historic performance of the model and its confusion matrix. There is an option to view the result generated from predicting on synthetic or non synthetic passengers.
