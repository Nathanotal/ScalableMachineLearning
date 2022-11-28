import os
import modal

LOCAL = False
SYNTHETIC = False

if LOCAL == False:
    stub = modal.Stub()
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn", "dataframe-image"])

    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import random

    # Login to Hopsworks and get the feature store
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get the trained model
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=10)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    # Get batch data from the regular feature view
    feature_view = fs.get_feature_view(name="titanic_modal", version=10)
    batch_data = feature_view.get_batch_data()

    # Run the batch data through the model and get the predictions
    y_pred = model.predict(batch_data)
    # print(y_pred) # Debug

    # Get a random person and the predicted survival to inspect the result and use for the confusion matrix
    # (this could be done in batch but for simplicity we do it one at a time)
    randIndex = y_pred.size-random.randint(0, y_pred.size-1)
    person = batch_data.iloc[randIndex]
    survived = y_pred[randIndex]

    dataset_api = project.get_dataset_api()
    link = './latest_survivor_pred.png'
    # If we are using synthetic data add that to the name (store in another location)

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=10)
    df = titanic_fg.read()

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=10,
                                                primary_key=["datetime"],
                                                description="Titanic survived Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [str(label)],
        'datetime': [now],
    }

    # Get the full history
    history_df = monitor_fg.read()
    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent_titanic.png',
               table_conversion='matplotlib')
    dataset_api.upload("./df_recent_titanic.png",
                       "Resources/images", overwrite=True)

    # Get predictions and labels and check if we have enough predictions to generate a confusion matrix
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris surviveds


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
