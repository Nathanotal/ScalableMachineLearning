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
    """
    1. Get the latest model from Hopsworks
    2. Get batch data from the feature view in Hopsworks
    3. Run the batch data through the model
    4. Spot check the results
    5. Upload three images which show the results of the model and the actual label to Hopsworks
    6. Add the spot checked result to the history and upload it to Hopsworks (as an image and table)
    7. Generate a confusion matrix using the historic predictions and upload it as an image to Hopsworks

    * If SYNTHETIC == True, then we will use the synthetic data from the feature view generated by the feature pipeline 
    """
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

    feature_view = None
    batch_data = None
    # If we want to use synthetic data get that view from the feature store
    if SYNTHETIC:
        try:
            feature_view = fs.get_feature_view(
                name="titanic_modal_synthetic", version=10)
        except:
            print('fel')
            # If we cant get the view create it (this is only needed the first time)
            titanic_fg_synthetic = fs.get_feature_group(
                name="titanic_modal_synthetic", version=10)
            query = titanic_fg_synthetic.select_all()
            feature_view = fs.create_feature_view(name="titanic_modal_synthetic",
                                                  version=10,
                                                  description="Generate synthetic dataset",
                                                  labels=["survived"],
                                                  query=query)
        # X_train, X_test, y_train, y_test = feature_view.train_test_split(
        #     0.2)  # Create training data (needs to be done first time we make a new version)

        # Get batch data from the synthetic feature view
        batch_data = feature_view.get_batch_data()
    else:
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
    print("Survived: ", survived)

    # Generate an image of the person
    int_to_gender = {
        0: "male",
        1: "female"
    }
    age = person["age"]
    gender = person["sex"]
    gender = int_to_gender.get(gender)

    # The API only covers this range of ages
    if age < 9:
        age = 9
    if age > 75:
        age = 75

    generate_passenger_url = f'https://fakeface.rest/face/json?maximum_age={age}&gender={gender}&minimum_age={age}'
    randomized_face_url = requests.get(
        generate_passenger_url).json()["image_url"]

    # Save the image of the person
    img = Image.open(requests.get(randomized_face_url, stream=True).raw)

    dataset_api = project.get_dataset_api()

    link = './latest_survivor_pred.png'
    # If we are using synthetic data add that to the name (store in another location)
    if SYNTHETIC:
        link = './latest_survivor_pred_synthetic.png'
    img.save(link)
    # Upload the image of the person
    dataset_api.upload(link,
                       "Resources/images", overwrite=True)

    # If the person survived we want to upload a green checkmark, otherwise a red cross
    red_cross_url = "https://www.iconsdb.com/icons/preview/red/x-mark-xxl.png"
    green_check_mark_url = "https://www.iconsdb.com/icons/preview/green/checkmark-xxl.png"

    label_to_url = {
        0: red_cross_url,
        1: green_check_mark_url
    }

    # Save the predicted survival
    label_url = label_to_url.get(int(survived))
    img = Image.open(requests.get(label_url, stream=True).raw)

    link = './latest_survivor_label_pred.png'
    if SYNTHETIC:
        link = './latest_survivor_label_pred_synthetic.png'
    img.save(link)
    # Upload the predicted survival label
    dataset_api.upload(link,
                       "Resources/images", overwrite=True)

    # See if the person actually survived
    # Get the feature group
    titanic_fg = None
    if SYNTHETIC:
        titanic_fg = fs.get_feature_group(
            name="titanic_modal_synthetic", version=10)
    else:
        titanic_fg = fs.get_feature_group(name="titanic_modal", version=10)
    df = titanic_fg.read()

    # Get the label for the person
    label = int(df.iloc[randIndex]["survived"])

    # Save the actual survival label
    label_url = label_to_url.get(label)
    print("survived actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)

    link = './latest_survivor_label_actual.png'
    if SYNTHETIC:
        link = './latest_survivor_label_actual_synthetic.png'
    img.save(link)

    # Upload the actual survival label
    dataset_api.upload(link,
                       "Resources/images", overwrite=True)

    # Get a table of the historic predictions
    monitor_fg = None
    if SYNTHETIC:
        monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions_synthetic",
                                                    version=10,
                                                    primary_key=["datetime"],
                                                    description="Titanic survived Prediction/Outcome Monitoring"
                                                    )
    else:
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

    # Make a new df of our spot checked data point
    # (as mentioned this could be done in batch but for simplicity we do it one at a time)
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    # Get the full history
    history_df = monitor_fg.read()

    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)

    # Save the latest history as an image on hopsworks
    if SYNTHETIC:
        dfi.export(df_recent, './df_recent_titanic_synthetic.png',
                   table_conversion='matplotlib')
        dataset_api.upload("./df_recent_titanic_synthetic.png",
                           "Resources/images", overwrite=True)
    else:
        dfi.export(df_recent, './df_recent_titanic.png',
                   table_conversion='matplotlib')
        dataset_api.upload("./df_recent_titanic.png",
                           "Resources/images", overwrite=True)

    # Get predictions and labels and check if we have enough predictions to generate a confusion matrix
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris surviveds
    print("Number of different survived predictions to date: " +
          str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        # Convert labels from string to int
        print(labels)
        labels = labels.astype('int')
        # Create a confusion matrix
        results = confusion_matrix(labels, predictions)

        # Set the labels for the confusion matrix
        df_cm = pd.DataFrame(results, ["Drowned", "Survived"], [
            "Drowned", "Survived"])

        # Create a heatmap with seaborn (the actual resulting image)
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()

        # Save the confusion matrix as an image on hopsworks
        if SYNTHETIC:
            fig.savefig("./confusion_matrix_synthetic.png")
            dataset_api.upload("./confusion_matrix_synthetic.png",
                               "Resources/images", overwrite=True)
        else:
            fig.savefig("./confusion_matrix.png")
            dataset_api.upload("./confusion_matrix.png",
                               "Resources/images", overwrite=True)
    else:
        print("You need 2 different survived predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different titanic survived predictions")


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
