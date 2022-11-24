import os
import modal

# LOCAL = True => run locally
# LOCAL = False => run on Modal
LOCAL = False

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(
        ["hopsworks", "seaborn", "joblib", "scikit-learn"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    """
    Create/fetch a feature view from the Hopsworks feature store.
    Split the data into training and test sets.
    Train a K-nearest neighbour model on the train dataset.
    Evaluate the model on the test dataset.
    Generate a confusion matrix of the model's performance.
    Specify the model's schema; what data represent the features, and what data represents the labels.
    Create an entry for the model in the model registry and save the model and its confusion matrix to it.
    """
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # Fetch or create the feature view. The feature view queries the Titanic dataset from the feature store.
    try:
        feature_view = fs.get_feature_view(name="titanic_modal", version=10)
    except:
        titanic_fg = fs.get_feature_group(name="titanic_modal", version=10)
        query = titanic_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal",
                                              version=10,
                                              description="Read from Titanic dataset",
                                              labels=["survived"],
                                              query=query)

    # Split data into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    print("Training set size: ", X_train.shape)
    print("Test set size: ", X_test.shape)
    print(X_train.head())

    # Initialize and train a KNN classifier (neighbours=5)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train.values.ravel())

    # Evaluate model performance by comparing the predicted labels (y_pred) to the true labels (y_test)
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)

    # Generate a confusion matrix of the model's performance
    results = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(results, ["0", "1"],
                         ["0", "1"])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    # Create an object for the Hopsworks model registry
    mr = project.get_model_registry()

    # Create a directory in which the model and confusion matrix are saved
    model_dir = "titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    joblib.dump(model, model_dir + "/titanic_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")

    # Create a schema for the model which specifies the input (=X_train) and output (=y_train) data
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry for the model in the model registry
    titanic_model = mr.python.create_model(
        name="titanic_modal",
        version=10,
        metrics={"accuracy": metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Survival Predictor"
    )

    # Upload the model and its confusion matrix to the model registry
    titanic_model.save(model_dir)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
