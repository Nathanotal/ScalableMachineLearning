import os
import modal

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from faker import Faker

# LOCAL = True => run locally
# LOCAL = False => run on Modal
LOCAL = False

# SYNTHETIC = True => generate synthetic data
# SYNTHETIC = False => use the Titanic dataset
SYNTHETIC = False


def generateFakePerson():
    """
    Generate a fake person from randomized synthetic data using the Faker library.
    :return: A dictionary with the fake person data.
    """
    fake = Faker()
    fake_data = {
        "Age": fake.random_int(min=0, max=100),
        "Sex": fake.random_int(min=0, max=1),
        "Pclass": fake.random_int(min=1, max=3),
        "SibSp": fake.random_int(min=0, max=10),
        "Parch": fake.random_int(min=0, max=10),
        "Fare": fake.random_int(min=0, max=1000),
        "Embarked": fake.random_int(min=0, max=2),
        "Survived": fake.random_int(min=0, max=1)
    }
    return fake_data


def initialize_data(df):
    """
    Description of the dataset.
    Noteworthy observations.
    Drop columns deemed not useful for the model.
    :param df: Dataframe to be initialized.
    :return: Initialized dataframe.
    """
    # _____ Titanic dataset description _____
    # Survived: Label (0 = No, 1 = Yes)
    # Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
    # Age: Age in years
    # Name: The name of the passenger
    # Sex: male/female
    # SibSp: no. of siblings / spouses aboard the Titanic
    # Parch: no. of parents / children aboard the Titanic
    # Ticket: Ticket number
    # Fare: Passenger fare
    # Cabin: Cabin number
    # Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    # _____ End _____

    # _____ Noteworthy observations _____
    # Three columns have missing values: Age, Cabin, Embarked
    # Age: A random forest regressor will predict the age based on the other features. (see get_age_model())
    # Cabin: Too many missing values to be useful ==> drop feature
    # Embarked: Only 2 missing values ==> drop these rows
    #
    # Columns deemed not useful for the model: Name, Ticket, Cabin, PassengerId ==> drop features
    # _____ End _____

    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)
    df.dropna(subset=["Embarked"], inplace=True)

    return df


def get_age_model(df):
    """
    Train a random forest regressor to predict the age based on the other features.
    :param df: The dataframe containing the data to train the model.
    :return: Trained random forest regressor.
    """
    # Prepare data
    dfAge = df.dropna(subset=["Age"])
    yage = dfAge["Age"]
    dfAge.drop(["Survived", "Age"], axis=1, inplace=True)

    # Convert categorical features to numerical
    le = preprocessing.LabelEncoder()
    Xage = dfAge.apply(le.fit_transform)

    # Split data into train (75%) and test (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        Xage, yage, test_size=0.25)

    # Initialize the random forest regressor and train it
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)

    # Test the model
    predictions = rf.predict(X_test)
    errors = abs(predictions - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'years.')

    return rf


def fix_null_values(df, age_model):
    """
    Replace null values in the age column by predicting new values with the trained random forest regressor.
    :param df: Dataframe to be fixed.
    :param age_model: Trained random forest regressor.
    :return: Fixed dataframe.
    """
    # Prepare the data
    dfAge = df[df["Age"].isnull()]
    Xage = dfAge.drop(["Survived", "Age"], axis=1)

    # Convert categorical features to numerical
    le = preprocessing.LabelEncoder()
    Xage = Xage.apply(le.fit_transform)

    # Predict the value (age) of the null values and replace them
    predictions = age_model.predict(Xage)
    df.loc[df["Age"].isnull(), "Age"] = predictions

    # Round to the nearest integer
    df["Age"] = df["Age"].round().astype(int)

    return df


def prepare_for_write(df):
    """
    Convert categorical features ("sex" and "embarked") to numerical.
    Convert all column names to lowercase.
    :param df: Dataframe to be prepared.
    :return: Prepared dataframe ready to be written to the Hopsworks feature store.
    """

    def sexToInt(x):
        """
        Convert sex to integer.
        """
        if x in [0, 1]:
            return x
        if x == "male":
            return 0
        elif x == "female":
            return 1
        else:
            raise Exception("Unsupported sex value: " + x)

    def embarkedToInt(x):
        """
        Convert embarked to integer.
        """
        if x in [0, 1, 2]:
            return x
        if x == "S":
            return 0
        elif x == "C":
            return 1
        elif x == "Q":
            return 2
        else:
            raise Exception("Unsupported embarked value: " + x)

    df["Sex"] = df["Sex"].apply(sexToInt)
    df["Embarked"] = df["Embarked"].apply(embarkedToInt)
    df.columns = df.columns.str.lower()
    return df


if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn", "dataframe-image", "faker"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def generate_synthetic_data():
    """
    Generate synthetic data (1000 fake persons) and append it to a dataframe.
    :return: Dataframe containing the synthetic data.
    """
    fakeData = []
    for i in range(1000):
        fakePerson = generateFakePerson()
        fakeData.append(fakePerson)

    df = pd.DataFrame(fakeData)
    return df


def g():
    """
    This function is executed by the modal job
    """
    import hopsworks
    import pandas as pd

    # Login to Hopsworks and fetch the feature store
    project = hopsworks.login()
    fs = project.get_feature_store()

    if SYNTHETIC:
        # Generate a synthetic dataset and prepare it for being uploaded to the feature store
        synthetic_df = generate_synthetic_data()
        synthetic_df = prepare_for_write(synthetic_df)

        # Create or get a feature group for the synthetic dataset.
        iris_fg = fs.get_or_create_feature_group(
            name="titanic_modal_synthetic",
            version=10,
            primary_key=["survived", "pclass", "sex", "age",
                         "sibsp", "parch", "fare", "embarked"],
            description="Titanic synthetic passenger dataset")
        # Upload the synthetic dataset to the feature store by adding it to the feature group
        iris_fg.insert(synthetic_df, write_options={"wait_for_job": False}) # overwrite=False, operation="upsert"/"insert"
    else:
        # Download Titanic dataset from the web
        titanic_df = pd.read_csv(
            "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

        # Initialize the dataset (clean it up)
        titanic_df = initialize_data(titanic_df)

        # Train a model to predict the age based on the other features.
        age_model = get_age_model(titanic_df)

        # Use the model to predict the age of the passengers with null values in the age column.
        titanic_df = fix_null_values(titanic_df, age_model)

        # Prepare the dataset for being uploaded to the feature store
        titanic_df = prepare_for_write(titanic_df)

        # Create or get a feature group for the training dataset.
        iris_fg = fs.get_or_create_feature_group(
            name="titanic_modal",
            version=10,
            primary_key=["survived", "pclass", "sex", "age",
                         "sibsp", "parch", "fare", "embarked"],
            description="Titanic passenger dataset")
        # Upload the training dataset to the feature store by adding it to the feature group
        iris_fg.insert(titanic_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
