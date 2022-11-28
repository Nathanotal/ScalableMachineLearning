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

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn", "dataframe-image", "faker"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    """
    This function is executed by the modal job
    """
    import hopsworks
    import pandas as pd

    # Login to Hopsworks and fetch the feature store
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Download Titanic dataset from the web
    titanic_df = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

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
