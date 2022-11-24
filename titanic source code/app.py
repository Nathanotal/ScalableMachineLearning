import gradio as gr
import numpy as np
from PIL import Image
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import hopsworks
import joblib

# Convert the input to the format the model expects


def prepare_for_write(df):
    # Convert the categorical features to numerical
    def sexToInt(x):
        if x == "male":
            return 0
        elif x == "female":
            return 1
        else:
            raise Exception("Unsupported sex value: " + x)

    def embarkedToInt(x):
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
    # le = preprocessing.LabelEncoder()
    # df = df.apply(le.fit_transform)
    df.columns = df.columns.str.lower()
    return df


# Login to hopsworks and get the feature store
project = hopsworks.login()
fs = project.get_feature_store()

# Get the model from Hopsworks
mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=10)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

# For generating the input form
catToInput = {
    "Sex": ["male", "female"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
    "Pclass": ["First", "Second", "Third"]
}

cityToInput = {
    "Southampton": "S",
    "Cherbourg": "C",
    "Queenstown": "Q"
}

classToInput = {
    "First": 1,
    "Second": 2,
    "Third": 3
}


inputs = []
numericalInputs = ["Age", "SibSp", "Parch", "Fare"]
# Maybe move cabin to categorical (or just remove it)
worthlessInputs = ["Name", "Ticket", "Cabin", "Title"]
categoricalInputs = ["Sex", "Embarked", "Pclass"]

columnHeaders = ["Pclass", "Sex", "Age", "SibSp",
                 "Parch", "Fare", "Embarked"]


def titanic(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Parse the unput and save it so we can run it through the model
    Embarked = cityToInput[Embarked]
    Pclass = classToInput[Pclass]
    # Create a dataframe from the input values
    input_variables = pd.DataFrame(
        [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]], columns=columnHeaders)
    df = prepare_for_write(input_variables)

    # Save first row as a numpy array
    input_list = df.iloc[0].to_numpy()

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))

    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.

    intLabelToText = {0: "Died", 1: "Survived"}  # Debug

    survived = res[0]
    # The API we are using only supports this age range
    if Age < 9:
        Age = 9
    if Age > 75:
        Age = 75

    # Generate a face of the inputted person
    generate_survivor_url = f'https://fakeface.rest/face/json?maximum_age={int(Age)}&gender={Sex}&minimum_age={int(Age)}'
    randomized_face_url = requests.get(
        generate_survivor_url).json()["image_url"]

    survivor_url = randomized_face_url
    img = Image.open(requests.get(survivor_url, stream=True).raw)

    # Show a green check mark if the person is predicted to survive, otherwise show a red x
    red_cross_url = "https://www.iconsdb.com/icons/preview/red/x-mark-xxl.png"
    green_check_mark_url = "https://www.iconsdb.com/icons/preview/green/checkmark-xxl.png"

    label_to_url = {
        0: red_cross_url,
        1: green_check_mark_url
    }

    url = label_to_url.get(survived)

    # Save the image of the person
    img2 = Image.open(requests.get(url, stream=True).raw)

    return img, img2


# All features present in the titanic dataset
featureLabels = ["Pclass", "Name", "Sex", "Age", "SibSp",
                 "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

# Generate the input form
for feature in featureLabels:
    if feature in numericalInputs:
        if feature == 'Age':
            inputs.append(gr.inputs.Slider(9, 75, 1, label='Age (years)'))
        elif feature == 'SibSp':
            inputs.append(gr.inputs.Slider(
                0, 10, 1, label='Number of siblings/spouses aboard'))
        elif feature == 'Parch':
            inputs.append(gr.inputs.Slider(
                0, 10, 1, label='Number of parents/children aboard'))
        elif feature == 'Fare':
            inputs.append(gr.inputs.Slider(0, 1000, 1, label='Ticket fare'))
        else:
            raise Exception(f'Feature: "{feature}" not found')
    elif feature in worthlessInputs:
        pass
        # inputs.append(gr.Inputs.Textbox(default='text', label=feature))
    elif feature in categoricalInputs:
        if feature == "Sex":
            inputs.append(gr.inputs.Dropdown(
                choices=catToInput.get(feature), default="male", label=feature))
        elif feature == "Embarked":
            inputs.append(gr.inputs.Dropdown(
                choices=catToInput.get(feature), default="Southampton", label='City of embarkation'))
        elif feature == "Pclass":
            inputs.append(gr.inputs.Dropdown(
                choices=catToInput.get(feature), default=3, label='Ticket class'))
    else:
        raise Exception(f'Feature: "{feature}" not found')

# Create the interface
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survivor Predictive Analytics",
    description="Experiment with person features to predict which survivor it is.",
    allow_flagging="never",
    inputs=inputs,
    outputs=[gr.Image(type="pil").style(
        height='100', rounded=False), gr.Image(type="pil").style(
        height='100', rounded=False)])

demo.launch()
