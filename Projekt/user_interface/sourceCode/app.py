import gradio as gr
import numpy as np
from PIL import Image
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import hopsworks

# Convert the input to the format the model expects


# Login to hopsworks and get the feature store
project = hopsworks.login()
fs = project.get_feature_store()

# Get the model from Hopsworks
mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=10)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Parse the unput and save it so we can run it through the model
    # Create a dataframe from the input values
    input_variables = pd.DataFrame(
        [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]], columns=columnHeaders)
    df = prepare_for_write(input_variables)

    # Save first row as a numpy array
    input_list = df.iloc[0].to_numpy()

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))

    return None, None


# All features present in the titanic dataset
featureLabels = ['']
numericalInputs = ['']
worthlessInputs = ['']
categoricalInputs = ['']
inputs = []
catToInput = {}
    
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
