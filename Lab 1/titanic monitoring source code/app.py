import gradio as gr
from PIL import Image
import hopsworks

# If You want to inspect the results for the synthetic data set SYNTHETIC = TRUE
SYNTHETIC = False

latestSurvivorImage = 'latest_survivor_pred'
latestSurvivorPred = 'latest_survivor_label_pred'
latestSurvivorLabel = 'latest_survivor_label_actual'
recentHistory = 'df_recent_titanic'
confusionMatrix = 'confusion_matrix'

if SYNTHETIC:
    latestSurvivorImage += '_synthetic'
    latestSurvivorPred += '_synthetic'
    latestSurvivorLabel += '_synthetic'
    recentHistory += '_synthetic'
    confusionMatrix += '_synthetic'

latestSurvivorImage += '.png'
latestSurvivorPred += '.png'
latestSurvivorLabel += '.png'
recentHistory += '.png'
confusionMatrix += '.png'

with gr.Blocks() as demo:
    # Login to hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Download all the necessary files
    dataset_api = project.get_dataset_api()

    print('Downloading...')
    dataset_api.download(f"Resources/images/{latestSurvivorImage}")
    dataset_api.download(
        f"Resources/images/{latestSurvivorPred}")
    dataset_api.download(
        f"Resources/images/{latestSurvivorLabel}")
    dataset_api.download(f"Resources/images/{recentHistory}")
    dataset_api.download(f"Resources/images/{confusionMatrix}")

    # Arrange the images
    with gr.Column():
        gr.Label("Today's passenger")
        input_img = gr.Image(f"{latestSurvivorImage}",
                             elem_id="passenger-img").style(
            height='100', rounded=False)
        with gr.Row():
            with gr.Column():
                gr.Label("Today's predicted survival")
                input_img = gr.Image(
                    f"{latestSurvivorPred}", elem_id="predicted-img").style(
                    height='100', rounded=False)
            with gr.Column():
                gr.Label("Today's actual survival")
                input_img = gr.Image(
                    f"{latestSurvivorLabel}", elem_id="actual-img").style(
                    height='100', rounded=False)
        with gr.Row():
            with gr.Column():
                gr.Label("Recent Prediction History")
                input_img = gr.Image(
                    f"{recentHistory}", elem_id="recent-predictions")
            with gr.Column():
                gr.Label(
                    "Confusion Maxtrix with Historical Prediction Performance")
                input_img = gr.Image(f"{confusionMatrix}",
                                     elem_id="confusion-matrix")

demo.launch()
