import os
import modal

# LOCAL = True => run locally
# LOCAL = False => run on Modal
LOCAL = False

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(
        ["hopsworks", "seaborn", "git+https://github.com/huggingface/transformers", "evaluate"])

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
    # Install all dependencies
    # Login to HuggingFace and Hopsworks
    from huggingface_hub import notebook_login
    import hopsworks

    notebook_login()
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    # Make directories to save the dataset
    import os
    os.mkdir("common_voice2")
    os.mkdir("common_voice2/train")
    os.mkdir("common_voice2/test")

    # Get dict which explains the dataset
    downloaded_file_path = dataset_api.download(
        "Voice/dataset_dict.json", local_path="common_voice2/", overwrite=True)
    # Get the state and info of the training dataset
    downloaded_file_path = dataset_api.download(
        "Voice/train/state.json", local_path="common_voice2/train", overwrite=True)
    downloaded_file_path = dataset_api.download(
        "Voice/train/dataset_info.json", local_path="common_voice2/train", overwrite=True)
    # Get the state and info of the testing dataset
    downloaded_file_path = dataset_api.download(
        "Voice/test/state.json", local_path="common_voice2/test", overwrite=True)
    downloaded_file_path = dataset_api.download(
        "Voice/test/dataset_info.json", local_path="common_voice2/test", overwrite=True)

    # Get the Test data
    downloaded_file_path = dataset_api.download(
        "Voice/test/dataset.arrow", local_path="common_voice2/test", overwrite=False)

    # Get the Train data
    downloaded_file_path = dataset_api.download(
        "Voice/train/dataset.arrow", local_path="common_voice2/train", overwrite=False)

    # Load the dataset
    from datasets import load_dataset, DatasetDict
    from datasets import list_datasets

    # TODO - load the downloaded Hugging Face dataset from local disk 
    cc = DatasetDict.load_from_disk("common_voice2/")

    # Load the common_voice, TODO: Fix
    common_voice = cc
    # common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
    # common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)

    print(common_voice)

    # Define and initializea data collator
    import torch

    from transformers import WhisperProcessor
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union


    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define evaluation metrics and define the trainer
    import evaluate
    from transformers import WhisperTokenizer
    from transformers import WhisperForConditionalGeneration
    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=1,
        output_dir="./whisper-small-hi",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=5, # 500
        max_steps=40, # 4000
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=4, # 8
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    """
    function ConnectButton(){
        console.log("Connect pushed"); 
        document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
    }
    setInterval(ConnectButton, 60000);
    """

    trainer.train()

    # Publish the model to the Hugging Face Hub
    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: hi, split: test",
        "language": "hi",
        "model_name": "Whisper Small Hi - Swedish",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
    trainer.push_to_hub(**kwargs)

    # import hopsworks
    # import pandas as pd
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import classification_report
    # import seaborn as sns
    # from matplotlib import pyplot
    # from hsml.schema import Schema
    # from hsml.model_schema import ModelSchema
    # import joblib

    # # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    # project = hopsworks.login()
    # # fs is a reference to the Hopsworks Feature Store
    # fs = project.get_feature_store()

    # # Fetch or create the feature view. The feature view queries the Titanic dataset from the feature store.
    # try:
    #     feature_view = fs.get_feature_view(name="titanic_modal", version=10)
    # except:
    #     titanic_fg = fs.get_feature_group(name="titanic_modal", version=10)
    #     query = titanic_fg.select_all()
    #     feature_view = fs.create_feature_view(name="titanic_modal",
    #                                           version=10,
    #                                           description="Read from Titanic dataset",
    #                                           labels=["survived"],
    #                                           query=query)

    # # Split data into train (80%) and test (20%) sets
    # X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    # print("Training set size: ", X_train.shape)
    # print("Test set size: ", X_test.shape)
    # print(X_train.head())

    # # Initialize and train a KNN classifier (neighbours=5)
    # model = KNeighborsClassifier(n_neighbors=5)
    # model.fit(X_train, y_train.values.ravel())

    # # Evaluate model performance by comparing the predicted labels (y_pred) to the true labels (y_test)
    # y_pred = model.predict(X_test)
    # metrics = classification_report(y_test, y_pred, output_dict=True)

    # # Generate a confusion matrix of the model's performance
    # results = confusion_matrix(y_test, y_pred)
    # df_cm = pd.DataFrame(results, ["0", "1"],
    #                      ["0", "1"])
    # cm = sns.heatmap(df_cm, annot=True)
    # fig = cm.get_figure()

    # # Create an object for the Hopsworks model registry
    # mr = project.get_model_registry()

    # # Create a directory in which the model and confusion matrix are saved
    # model_dir = "titanic_model"
    # if os.path.isdir(model_dir) == False:
    #     os.mkdir(model_dir)
    # joblib.dump(model, model_dir + "/titanic_model.pkl")
    # fig.savefig(model_dir + "/confusion_matrix.png")

    # # Create a schema for the model which specifies the input (=X_train) and output (=y_train) data
    # input_schema = Schema(X_train)
    # output_schema = Schema(y_train)
    # model_schema = ModelSchema(input_schema, output_schema)

    # # Create an entry for the model in the model registry
    # titanic_model = mr.python.create_model(
    #     name="titanic_modal",
    #     version=10,
    #     metrics={"accuracy": metrics['accuracy']},
    #     model_schema=model_schema,
    #     description="Titanic Survival Predictor"
    # )

    # # Upload the model and its confusion matrix to the model registry
    # titanic_model.save(model_dir)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
