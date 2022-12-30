from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from datasets import list_datasets
import torch
import hopsworks
import os

# TODO: figure out
def main():
    # dataset_api = setUp()
    # downloadData(dataset_api)
    # dataset = loadDataset()
    # trainer = initializeTrainer(dataset)
    # trainer.train()
    # uploadModel(trainer)
    ...

def initializeTrainer(dataset):
    ...
    
def uploadModel(trainer): # TODO: fix
    kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: hi, split: test",
    "language": "se",
    "model_name": "Whisper Small V2 - Swedish",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
    }
    trainer.push_to_hub(**kwargs)

def loadDataset():
    # Load the dataset
    return DatasetDict.load_from_disk("dataset/")

def setUp(): # TODO: figure this out
    # Login to HuggingFace and Hopsworks
    # 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
    notebook_login()
    # 993jhbhPecCt6fS5.gvlZik4edWefbGbguZVwrES34rJrBQuaUBpHcJapmRlD6UseqKirncAUSNBOCTBq
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    
    return dataset_api
    
def downloadData(dataset_api):
    os.mkdir("dataset")
    os.mkdir("dataset/train")
    os.mkdir("dataset/test")
    # Get dict which explains the dataset
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/dataset_dict.json", local_path="dataset/", overwrite=True)
    # Get the state and info of the training dataset
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/train/state.json", local_path="dataset/train", overwrite=True)
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/train/dataset_info.json", local_path="dataset/train", overwrite=True)
    # Get the state and info of the testing dataset
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/test/state.json", local_path="dataset/test", overwrite=True)
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/test/dataset_info.json", local_path="dataset/test", overwrite=True)

    # Get the Test data
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/test/dataset.arrow", local_path="dataset/test", overwrite=True)

    # Get the Train data
    downloaded_file_path = dataset_api.download(
        "sthlm_housing/train/dataset.arrow", local_path="dataset/train", overwrite=True)

# This is irrelevant
def voiceMethod(dataset):
    # Define and initializea data collator
    import torch
    from transformers import WhisperProcessor
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import evaluate
    from transformers import WhisperTokenizer
    from transformers import WhisperForConditionalGeneration
    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer

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
        output_dir="./whisper-small-v2",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500, # 500
        max_steps=4000, # 4000
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8, # 8
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    return trainer


if __name__ == '__main__':
    main()