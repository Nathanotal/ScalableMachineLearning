# Lab 2: Swedish Text Transcription using Transformers

## Architecture
### 1. A feature pipeline
Downloads the [dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), converts it to features and labels accepted by the model and uploads it to Hopsworks.
### 2. Training pipeline
Downloads the features from Hopsworks, downloads the model to fine tune from Huggingface and fine tunes it. Then uploads the model to Huggingface.
### 3. Inference pipeline (UI)
Downloads the finetuned model from Huggingface and lets the user use the model. The UI is a guessing game where the user gets to input a link to a youtube video. The video is then split into two parts (only the first 10 seconds are considered) and transcribed. The two players then get to guess what is said next in the video by first watching the first 5 seconds of the video and then using the microphone to input their guess of what is said in the next 5 seconds. The interface then transcribes what the players said and compares their guesses with what is actually said. The player with the closest guess wins! Try it out [here](https://huggingface.co/spaces/Nathanotal/GuessTheTranscription).

## Description of task 2 (ways of improving the model performance)
#### A. Model-centric approach. 
In order to tune the model further we could have changed the optimizer (e.g. the Adam to Adagrad) or tuned the hyperparameters e.g. the learning rate. We initially fine tuned the "tiny" model using the default parameters. We examined some changes to the parameters but were not able to achieve a significant improvement. After this we fine tuned the "small" whisper model using the default parameters (500 warm up steps, 4000 training steps), which performed significantly better. In the end we were able to achieve a word error rate of 19.78. We also attempted to tune the hyperparameters when training the small model. However, as it takes more than 12 hours to train the model fully we were not able to try many configurations. Our best other configuration (where lr = 1e-6, we determined 1e-7 to be to low, but 1e-5 seems to overfit the model somewhat as the validaiton loss increased towards the later training steps) achieved a better validation loss of 0.296 (compared to 0.328), but a worse WER of 21.68 (compared to 19.78). We also considered increasing the number of epochs but given constraints in our computing power this was not feasible. In hindsight we should have fine tuned the tiny model with a little data to try out different configurations.
#### B. Data-centric approach. 
In order to improve the model using a data-centric approach we could have either added more data to the dataset from another source or completely switched dataset for the fine tuning. We attempted to integrate the dataset found on the website of the Norwegian language bank ([link](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/)). But after converting the audio files to arrays using librosa, we did not manage to import the data correctly into Google Colab. This was a shame as this dataset was of high quality and was much larger (~80GB) than the one we ended up fine tuning the model with. We could also have tampered with the train/val/test ratio as the current model uses a very large portion of data for validaiton.

## A couple of the fine tuned whisper models
1. [Whisper-small lr=1e-5](https://huggingface.co/Nathanotal/whisper-small-v2) **(Used in app)**
2. [Whisper-small lr=1e-6](https://huggingface.co/Alexao/whisper-small-swe2)
3. [Tiny-swe](https://huggingface.co/Alexao/whisper-tiny-swe)
