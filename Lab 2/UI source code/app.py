"""## Divide the video into one 10s before and one 10s after"""

# Login to HuggingFace somehow (todo)

"""## Convert to audio

## Get model
"""
import subprocess
from transformers import pipeline
pipe = pipeline(model="Nathanotal/whisper-small-v2")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]

    return text

"""## Download and trim the video"""

# Commented out IPython magic to ensure Python compatibility.
def downloadAndTranscribeVideo(source_url):
  """**Input url to youtube video**"""
  if "=" in source_url:
    id = source_url.split('=', 1)[1]
  else:
    id = source_url.split('/')[-1]

  # Empty folder
#   %rm -rf '/content/drive/My Drive/ID2223/LAB2/'

  """**Create output folder**"""

  # change this to /content/drive/My Drive/folder_you_want
  # output_folder = '/content/drive/My Drive/ID2223/LAB2/'

  # import os
  # def my_mkdirs(folder):
  #   if os.path.exists(folder)==False:
  #     os.makedirs(folder)
  # my_mkdirs('/content/tmp/')

  # my_mkdirs(output_folder)

  # Get URLs to video file and audio file
  # Attempt to get 720p clip, else get best possible quality
  try:  
    proc = subprocess.Popen(f'yt-dlp -g -f bv[height=720][ext=webm]+ba[ext=m4a] "{source_url}"', shell=True, stdout=subprocess.PIPE)
    print(proc)
    video_url, audio_url = proc.stdout.read().decode('utf-8').split()
    # video_url, audio_url = proc.communicate()[0].decode('utf-8')
  except:
    proc = subprocess.Popen(f'yt-dlp -g -f bv[ext=webm]+ba[ext=m4a] "{source_url}"', shell=True, stdout=subprocess.PIPE)
    print(proc)
    # video_url, audio_url = proc.communicate()[0].decode('utf-8')
    video_url, audio_url = proc.stdout.read().decode('utf-8').split()

  print('Video:', video_url)
  print('Audio:', audio_url)

  """**Download part of video and audio files**"""

  temp_video = "temp_video.mkv"
  temp_audio = "temp_audio.m4a"

  # Download video file (first 10 seconds)
  subprocess.run(f'ffmpeg -probesize 10M -y -i "{video_url}" -ss 00:00:00 -t 00:00:10 -c copy "{temp_video}"', shell=True)

  # Download audio file (first 10 seconds)
  subprocess.run(f'ffmpeg -probesize 10M -y -i "{audio_url}" -ss 00:00:00 -t 00:00:10 -c copy "{temp_audio}"', shell=True)


  """**MUX video and audio files**"""
  temp_output = "output.mp4"

  # MUX video and audio files into final output [mkv]
  subprocess.run(f'ffmpeg  -hide_banner -loglevel error -y -i "{temp_video}" -i "{temp_audio}" -c copy "{temp_output}"', shell=True)

  first10Video = "first10Video.mp4"
  second10Video = "second10Video.mp4"

  subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -i "{temp_output}" -ss 00:00:00 -to 00:00:05 -c copy "{first10Video}"', shell=True)
  subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -i "{temp_output}" -ss 00:00:05 -to 00:00:10 -c copy "{second10Video}"', shell=True)

  first10Audio = "first10Audio.m4a"
  second10Audio = "second10Audio.m4a"

  subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -i "{first10Video}" -vn -acodec copy "{first10Audio}"', shell=True)
  subprocess.run(f'ffmpeg -hide_banner -loglevel error -y -i "{second10Video}" -vn -acodec copy "{second10Audio}"', shell=True)

  first10AudioFinal = "first10AudioFinal.mp3"
  second10AudioFinal = "second10AudioFinal.mp3"

  subprocess.run(f'ffmpeg -y -i "{first10Audio}" -c:v copy -c:a libmp3lame -q:a 4 "{first10AudioFinal}"', shell=True)
  subprocess.run(f'ffmpeg -y -i "{second10Audio}" -c:v copy -c:a libmp3lame -q:a 4 "{second10AudioFinal}"', shell=True)

  firstVideoText = transcribe('first10AudioFinal.mp3')
  secondVideoText = transcribe('second10AudioFinal.mp3')

  # Delete temporary files
  subprocess.run(f'rm "{temp_video}"', shell=True)
  subprocess.run(f'rm "{temp_audio}"', shell=True)

  return firstVideoText, secondVideoText

# print(downloadAndTranscribeVideo('https://www.youtube.com/watch?v=93WrIPY4_4E'))

"""## Build UI"""

from transformers import pipeline
import gradio as gr

def calculateSimilarity(texta, actualText):
  texta = texta.lower().strip()
  actualText = actualText.lower().strip()
  textaL = texta.split(" ")
  actualTextL = actualText.split(" ")

  totalWords = len(actualTextL)
  matchingWords = 0

  for word in textaL:
    if word in actualTextL:
      matchingWords += 1

  return int(100*(matchingWords / totalWords))


def game(videoLink, loadVideo, audio1, audio2, theState):
  theState = theState or []
  firstText = "test"
  secondText = "test"
  if loadVideo:
    firstText, secondText = downloadAndTranscribeVideo(videoLink)
    theState = [firstText, secondText]
    return "first10Video.mp4", firstText, "", "", "", "", "", "second10Video.mp4", "", theState
  elif len(theState) == 0:
    return "first10Video.mp4", "", "", "", "", "", "", "second10Video.mp4", "", theState
  else:
    firstText, secondText = theState[0], theState[1]

  t1 = transcribe(audio1)
  t2 = transcribe(audio2)
  t1Res = calculateSimilarity(t1, secondText)
  t2Res = calculateSimilarity(t2, secondText)

  res = 'The game is even, everybody wins!'
  if t1Res > t2Res:
    res = 'Player 1 wins!'
  elif t1Res < t2Res:
    res = 'Player 2 wins!'

  return "first10Video.mp4", firstText, t1, str(t1Res) + '% match', t2, str(t2Res) + '% match', res, "second10Video.mp4", secondText, theState

# exInputs = [[None], [None], ["/content/ut.webm"]]

gameInterface = gr.Interface(fn=game, 
                    inputs=[gr.Textbox(label='Link to video'), 
                            gr.Checkbox(label='Load a new video'), 
                            gr.Audio(source="microphone", type="filepath", label='Player 1\'s guess'), 
                            gr.Audio(source="microphone", type="filepath", label='Player 2\'s guess'), 
                            "state"], 
                    outputs=[gr.Video(label='First ten seconds'), 
                             gr.Textbox(label='Transcription of first ten seconds'),
                             gr.Textbox(label='Transcription for player 1'), 
                             gr.Textbox(label='Percentage match:'), 
                             gr.Textbox(label='Transcription for player 2'), 
                             gr.Textbox(label='Percentage match:'), 
                             gr.Textbox(label='Result:'), 
                             gr.Video(label='Next ten seconds'), 
                             gr.Textbox(label='Transcription of next ten seconds'), 
                             "state"],
                    title="Phrase guessing game",
                    description="1. Paste a link to a youtube video in the \"Link to video\" box and check the checkmark \"Load new video\".\n2. Click \"Submit\". \n3. Uncheck the checkmark \"Load new video\". \n4. Listen to the first five seconds of the video. \n5. Player 1 and 2 guess what is spoken in the next 5 seconds of the video by recording it using the microphone. \n6. Click \"Submit\". \n7. See who made the best matching guess! \n9. Repeat steps 1-7. \n10. Enjoy!"
                    )

gameInterface.launch()