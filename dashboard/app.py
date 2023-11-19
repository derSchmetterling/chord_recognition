import streamlit as st
from st_audiorec import st_audiorec
from audiorecorder import audiorecorder
import numpy as np
from pydub import AudioSegment

import nn_amodel as nn 
import preprocessing_pipeline as pp


st.set_page_config(
    page_title="Reconhecimento de Acordes",
    page_icon="✅",
    layout="wide",
)









# dashboard title
st.title("Reconhecimento de Acordes Guitarra")



wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    audio = st.audio(wav_audio_data, format='audio/wav')

    # data_s16 é o array do audio
    data_s16 = np.frombuffer(wav_audio_data, dtype=np.int16, count=len(wav_audio_data)//2, offset=0)
    #np.save('chord_x', data_s16)



from scipy.io import wavfile

# faz o download do áudio em .wav para permitir leitura pelo librosa
wavfile.write('aux.wav',48000, data_s16)


chroma =  pp.prepro_pipeline('aux.wav')

predict_chord = nn.load_predict(chroma)







# numpy to .wav
# audio = np.load('dashboard/teste.npy')
# wavfile.write('stereoAudio.wav',48000, audio)

# st.title("Audio Recorder")
# audio = audiorecorder("Click to record", "Click to stop recording")

# if len(audio) > 0:

#     # To save audio to a file, use pydub export method:
#     audio.export("audio.wav", format="wav")

#     # To play audio in frontend:
#     #st.audio(audio.export().read())  

    

#     # To get audio properties, use pydub AudioSegment properties:
#     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
