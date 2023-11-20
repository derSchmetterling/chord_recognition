import streamlit as st
from st_audiorec import st_audiorec
from audiorecorder import audiorecorder
import numpy as np
from pydub import AudioSegment

import nn_amodel as nn 
import preprocessing_pipeline as pp
from scipy.io import wavfile


st.set_page_config(
    page_title="Reconhecimento de Acordes",
    page_icon="✅",
    layout="wide",
)




# dashboard title
st.title("Reconhecimento de Acordes de Violão")


st.header("Grave seu Acorde", divider = 'blue')
wav_audio_data = st_audiorec()
if wav_audio_data is not None:


    audio = st.audio(wav_audio_data, format='audio/wav')
    data_s16 = np.frombuffer(wav_audio_data, dtype=np.int16, count=len(wav_audio_data)//2, offset=0)
    # data_s16 é o array do audio
    #if audio is not None:
        
        #np.save('chord_x', data_s16)    

st.header("Predição", divider = 'blue')

pred_button = st.button('Qual o acorde?')

if pred_button:

    wavfile.write(filename = 'chord.wav', rate = 48000, data= data_s16)

    try:
        
        chroma =  pp.prepro_pipeline('chord.wav')
        pred,predict_chord = nn.load_predict(chroma)
        st.text(f'O acorde predito é:{predict_chord}')
        st.text(pred)
    
    except Exception as error:
        st.text("Nenhuma gravação foi submetida.")
        print(error)

    




st.header('Koda')
st.image('Koda.webp')


col1, col2 = st.columns((2))

# with col1:
#     st.



# faz o download do áudio em .wav para permitir leitura pelo librosa







tab1,  = st.tabs(["Acorde"])

# with tab1:
#    st.header("Acorde1")




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
