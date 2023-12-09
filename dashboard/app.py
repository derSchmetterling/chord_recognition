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
    page_icon="🎻",
    layout="wide",
)



tab1, tab2= st.tabs(['Predição de Acordes', 'Sobre mim'])






with tab1:
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
            pred = nn.load_predict(chroma)
            #st.text(pred)

            col1, col2, col3 = st.columns(3)
            col1.metric(pred.columns[0], pred.iloc[0,0])
            col2.metric(pred.columns[1], pred.iloc[0,1])
            col3.metric(pred.columns[2], pred.iloc[0,2])

        
        except Exception as error:
            st.text("Nenhuma gravação foi submetida.")
            print(error)

    


with tab2:
    st.title('Sobre mim')

    st.write("Esse e outros códigos estão disponíveis em: [Github](https://github.com/derSchmetterling)")
    st.write("Contato: [LinkedIn](https://www.linkedin.com/in/pedro-vinicius-3b31aa13b/)")
    st.image('Koda.webp')

