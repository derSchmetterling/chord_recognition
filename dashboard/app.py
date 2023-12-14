import streamlit as st
from st_audiorec import st_audiorec
from audiorecorder import audiorecorder
import numpy as np
from pydub import AudioSegment
import pandas as pd
import nn_amodel as nn 
import preprocessing_pipeline as pp
from scipy.io import wavfile


## Configurações da página
st.set_page_config(
    page_title="Reconhecimento de Acordes",
    page_icon="🎻",
    layout="wide",
)



# Abas da página
tab1, tab2= st.tabs(['Predição de Acordes', 'Sobre mim'])





# Primeira aba:
with tab1:
    # Título
    st.title("Reconhecimento de Acordes de Violão")
    st.text('Atualmente, os seguinte acordes estão disponíveis para reconhecimento:')


    # Acordes Disponíveis
    available_chords = pd.DataFrame(['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']).T
    st.dataframe(available_chords)


    # Submete seu acorde:
    st.header("Grave seu Acorde", divider = 'blue')
    st.text('Para gravar, basta clicar em Start Recording, tocar o acorde e clicar em Stop.')
    st.text('Toque o acorde lentamente e de preferência mais de uma vez.')
    st.text('Caso não seja possível gravar ao clicar em Start Recording, clique em Reset e tente novamente.')
    
    # Gravação do acorde:
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:


        audio = st.audio(wav_audio_data, format='audio/wav')

        data_s16 = np.frombuffer(wav_audio_data, dtype=np.int16, count=len(wav_audio_data)//2, offset=0)
        # data_s16 é o array do audio
        #if audio is not None:
            
            #np.save('chord_x', data_s16)    

    st.header("Predição", divider = 'blue')

    pred_button = st.button('Qual o acorde?')

    # Se apertar o botão de predição, um arquivo chord.wav referente a gravação vai ser salvo localmente, passado na pipeline de preprocessamento
    # e um acorde será predito no modelo pretreinado
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
    st.image('imgs/Koda.webp')

