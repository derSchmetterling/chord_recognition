import streamlit as st
from st_audiorec import st_audiorec
from audiorecorder import audiorecorder

from pydub import AudioSegment




st.set_page_config(
    page_title="Reconhecimento de Acordes",
    page_icon="✅",
    layout="wide",
)









# dashboard title
st.title("Reconhecimento de Acordes Guitarra")



wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    sf.write('stereo_file1.wav', wav_audio_data, 48000, 'PCM_24')

    #print(wav_audio_data)
    #print(audio.audio())

    # data = json.load(audio)
    # print(data)


# st.title("Audio Recorder")
# audio = audiorecorder("Click to record", "Click to stop recording")

# if len(audio) > 0:

#     # To save audio to a file, use pydub export method:
#     audio.export("audio.wav", format="wav")

#     # To play audio in frontend:
#     #st.audio(audio.export().read())  

    

#     # To get audio properties, use pydub AudioSegment properties:
#     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
