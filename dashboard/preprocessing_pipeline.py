

import librosa
import numpy as np 
import pandas as pd
import tensorflow as tf
import scipy




## Gera o cromagrama do áudio enviado pelo usuário do dashboard
def PCP(filepath, sr = 44100, filters = True):

    y, sr = librosa.load(filepath, sr=sr)

    CQT = np.abs(librosa.cqt(y, sr=sr))
    chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0], n_chroma = 12)
    chromagram = chroma_map.dot(CQT)

    # Max-normalize each time step

    chromagram = librosa.util.normalize(chromagram, axis=0)

    ydb = librosa.amplitude_to_db(CQT,ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    if filters == True:
        chroma_min = chroma_filter = np.minimum(chroma,
                           librosa.decompose.nn_filter(chroma,
                                                       aggregate=np.median,
                                                       metric='cosine'))

        chroma_smooth = scipy.ndimage.median_filter(chroma_min, size=(1, 9))

        return chroma_smooth
    
    

    return chroma



# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, max_padding=1216):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if (size < max_padding):
            xDiff = max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded




# Corta o áudio enviado na duração máxima ou adiciona zeros em áudios com duração inferior a esperada
def prepro_pipeline(audio_path):
    chroma = PCP(audio_path)[:, :1216]
    padded_chroma = add_padding([chroma])
    reshaped_padded = padded_chroma[0].reshape(1,1216*12)


    return reshaped_padded