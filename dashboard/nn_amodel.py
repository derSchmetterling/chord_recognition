import pickle
import pandas as pd
import librosa
import os
import numpy as np 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import pickle
from joblib import load
from sklearn.ensemble import RandomForestClassifier



# Carrega o modelo pretreinado
def load_model():

    checkpoint_path = "models/svc_model2.joblib"


    # Create a new model instance
    model = load(checkpoint_path)

    return model







# Predição de scores de probabilidade no modelo pretreinado
def load_predict(chroma):
    model = load_model()
    pred = model.predict_proba(chroma)
    df_pred = pd.DataFrame(pred, columns=model.classes_)
    top3 = df_pred.sort_values(by = 0, axis = 1, ascending=False).iloc[:, :3]

    

    return round(top3, 3)
