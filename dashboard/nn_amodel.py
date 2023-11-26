from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, SpatialDropout2D, Activation
from keras.callbacks import ModelCheckpoint 
from keras.regularizers import l2
import tensorflow as tf



import pickle
import pandas as pd
import librosa
import os
import numpy as np 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import pickle
from joblib import load
from sklearn.ensemble import RandomForestClassifier



# Load model mapping that relates the model prediction with the name of the chord
with open('chords_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)




# # usar rede convolucional PCP
# X_train_ffn = X_train.reshape(1447,1216,12)
# X_test_ffn =  X_test.reshape(713,1216,12)




def model_instance(hidden_layer):
    model_relu = tf.keras.models.Sequential([ 
        tf.keras.layers.Flatten(input_shape=(1216,12)), 
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(42,activation='softmax')
  ]) 

#Compiling using loss function, Optimizer and Metrics
    model_relu.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model_relu


def load_model():

    checkpoint_path = "models/rf_model.joblib"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    #latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = load(checkpoint_path)

    return model



def get_chord(pred):
    # Load model mapping that relates the model prediction with the name of the chord
    with open('chords_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    pred_idx = np.argmax(pred, axis = 1)[0] -1 
    reversed_mapping = dict(map(reversed, mapping.items()))
    predicted_chord = reversed_mapping[pred_idx]

    return predicted_chord





def load_predict(chroma):
    model = load_model()
    pred = model.predict(chroma)



    return pred[0]


    
    
    



# le = LabelEncoder()
# y_test_encoded = le.fit_transform(y_test)
# y_train_encoded = le.fit_transform(y_train)