# Chord Recognition


## Report

A summary of all steps performed in this project is available in the **project_notebook.ipynb** file.


## How to reproduce this repository?

If you only want to reproduce the dashboard, you can skip all steps marked as optional.

### 1. (Optional) Data Download
Download the dataset at https://guitarset.weebly.com/ to the root folder of the directory.
The files used here are **audio_hex-pickup_original** and **annotation**.
It is not necessary to decompress the files.

### 2. Dependencies
Download the dependencies listed in the **requirements.txt** file.

### 3. (Optional) Preprocessing

Open the **preprocess.ipynb** file to be guided through the preprocessing procedures.

### 3.1 (Optional) Exploratory Analysis

A quick exploratory analysis of the extracted metadata is performed in **exploratory.ipynb.**

### 4. (Optional) Feature Extraction

Open the **PCP.ipynb** file to be guided through the feature extraction procedures.

### 5. (Optional) Models

Open the **models_pcp.ipynb** file to view the models tested in this project.

### 6. Dashboard

To run the dashboard, simply enter the command `streamlit run app.py` in the dashboard/ directory from the terminal.


### Notes:

The folders old_preprocessing/ and old_models contain preprocessing using Mel Spectrogram and some models based on neural networks.
These files are not used in the project but are available here for documentation purposes.
Some dependencies related to these folders are also not in the requirements.txt file.


### Video
For Portuguese speakers, I have recorded an explanation of this project, available at: https://www.youtube.com/watch?v=2J0rBIaD7Tk. 
If you don't speak Portuguese, you can check out how the dashboard works from 9:19 until the end of the video.


