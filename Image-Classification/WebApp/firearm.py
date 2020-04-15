import streamlit as st
import numpy as np
import pandas as pd
import math

import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras import backend, layers, models
from PIL import Image
import os
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

st.title('Threat Detection')
st.header("This tool automatically identifies a firearm using Convolutional Neural Networks (CNNs).")
st.write("Please pick an image using the drop-down menu on the left.")

# Sidebar that allows user to choose an image
st.sidebar.title("Image Selection")

from os import listdir
from os.path import isfile, join

# Path to images that will be used for detection
image_path = "/Users/zakijefferson/Desktop/Demo-Images/"
onlyfiles = [f for f in listdir(image_path) if isfile(join(image_path, f))]
imageselect = st.sidebar.selectbox("Please pick an image using this drop-down menu.", onlyfiles)

image = Image.open(image_path + imageselect)
st.image(image, use_column_width=True)

# Importing other python file
import firearm_testing


# Function that leades to hdf5 files of the saved CNN model
@st.cache
def firearm_detection():
    """
    This function leads to the
    hdf5 files of the saved CNN model.
    From here you will be able to run
    the selected image through the model
    and recieve results on the type of class.
    """
    model_1_path = '/Users/zakijefferson/code/Threat-Detection/Image-Classification/specified_model.hdf5'
    model_1 = load_model(model_1_path)
    return model_1


model_1 = firearm_detection()

prediction_1 = firearm_testing.predict((model_1),image_path + imageselect)
st.subheader('Step 1:')
st.write('Does the image have an Assault Rifle or Handgun ?')
st.title(prediction_1)
