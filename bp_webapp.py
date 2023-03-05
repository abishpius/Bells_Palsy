import streamlit as st
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
import PIL

import warnings
warnings.filterwarnings('ignore')



st.title("Bell's Palsy Identifier")

upload_file = st.file_uploader("Please upload an png of jpg file", type = ['png', 'jpg'])

if upload_file is not None:
    im = PIL.Image.open(upload_file)
    rgb_im = im.convert('RGB')
    rgb_im.save('tmp.jpg')
    
    # Class Names
    class_names = ['Bells_Palsy', 'Normal']
    
    # Load Model
    model = tf.keras.models.load_model('my_model.h5')
        
    # Prepare image
    img = tf.keras.utils.load_img(
    'tmp.jpg', target_size=(224,224)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Print what the top predicted class is
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    
    st.subheader(
        "Model Predicts: {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
  )