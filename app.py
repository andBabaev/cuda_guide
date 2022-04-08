import os
from time import time

import streamlit as st
from utils import load_image
from torch import sigmoid, cuda

from cnn import ClassificationModel
from config import IMG_SIZE, CLASSES_BY_IDX, DSET_BASE_DIR


st.header('Chihuahua or muffin?')


# init model
# @st.cache(show_spinner=False)
def create_model(device: str):
    path = 'models/model.ckpt'
    return ClassificationModel.load_from_checkpoint(path).to(device)
device: str = st.radio('Select device', options=['cuda', 'cpu'])
st.text(f"GPU is available: {cuda.is_available()}")

model = create_model(device)
model.eval()
model.warmup(device)


# images list
options = sorted(os.listdir(DSET_BASE_DIR))
fname  = st.selectbox('Select an image', options=options)
fname = os.path.join(DSET_BASE_DIR, fname)



# image upload form
image_file = st.file_uploader("or upload your image", type=["png","jpg","jpeg"])

# inference
if image_file is not None:
    fname = image_file

predict_btn = st.button('Predict!')
if predict_btn:
    image = load_image(fname, IMG_SIZE)
    
    img = model.preprocessing_fn(image).unsqueeze(0).to(device)
    start = time()
    pred = sigmoid(model(img))
    time_range = time() - start
    pred = pred.detach().cpu().tolist()[0][0]
    label = CLASSES_BY_IDX[int(pred>0.5)]

    # To View Uploaded Image
    st.image(image, width=250)
    pred = pred if pred > 0.5 else 1 - pred
    st.write(f"Label: {label} ({pred:0.2%})")
    
    time_msec = round(time_range*1000, 2)
    st.write(f"'{device}' inference time: {time_msec} msec")

