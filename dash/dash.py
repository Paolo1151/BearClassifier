import requests
import sys
import os
import requests
import json
from typing import List

from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st
from PIL import Image
import plotly.graph_objects as go


BEAR_TYPES = ['Black', 'Grizzly', 'Panda', 'Polar', 'Teddy']
BACKEND = 'http://bear:8000/predict'

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r

def generate_prob_figure(data: List[float]):
    fig = go.Figure(data=[
        go.Bar(name='Bear Probability', x=BEAR_TYPES, y=data)
    ])
    return fig

def get_most_probable_label(data: List[float]):
    max_i = -1
    max_k = -1
    for i, prob in enumerate(data):
        if prob > max_k:
            max_i = i
            max_k = prob
    return BEAR_TYPES[max_i]


def process_image():
    uploaded_image = st.file_uploader('Input Bear Picture Here')
    if uploaded_image:
        st.image(uploaded_image)

    if st.button("Submit"):
        pred = json.loads(process(uploaded_image, BACKEND).content.decode('utf-8'))['Prediction']
        label = get_most_probable_label(pred)
        fig = generate_prob_figure(pred)
        st.write(f"Bear is most likely a: {label} Bear!")
        st.plotly_chart(fig)
        

def render():
    st.title("How much of a Bear are you?")
    process_image()


if __name__ == '__main__':
    render()