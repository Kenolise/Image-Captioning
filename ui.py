import streamlit as st
import pandas as pd
import numpy as np
from engine import inference

def ui():
    st.markdown(f'<h1 style="color:#3f5299;">{"🧑🏽‍🦲 Image Captioning Tinkering!"}</h1>', unsafe_allow_html=True)

    st.markdown("\
    This is a sequence to sequence model to caption images built using PyTorch.\
    This model uses inceptionV3 as encoder and LSTM layers as decoder.\
    This model was trained on Flickr30k dataset.")

    st.markdown(f'<h3 style="color:#3f5299;">{"Samples"}</h3>', unsafe_allow_html=True)
    st.image('./test_examples/pawpaw.jpg', width = 500, channels = 'RGB')
    st.markdown('**PREDICTION:** ***a boy wailing with hand on head .***')

    st.markdown('')
    st.image('./test_examples/pic.jpg', width = 500, channels = 'RGB')
    st.markdown('**PREDICTION:** ***a man holding a paper and rejecting a drink offer.***')
   
    st.markdown(f'<h3 style="color:#3f5299;">{"Now try it out"}</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        out = inference(uploaded_file)
        st.image(uploaded_file, width = 500, channels = 'RGB')
        st.markdown("**PREDICTION:** " + out)

    st.markdown('## [Code on GitHub](https://github.com/Kenolise/Image-Captioning)')
    st.markdown('')
    st.markdown("""# Connect with me
  [<img height="30" src = "https://img.shields.io/twitter/follow/kenoseio"/>][Twitter]
  
  [Twitter]: https://twitter.com/intent/follow?screen_name=kenoseio""", unsafe_allow_html=True)
if __name__ == '__main__':
    ui()
