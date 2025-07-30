import streamlit as st
import os
from PIL import Image
import time

st.title("CNN-Based 'Short Pass Success Probability'")

st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Left content")
with col2:
    game_id, play_id = (2021102404, 108)

    mp4_dir = "play_mp4s"
    mp4_files = sorted([f for f in os.listdir(mp4_dir) if f.endswith(".mp4")])
    if not mp4_files:
        st.warning("No MP4 files found in play_mp4s/")
        st.stop()

    selected = st.selectbox("Choose a play:", mp4_files)

    mp4_path = os.path.join(mp4_dir, selected)
    st.video(mp4_path)