import streamlit as st
import os
from PIL import Image
import time

st.title("CNN-Based 'Short Pass Success Probability'")

st.set_page_config(layout='wide')

main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.write('filler')
with main_col2:

    game_id = '2021102404'
    play_id = '108'
    fps = 10  # controls playback speed
    frame_delay = 1.0 / fps
    folder_path = f'play_frames/{game_id}_{play_id}_behind_los_norm_centered'
    frames = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.png')
    ])
    if not frames:
        st.stop()

    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    # Play navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
    with col1:
        if st.button('< frame'):
            st.session_state.idx = max(st.session_state.idx - 1, 0)
    with col2:
        if st.button('frame >'):
            st.session_state.idx = min(st.session_state.idx + 1, len(frames) - 1)
    with col3:
        if st.button('🔁'):
            st.session_state.idx = 0


    # Slider (updates when using navigation buttons)
    idx = st.slider('Frame', 0, len(frames) - 1, value=st.session_state.idx)
    st.session_state.idx = idx


    # Frame display
    frame_path = frames[st.session_state.idx]
    img = Image.open(frame_path)
    st.image(img, caption=f'Frame {st.session_state.idx + 1} / {len(frames)}', use_container_width=True)


# with col1:
#     st.write('Left content')
# with col2:
#     game_id, play_id = (2021102404, 108)

#     mp4_dir = 'play_mp4s'
#     mp4_files = sorted([f for f in os.listdir(mp4_dir) if f.endswith('.mp4')])
#     if not mp4_files:
#         st.warning('No MP4 files found in play_mp4s/')
#         st.stop()

#     selected = st.selectbox('Choose a play:', mp4_files)

#     mp4_path = os.path.join(mp4_dir, selected)
#     # st.video(mp4_path)
