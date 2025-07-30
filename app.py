import streamlit as st
import os
from PIL import Image
import time

st.title("CNN-Based 'Short Pass Success Probability'")

st.set_page_config(layout='wide')

# FUNCTIONS
#######################################################################
def load_frames(path):
    frames = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith('.png')
    ])
    return frames

def set_idx(new_idx):
    st.session_state.idx = max(0, min(new_idx, len(frames) - 1))

#######################################################################

game_id = '2021102404'
play_id = '108'
fps = 10
frame_delay = 1.0 / fps

# Get play frames
frames = load_frames(f'play_frames/{game_id}_{play_id}_behind_los_norm_centered')
if not frames:
    st.stop()

# Get probability frames
prob_frames = load_frames(f'play_prob_frames/{game_id}_{play_id}_behind_los_norm_centered_probs')
if not prob_frames:
    st.stop()


if 'idx' not in st.session_state:
    st.session_state.idx = 0


# BUTTONS
c1,c2,c3,c4 = st.columns([1,1,1,4])
with c1:
    if st.button('⬅️'):
        set_idx(st.session_state.idx - 1)
with c2:
    if st.button('➡️'):
        set_idx(st.session_state.idx + 1)
with c3:
    if st.button('🔁'):
        set_idx(0)

# SLIDER
idx = st.slider('Frame', 0, len(frames)-1, value=st.session_state.idx)
set_idx(idx)


# MAIN VIEWERS
main_col1, main_col2 = st.columns([1.38, 1])

# PLAY FRAMES
with main_col1:
    play_slot = st.empty()
    play_slot.image(frames[st.session_state.idx],
                    caption=f"Play Frame {st.session_state.idx + 1}",
                    use_container_width=True)

# PROBABILITY FRAMES
with main_col2:
    prob_slot = st.empty()
    prob_slot.image(prob_frames[st.session_state.idx],
                    caption=f"Probability Frame {st.session_state.idx + 1}",
                    use_container_width=True)



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
