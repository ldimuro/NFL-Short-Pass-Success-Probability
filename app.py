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

def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing

#######################################################################


# plays = [
#     (2018091612, 1271), (2021102404, 108), (2021102410, 3434), (2021100303, 1951), (2021091212, 611), (2022102311, 4025),
#     (2021091910, 3540), (2018090902, 821), (2018091612, 1271), (2018100710, 2454)
# ]
plays = [
    (2022091112, 917),
    (2021100303, 1951),
    (2021102410, 3434),
    (2021102404, 108),
    (2021091212, 611),
    (2021101704, 1613),
    (2021091911, 127),
    (2021091202, 3536),
    (2022102311, 4025),
    (2021091910, 3540),
    (2022103009, 152),
    (2022091110, 514),
    (2021103110, 1111),
    (2021102404, 2345),
    (2021092601, 506),
    (2021092000, 2983),
    (2021102500, 3305),
    (2022092510, 181),
    (2022091102, 3695),
    (2021091911, 2866),
    (2022100911, 1622),
    (2022101607, 3596),
    (2021091204, 3195),
    (2022110609, 3817),
    (2022092508, 1955),
    (2022102301, 2667),
    (2021103110, 719),
    (2021102410, 3833),
    (2021103104, 1372),
    (2021092613, 2258),
    (2022091105, 2544)
]

if 'idx' not in st.session_state:
    st.session_state.idx = 0

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False



# VIEWERS
#######################################################################
    
# left_col, right_col = st.columns([1, 5])

# Play Selection View
# with left_col:
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    selected_play = st.selectbox('Choose a play:', plays)

# selected_play = st.selectbox('Choose a play:', plays)
game_id, play_id = selected_play

# Initialize or check for previous play
if "last_play" not in st.session_state:
    st.session_state.last_play = selected_play

# If user selects a new play, reset frame & stop playback
if selected_play != st.session_state.last_play:
    st.session_state.idx = 0
    st.session_state.is_playing = False
    st.session_state.last_play = selected_play


# Get play frames
frames = load_frames(f'play_frames/{game_id}_{play_id}_behind_los_norm_centered')
if not frames:
    st.stop()

# Get probability frames
prob_frames = load_frames(f'play_prob_frames/{game_id}_{play_id}_behind_los_norm_centered_probs')
if not prob_frames:
    st.stop()


# Data Views
# with right_col:

main_col1, main_col2 = st.columns([1.5, 1])

# Main Play View
with main_col1:
    play_slot = st.empty()
    play_slot.image(frames[st.session_state.idx],
                    #caption=f"Play Frame {st.session_state.idx + 1}",
                    use_container_width=True)

# Probability View
with main_col2:
    prob_slot = st.empty()
    prob_slot.image(prob_frames[st.session_state.idx],
                    # caption=f"Probability Frame {st.session_state.idx + 1}",
                    use_container_width=True)
    
# Controls
spacer1, c1, c2, c3, c4, spacer2 = st.columns([4, 0.5, 0.5, 0.5, 0.5, 4])#c1,c2,c3,c4 = st.columns([1,1,1,8])
with c1:
    st.button('<', on_click=lambda: set_idx(st.session_state.idx - 1), disabled=st.session_state.is_playing)
with c2:
    is_last_frame = st.session_state.idx == len(frames) - 1
    st.button("▶" if not st.session_state.is_playing else "⏸", #▶ ⏸
            on_click=toggle_play,
            disabled=is_last_frame and not st.session_state.is_playing)
with c3:
    st.button('\>', on_click=lambda: set_idx(st.session_state.idx + 1), disabled=st.session_state.is_playing)
with c4:
    if st.button('↺'): #↺
        set_idx(0)
        if not st.session_state.is_playing:
            st.rerun()
    
#######################################################################


# Playback Loop
fps = 10
frame_delay = 1.0 / fps
if st.session_state.is_playing:
    while st.session_state.idx < len(frames) - 1:
        time.sleep(frame_delay)
        st.session_state.idx += 1

        play_slot.image(frames[st.session_state.idx],
                        # caption=f"Play Frame {st.session_state.idx + 1}",
                        use_container_width=True)
        prob_slot.image(prob_frames[st.session_state.idx],
                        # caption=f"Probability Frame {st.session_state.idx + 1}",
                        use_container_width=True)

        if not st.session_state.is_playing:
            break

    # Stop playback when last frame is reached
    st.session_state.is_playing = False
    st.rerun()

