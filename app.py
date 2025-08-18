import streamlit as st
import os
from PIL import Image
from pathlib import Path
import time

st.markdown("<h1 style='text-align: center;'>Predicting NFL \"Short Pass Success\" with a CNN</h1>", unsafe_allow_html=True)
st.set_page_config(layout='wide')

# Remove Streamlit auto-padding
st.markdown("""
    <style>
        /* Remove unnecessary padding */
        .stApp { padding:0px !important; }

        /* Default: desktop-friendly padding */
        .block-container {
            padding-left:13rem !important;
            padding-right:13rem !important;
            padding-top:2rem !important;
            padding-bottom:0px;
        }

        /* Override for tablets/phones */
        @media (max-width: 1200px) {
            .block-container {
                padding-left:2rem !important;
                padding-right:2rem !important;
            }
        }
    </style>
""", unsafe_allow_html=True)


# FUNCTIONS
#######################################################################

# Return a sorted list of Path objects for all .png in folder
@st.cache_data(show_spinner=False)   # 0‑cost after first call
def list_png_paths(folder: str) -> list[Path]:
    base = Path(__file__).parent / folder
    return sorted(p for p in base.iterdir() if p.suffix.lower() == ".png")

# Read a PNG once and return the raw bytes (fastest for st.image)
@st.cache_data(show_spinner=False)
def load_png_bytes(p: Path) -> bytes:
    return p.read_bytes()

# Load ALL PNGs in a folder and return a list of raw bytes
def load_all_frames(folder: str) -> list[bytes]:
    paths = list_png_paths(folder)
    # The list comprehension calls the cached load_png_bytes for each file
    return [load_png_bytes(p) for p in paths]

def set_idx(new_idx):
    st.session_state.idx = max(0, min(new_idx, len(frames) - 1))

def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing

#######################################################################


plays = [
    (2021102404, 108,  '3rd & 3  | 2 yd gain  | Predicted = 🔴, Actual = ❌'), (2021091212, 611,  '2nd & 8  | 1 yd gain  | Predicted = 🔴, Actual = ❌'), 
    (2022091112, 917,  '2nd & 7  | 22 yd gain | Predicted = 🟢, Actual = ✅'), (2021100303, 1951, '3rd & 14 | 9 yd gain  | Predicted = 🔴, Actual = ❌'), 
    (2021091912, 3953, '2nd & 6  | 5 yd gain  | Predicted = 🟢, Actual = ✅'), (2021091204, 2742, '1st & 10 | 11 yd gain | Predicted = 🟢, Actual = ✅'), 
    (2021100400, 262,  '1st & 10 | 7 yd gain  | Predicted = 🟢, Actual = ✅'), (2022103008, 2713, '4th & 3  | 9 yd gain  | Predicted = 🔴, Actual = ✅'), 
    (2022092509, 3717, '1st & 10 | 6 yd gain  | Predicted = 🟢, Actual = ✅'), (2022101606, 1414, '3rd & 4  | 5 yd gain  | Predicted = 🟢, Actual = ✅'), 
    (2021110100, 1351, '1st & 10 | 7 yd gain  | Predicted = 🟢, Actual = ✅'), (2022091105, 2544, '3rd & 10 | 4 yd gain  | Predicted = 🔴, Actual = ❌'), 
    (2021102405, 1665, '2nd & 14 | 14 yd gain | Predicted = 🟠, Actual = ✅'), (2021091206, 1171, '1st & 10 | 14 yd gain | Predicted = 🟢, Actual = ✅'),
    (2022092900, 2204, '1st & 10 | 8 yd gain  | Predicted = 🟢, Actual = ✅'), (2022092200, 2589, '1st & 10 | 12 yd gain | Predicted = 🟠, Actual = ✅'), 
    (2021092605, 3769, '1st & 10 | 9 yd gain  | Predicted = 🟢, Actual = ✅'), (2021091909, 2392, '1st & 10 | 6 yd gain  | Predicted = 🟠, Actual = ✅'), 
    (2022110609, 3668, '2nd & 6  | 6 yd gain  | Predicted = 🟢, Actual = ✅'), (2021091203, 672,  '1st & 10 | 9 yd gain  | Predicted = 🟢, Actual = ✅'), 
    (2022101603, 2950, '1st & 10 | 8 yd gain  | Predicted = 🟠, Actual = ✅'), (2022100901, 2020, '1st & 10 | 5 yd gain  | Predicted = 🟢, Actual = ✅'), 
    (2022102309, 2438, '1st & 10 | 12 yd gain | Predicted = 🟢, Actual = ✅'), (2022103007, 1756, '1st & 10 | 8 yd gain  | Predicted = 🟢, Actual = ✅'),
    (2022102400, 1163, '3rd & 5  | 20 yd gain | Predicted = 🟠, Actual = ✅'), (2022100900, 3109, '1st & 10 | 0 yd gain  | Predicted = 🟢, Actual = ❌'), 
    (2021103105, 4042, '2nd & 10 | 2 yd gain  | Predicted = 🟠, Actual = ❌'), (2022091800, 3523, '2nd & 8  | 2 yd gain  | Predicted = 🔴, Actual = ❌'), 
    (2021091202, 3512, '2nd & 10 | 3 yd gain  | Predicted = 🔴, Actual = ❌'), (2021092610, 3481, '1st & 10 | 3 yd gain  | Predicted = 🟠, Actual = ❌'), 
    (2021101011, 1501, '2nd & 12 | 1 yd gain  | Predicted = 🟠, Actual = ❌'), (2022100908, 2851, '2nd & 12 | -4 yd gain | Predicted = 🔴, Actual = ❌'), 
    (2022091901, 1311, '2nd & 10 | -2 yd gain | Predicted = 🔴, Actual = ❌'), (2022102301, 1988, '3rd & 11 | 3 yd gain  | Predicted = 🔴, Actual = ❌'),
    (2022102304, 1087, '3rd & 10 | 5 yd gain  | Predicted = 🔴, Actual = ❌'), (2021100304, 484,  '2nd & 10 | 1 yd gain  | Predicted = 🟠, Actual = ❌')
]

if 'idx' not in st.session_state:
    st.session_state.idx = 0

if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

if 'last_play' not in st.session_state:
    st.session_state.last_play = None


# VIEWERS
#######################################################################

# Project Decription/Context
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.expander("🏈 PROJECT DETAILS (click to expand)"):
        st.markdown(
            """
            **"SHORT PASS SUCCESS PROBABILITY" (SPSP)** 
            - **Training Data**: NFL player-tracking (2021-2022) - 3,269 short-pass plays (≤2 yds from the line of scrimmage) at the moment the QB throws
            - **Goal**: Predict (before the ball is released) the probability that the play will meet the **"success"** thresholds:
                - **40+%** yards-to-go gained on **1st-down**
                - **60+%** yards-to-go gained on **2nd-down**
                - **100%** yards-to-go gained on **3rd/4th-down**
            - **Output**: **SPSP** (Short-Pass Success Probability), indicated by a coloured circle around the intended receiver:
                - 🟢 = **<span style='color:green'>70+%</span>** SPSP (high)
                - 🟠 = **<span style='color:orange'>40-69%</span>** SPSP (medium)
                - 🔴 = **<span style='color:red'>< 40%</span>** SPSP (low)

            **<a href='https://medium.com/@lou.j.dimuro/predicting-nfl-short-pass-success-with-a-cnn-2f279c985769'>READ FULL PAPER</a>**
            """,unsafe_allow_html=True,
        )
    

# Play Selection View
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_play = st.selectbox('Choose a play:', plays, format_func=lambda x: x[2])

game_id, play_id, _ = selected_play

# Initialize or check for previous play
if 'last_play' not in st.session_state:
    st.session_state.last_play = selected_play

# If the user picks a new play, reset everything and clear the cache
if selected_play != st.session_state.last_play:
    # Reset index / playback flag
    st.session_state.idx = 0
    st.session_state.is_playing = False

    # Clear any previously cached frames (don’t keep old play in memory)
    for key in ("frames", "prob_frames"):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.last_play = selected_play

    # Auto-play the selected play
    st.session_state.is_playing = True


frame_folder = f'play_frames/{game_id}_{play_id}'
prob_folder   = f'play_prob_frames/{game_id}_{play_id}_probs'

# These calls are cached, hit the disk only once per folder
frame_paths = list_png_paths(frame_folder)
prob_paths  = list_png_paths(prob_folder)
if not frame_paths or not prob_paths:
    st.error('No frames found for this play.')
    st.stop()   # nothing to show

# Store the full list of bytes in session_state so we never read from disk again
if 'frames' not in st.session_state:
    st.session_state['frames'] = [load_png_bytes(p) for p in frame_paths]
if 'prob_frames' not in st.session_state:
    st.session_state['prob_frames'] = [load_png_bytes(p) for p in prob_paths]

# Bytes, not file‑paths
frames = st.session_state['frames']
prob_frames = st.session_state['prob_frames']


main_col1, main_col2 = st.columns([1, 1])#st.columns([1.5, 1])

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
spacer1, frame_back_btn, play_pause_btn, frame_forward_btn, replay_btn, spacer2 = st.columns([4, 0.5, 0.5, 0.5, 0.5, 4])
is_first_frame = st.session_state.idx == 0
is_last_frame = st.session_state.idx == len(frames) - 1

with frame_back_btn:
    st.button('\u23EA\uFE0E', on_click=lambda: set_idx(st.session_state.idx - 1), disabled=st.session_state.is_playing or is_first_frame)
with play_pause_btn:
    st.button('\u25B6\uFE0E' if not st.session_state.is_playing else '\u23F8\uFE0E', #▶ ⏸
            on_click=toggle_play,
            disabled=is_last_frame and not st.session_state.is_playing)
with frame_forward_btn:
    st.button('\u23E9\uFE0E', on_click=lambda: set_idx(st.session_state.idx + 1), disabled=st.session_state.is_playing or is_last_frame)
with replay_btn:
    if st.button('\u21BA\uFE0E'): #↺
        set_idx(0)
        if not st.session_state.is_playing:
            st.session_state.is_playing = True
            st.rerun()
    
#######################################################################


# Playback Loop
fps = 9#10
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

