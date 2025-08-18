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
            padding-left:20rem !important;
            padding-right:20rem !important;
            padding-top:2rem !important;
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

# ESSENTIAL PLAYS:
# - (2022091112, 917)  - (1) receiver becomes open as he passes the DL
# - (2021100303, 1951) - (0) receiver never really has a chance at a 1st down
# - (2021102410, 3434) - (0) receiver never really has a chance at a 1st down
# - (2021102404, 108)  - (0) CROWN JEWEL, perfect example of a receiver being open, but getting shut down later in the play
# - (2021091212, 611)  - (0) CROWN JEWEL, perfect example of a receiver being open, but getting shut down later in the play
# - (2021101704, 1613) - (0) High SPSP False Positive
# - (2021091912, 3953) - (1) Shows a receiver getting open in the middle of the field
# - (2021091204, 2742) - (1) Shows a receiver getting open in the middle of the field
# - (2021100400, 262)  - (1) Good upward trajectory play
# - (2022103008, 2713) - (1) Looks like an easy conversion, should be way higher SPSP
# - (2021091911, 881)  - (1) Low SPSP, but a great spin-move by Ceedee Lamb for a big gain
# - (2022092509, 3717) - (1) Good positive class example
# - (2022101606, 1414) - (1) Good example of a pick play
# - (2021110100, 1351) - (1) Notice how SPSP dips as #30 turns towards #31 / Try with #17 as a receiver too
# - (2022091105, 2544) - (0) PERFECT EXAMPLE OF FAILURE PREDICTION
# - (2021091911, 127)  - (1) False Negative, looks like #86 has no chance for any gain, but in the actual scenario, he breaks a tackle for a big gain
# - (2021091202, 3536) - (0) False Positive, great tackle by #40
# - (2021102405, 1665) - (1) Run this play with #1 as the intended receiver and compare SPSPs with #87
# - (2021091206, 1171) - (1) Extremely high SPSP since receiver is wide open
# - (2022092900, 2204) - (1) SPSP slowly rises as #21 backs further away from 85
# - (2022092200, 2589) - (1) Interesting screen setup 
# - (2021092605, 3769) - (1) Extremely high SPSP, can also run with #15 as a receiver as well
# - (2022091801, 2412) - (1) False Negative, very low SPSP, but the receiver has plenty of space to pick up the 1st down
# - (2021091909, 2392) - (1) QB hits the receiver at the highpoint of the SPSP
# - (2022100209, 2452) - (1) False Negative, although SPSP does rise as blockers start getting in front of the receiver
# - (2022100213, 2057) - DON'T USE, but interesting case where the model thought this was a short pass, but was a fade in the EZ
# - (2022110609, 3668) - (1) Just a solid play where the receiver gets open and the QB hit him at the highpoint of the SPSP
# - (2021091203, 672)  - (1) Another solid play where the receiver gets wide open
# - (2022101603, 2950) - (1) An interesting screen setup
# - (2022100901, 2020) - (1) QB could've maximized yards gained if thrown a bit earlier
# - (2022101608, 2859) - (1) False Negative, after the catch the receiver gets a ton of good blocking
# - (2022102700, 2820) - (1) PERFECT EXAMPLE OF False Negative, receiver shouldn't have a chance at the 1st down, but makes an insane juke that puts defenders on skates
# - (2022102309, 2438) - (1) Looks like it'l be a so-so gain, but the defender misses the tackle
# - (2022103007, 1756) - (1) Just a solid play
# - (2022091101, 2951) - DON'T USE, but there may be some kind of mistake, as the ball is being thrown as if by a left-handed QB
# - (2022102400, 1163) - (1) Should be a high SPSP, but it's only around 0.6
# - (2022091105, 619)  - (1) False Negative, very low SPSP but good screen play that gets a 1st down
# - (2022100900, 3109) - (0) Play just never has a chance
# - (2021103105, 4042) - (0) Good example of a play that could go either way
# - (2022091800, 3523) - (0) Another example of a play that could go either way
# - (2021091202, 3512) - (0) QB misses high point of SPSP
# - (2021092610, 3481) - (0) QB misses high point of SPSP
# - (2021101011, 1501) - (0) Just a good negative example
# - (2022100908, 2851) - (0) Play just never had a chance
# - (2022091901, 1311) - (0) QB completely misses the high point of SPSP
# - (2022110600, 3074) - (0) False Positive, but SPSP remains high when the receiver has a pick blocker
# - (2022102301, 1988) - (0) Play never has a chance
# - (2022102304, 1087) - (0) Very low SPSP
# - (2021100304, 484)  - (0) QB misses high point of SPSP
# - (2022101605, 2054) - (0) Play never has a chance

# TOTAL (1) examples (no Falses) = 22
# TOTAL (0) examples (no Falses) = 17


plays = [
    (2021102404, 108,  'Play 208  | 3rd & 3  | Predicted = 🔴, Actual = ❌'), (2021091212, 611,  'Play 611  | 2nd & 8  | Predicted = 🔴, Actual = ❌'), 
    (2022091112, 917,  'Play 917  | 2nd & 7  | Predicted = 🟢, Actual = ✅'), (2021100303, 1951, 'Play 1951 | 3rd & 14 | Predicted = 🔴, Actual = ❌'), 
    (2021091912, 3953, 'Play 3953 | 2nd & 6  | Predicted = 🟢, Actual = ✅'), (2021091204, 2742, 'Play 2742 | 1st & 10 | Predicted = 🟢, Actual = ✅'), 
    (2021100400, 262,  'Play 262  | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2022103008, 2713, 'Play 2713 | 4th & 3  | Predicted = 🔴, Actual = ✅'), 
    (2022092509, 3717, 'Play 3717 | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2022101606, 1414, 'Play 1414 | 3rd & 4  | Predicted = 🟢, Actual = ✅'), 
    (2021110100, 1351, 'Play 1351 | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2022091105, 2544, 'Play 2544 | 3rd & 10 | Predicted = 🔴, Actual = ❌'), 
    (2021102405, 1665, 'Play 1665 | 2nd & 14 | Predicted = 🟠, Actual = ✅'), (2021091206, 1171, 'Play 1171 | 1st & 10 | Predicted = 🟢, Actual = ✅'),
    (2022092900, 2204, 'Play 2204 | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2022092200, 2589, 'Play 2589 | 1st & 10 | Predicted = 🟠, Actual = ✅'), 
    (2021092605, 3769, 'Play 3769 | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2021091909, 2392, 'Play 2392 | 1st & 10 | Predicted = 🟠, Actual = ✅'), 
    (2022110609, 3668, 'Play 3668 | 2nd & 6  | Predicted = 🟢, Actual = ✅'), (2021091203, 672,  'Play 672  | 1st & 10 | Predicted = 🟢, Actual = ✅'), 
    (2022101603, 2950, 'Play 2950 | 1st & 10 | Predicted = 🟠, Actual = ✅'), (2022100901, 2020, 'Play 2020 | 1st & 10 | Predicted = 🟢, Actual = ✅'), 
    (2022102309, 2438, 'Play 2438 | 1st & 10 | Predicted = 🟢, Actual = ✅'), (2022103007, 1756, 'Play 1756 | 1st & 10 | Predicted = 🟢, Actual = ✅'),
    (2022102400, 1163, 'Play 1163 | 3rd & 5  | Predicted = 🟠, Actual = ✅'), (2022100900, 3109, 'Play 3109 | 1st & 10 | Predicted = 🟢, Actual = ❌'), 
    (2021103105, 4042, 'Play 4042 | 2nd & 10 | Predicted = 🟠, Actual = ❌'), (2022091800, 3523, 'Play 3523 | 2nd & 8  | Predicted = 🔴, Actual = ❌'), 
    (2021091202, 3512, 'Play 3512 | 2nd & 10 | Predicted = 🔴, Actual = ❌'), (2021092610, 3481, 'Play 3481 | 1st & 10 | Predicted = 🟠, Actual = ❌'), 
    (2021101011, 1501, 'Play 1501 | 2nd & 12 | Predicted = 🟠, Actual = ❌'), (2022100908, 2851, 'Play 2851 | 2nd & 12 | Predicted = 🔴, Actual = ❌'), 
    (2022091901, 1311, 'Play 1311 | 2nd & 10 | Predicted = 🔴, Actual = ❌'), (2022102301, 1988, 'Play 1988 | 3rd & 11 | Predicted = 🔴, Actual = ❌'),
    (2022102304, 1087, 'Play 1087 | 3rd & 10 | Predicted = 🔴, Actual = ❌'), (2021100304, 484,  'Play 484  | 2nd & 10 | Predicted = 🟠, Actual = ❌')
]

# plays = {
#     (2021102404, 108):  'Play 108  | 3rd & 3  | Predicted = 🔴, Actual = ❌',
#     (2021091212, 611):  'Play 611  | 2nd & 8  | Predicted = 🔴, Actual = ❌',
#     (2022091112, 917):  'Play 917  | 2nd & 7  | Predicted = 🟢, Actual = ✅',
#     (2021100303, 1951): 'Play 1951 | 3rd & 14 | Predicted = 🔴, Actual = ❌',
#     (2021091912, 3953): 'Play 3953 | 2nd & 6  | Predicted = 🟢, Actual = ✅',
#     (2021091204, 2742): 'Play 2742 | 1st & 10 | Predicted = 🟢, Actual = ✅',
#     (2021100400, 262):  'Play 262  | 1st & 10 | Predicted = 🟢, Actual = ✅',

# }

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

# Short aliases – now they are bytes, not file‑paths
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

