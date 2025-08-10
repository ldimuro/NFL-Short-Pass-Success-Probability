import matplotlib.pyplot as plt
import constants
import math
from pandas import DataFrame
import os
import imageio.v2 as imageio
import shutil
import numpy as np
import subprocess
import imageio_ffmpeg
import matplotlib.patheffects as pe

def plot_frame(frame, play_data, spsp_prob, spsp_rolling_avg, receiver_id, file_name, zoom):
    fig, ax = plt.subplots(figsize=(12, 7.5 if zoom else 6.5))

    # print('HERE:\n', frame)

    ball = frame[frame['team'] == 'football'].iloc[0] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] == 'football'].iloc[0]
    ball_x = ball['x']
    ball_y = ball['y']

    zoom_offset_x = 15
    zoom_offset_y = 8

    # Set green background for the field
    ax.set_facecolor('mediumseagreen')

    try:
        off_color = team_colors[play_data['possessionTeam']]
        def_color = team_colors[play_data['defensiveTeam']]
    except:
        off_color = team_colors['home']
        def_color = team_colors['away']

    # Draw red end zone (left) and blue end zone (right)
    ax.axvspan(0, 10, color=off_color, zorder=1)
    ax.axvspan(110, 120, color=def_color, zorder=1)

    # Draw yard lines every 10 yards
    for x in range(10, 111, 5):
        ax.axvline(x=x, color='white', linewidth=4 if zoom else 1, zorder=2)

    # Draw yard line numbers
    for x in range(20, 101, 10):
        field_val = str(x-10 if x < 60 else 110-x)

        # Bottom numbers
        ax.text(x=x, 
                y=constants.SIDELINE_TO_HASH/2, 
                s=f'{field_val[0]} {field_val[1]}', 
                fontsize=16, 
                ha='center', 
                va='center', 
                color='white', 
                fontname='Times New Roman',
                fontweight='bold')

        # Top numbers
        ax.text(x=x, 
                y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH/2, 
                s=f'{field_val[0]} {field_val[1]}', 
                fontsize=16, 
                ha='center', 
                va='center', 
                color='white', 
                fontname='Times New Roman', 
                rotation=180,
                fontweight='bold')
        
        # Arrows next to yard labels
        if x != 60:
            ax.text(x=x-2.5 if x < 60 else x+2.5,
                    y=constants.SIDELINE_TO_HASH/2 + 0.4,
                    s='\u25B6',
                    fontsize=6,
                    ha='center', 
                    va='center', 
                    color='white',
                    rotation=180 if x < 60 else 0)
            
            ax.text(x=x-2.5 if x < 60 else x+2.5,
                    y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH/2 - 0.4,
                    s='\u25B6',
                    fontsize=6,
                    ha='center', 
                    va='center', 
                    color='white',
                    rotation=180 if x < 60 else 0)


    # Draw Center Field and Goalines
    ax.axvline(x=constants.CENTER_FIELD, color='white', linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.OFF_GOALLINE, color='white', linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.DEF_GOALLINE, color='white', linewidth=6 if zoom else 2, zorder=2.1)

    # Draw hash marks
    ax.axhline(y=constants.SIDELINE_TO_HASH, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    ax.axhline(y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    # ax.axhline(y=0.5, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    # ax.axhline(y=constants.FIELD_WIDTH-0.5, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)

    # Draw LoS and 1st-Down marker
    # ax.axvline(x=constants.DEF_GOALLINE - play_data['absoluteYardlineNumber'] + constants.OFF_GOALLINE, color='#26248f', linewidth=6 if zoom else 2, zorder=2.2)
    ax.axvline(x=constants.MIDFIELD + play_data['yardsToGo'], color="#f2d627", linewidth=6 if zoom else 2, zorder=2.2)

    # Handle team colors
    teams = frame['team'].unique().tolist() if ('club' not in frame.columns or frame['club'].isna().all()) else frame['club'].unique().tolist()
    teams.remove('football')
    color_map = {teams[0]: team_colors[teams[0]], teams[1]: team_colors[teams[1]], 'football': '#dec000'}

    # Add ball
    football = frame[frame['team'] == 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] == 'football']
    ax.scatter(football['x'], football['y'], c='#dec000', s=500 if zoom else 25, marker='o',zorder=3.1)

    # Add players
    players = frame[frame['team'] != 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] != 'football']
    ax.scatter(players['x'], players['y'], c=players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map), s=1000 if zoom else 70, zorder=3)

    # Add SPSP indicator around receiver
    receiver_row = frame[frame['nflId'] == receiver_id]
    if not receiver_row.empty:
        if spsp_rolling_avg >= 0.7:
            indicator_color = '#15FF00'
        elif spsp_rolling_avg < 0.4 and spsp_rolling_avg > 0:
            indicator_color = '#FF2D2D'
        elif spsp_rolling_avg >= 0.4 and spsp_rolling_avg < 0.7:
            indicator_color = "#FFA02D"
        else:
            indicator_color = "#696969"

        ax.scatter(
            receiver_row['x'], 
            receiver_row['y'], 
            s=1100 if zoom else 100, 
            facecolors='none', 
            edgecolors=indicator_color, 
            linewidths=2, 
            zorder=4
        )

    # Convert angles to radians
    angles = np.deg2rad(players['o'].fillna(0))
    dx = np.sin(angles)   # X-component of direction
    dy = np.cos(angles)   # Y-component of direction

    # Offset distance: approximate radius of player circle
    marker_radius = 1.2 if zoom else 0.4

    # Apply offset to starting positions
    x_offset = players['x'] + dx * marker_radius
    y_offset = players['y'] + dy * marker_radius

    # Arrow length
    arrow_length = 0.8#1.0 if zoom else 0.5

    ax.quiver(
        x_offset, y_offset,   # Offset starting points
        dx, dy,               # Arrow directions
        angles='xy',
        scale_units='xy',
        scale=1 / arrow_length,
        color=players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map),
        width=0.005,#0.01 if zoom else 0.0015,
        zorder=3
    )

    # Add jersey numbers
    for _, row in frame.iterrows():
        # Only plot the labels of players in frame
        if (row['x'] > ball_x-zoom_offset_x and row['x'] <= ball_x+zoom_offset_x) and (row['y'] > ball_y-zoom_offset_y and row['y'] <= ball_y+zoom_offset_y) or not zoom:
            label = '' if math.isnan(row['jerseyNumber']) else int(row['jerseyNumber'])
            ax.text(row['x'] + (0.6 if zoom else 0.5), row['y'], label, fontsize=16 if zoom else 8, zorder=4)

    # Field settings
    if zoom:
        plt.xlim(ball_x - zoom_offset_x, ball_x + zoom_offset_x)
        plt.ylim(ball_y - zoom_offset_y, ball_y + zoom_offset_y)
    else:
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)

    title = f"game: {play_data['gameId']}, play: {play_data['playId']}, frame: {frame['frameId'].iloc[0]}, event: {str(frame['event'].iloc[0])}"
    fig.suptitle(title, fontsize=18)

    suffixes = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th'}

    try:
        possession_team = play_data['possessionTeam']
        defensive_team = play_data['defensiveTeam']
    except:
        possession_team = 'Team1'
        defensive_team = 'Team2'

    if str(play_data['gameId']).startswith('2018'):
        yards_gained = play_data['playResult']
    elif str(play_data['gameId']).startswith('2021'):
        yards_gained = play_data['prePenaltyPlayResult']
    elif str(play_data['gameId']).startswith('2022'):
        yards_gained = play_data['yardsGained']

    play_state = f"{possession_team} vs. {defensive_team}, Q{play_data['quarter']} {play_data['gameClock']}, {play_data['down']}{suffixes[play_data['down']]} & {play_data['yardsToGo']}"
    play_state += f", yardsGained: {yards_gained}, {spsp_prob*100:.2f}% SPSP ({spsp_rolling_avg*100:.2f}% rolling)"
    fig.text(0.5, 0.90, play_state, ha='center', fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"play_frames/{file_name}/{file_name}_{frame['frameId'].iloc[0]:04d}{'_zoomed' if zoom else ''}.png")
    plt.close()


def plot_frame_prob(frame_id, spsp_prob_per_frame, receiver_id, prob_count, file_name):
    fig, ax = plt.subplots()

    # Color bands
    ax.axhspan(0.7, 1.0, facecolor='lightgreen', alpha=0.3)
    ax.axhspan(0.4, 0.7, facecolor='orange', alpha=0.2)
    ax.axhspan(0.0, 0.4, facecolor='lightcoral', alpha=0.3)

    prob = spsp_prob_per_frame[:prob_count+1]
    rolling_probs = get_rolling_avg(spsp_prob_per_frame)
        
    ax.plot(range(prob_count+1), rolling_probs[:prob_count+1], color='black', marker='o') #spsp_prob_per_frame[:prob_count + 1]

    # Set fixed axis limits
    x_min, x_max = 0, len(spsp_prob_per_frame)-1
    y_min, y_max = 0.1, 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Seconds After Snap')
    ax.set_ylabel('Success Probability')
    ax.set_title(f'Success Probability of Receiver={receiver_id} over Time')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'play_prob_frames/{file_name}_probs/{file_name}_{frame_id:04d}_probs.png')
    plt.close()



def create_play_gif(play_data, frames: DataFrame, spsp_prob_per_frame, receiver_id, qb_id, file_name, zoom=False, loop=True, delete_frame_plots=False):
    frame_start = frames['frameId'].min()
    frame_end = frames['frameId'].max()

    print('frame_start:', frame_start)
    print('frame_end:', frame_end)

    # Create new folder for frame plots
    folder_name = f'play_frames/{file_name}'
    os.makedirs(folder_name, exist_ok=True)

    folder_name = f'play_prob_frames/{file_name}_probs'
    os.makedirs(folder_name, exist_ok=True)

    print('creating gif...')

    # Create a plot for every frame in the range
    prob_count = 0
    for frame_id in range(frame_start, frame_end+1):
        frame = frames[frames['frameId'] == frame_id]

        # Take the rolling average (window size = 3) of SPSP 
        if prob_count > 1:
            prob_rolling_avg = sum(spsp_prob_per_frame[prob_count-2:prob_count+1]) / 3
        else:
            prob_rolling_avg = 0.0

        plot_frame_simple(frame, play_data, spsp_prob_per_frame[prob_count], prob_rolling_avg, receiver_id, qb_id, file_name)
        plot_frame_prob(frame_id, spsp_prob_per_frame, receiver_id, prob_count, file_name)
        prob_count += 1
    
    frames_folder = f'play_frames/{file_name}'
    gif_output_path = f"play_gifs/{file_name}{'_zoomed' if zoom else ''}.gif"

    # Get list of image filenames in sorted order
    frame_files = sorted([
        os.path.join(frames_folder, fname)
        for fname in os.listdir(frames_folder)
        if fname.endswith('.png')
    ])

    # Create and save GIF
    loops = 0 if loop else 1
    with imageio.get_writer(gif_output_path, mode='I', duration=0.1, loop=loops) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Delete individual frame plots when completed
    if delete_frame_plots:
        if os.path.exists(frames_folder):
            shutil.rmtree(frames_folder)
            print(f'Deleted folder: {frames_folder}')
        else:
            print(f'Folder not found: {frames_folder}')

    print('gif created')


def convert_gif_to_mp4(gif_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-i", gif_path,
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Converted: {gif_path} → {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")


def get_rolling_avg(prob_list, window_size=3):
    rolling_avg = []
    for i in range(len(prob_list)):
        if i < window_size - 1:
            rolling_avg.append(None)
        else:
            avg = sum(prob_list[i - window_size + 1:i + 1]) / window_size
            rolling_avg.append(avg)

    return rolling_avg



def rotate_frame_90ccw(frame: DataFrame):
    # Swap the coordinates (no vertical flip)
    rot = frame.copy()
    rot['x'] = frame['y']           # new horizontal coordinate
    rot['y'] = frame['x']           # new vertical coordinate

    # Rotate the heading
    if 'o' in rot.columns:
        # rotate the heading in the opposite direction.
        rot['o'] = (90 - frame['o'].fillna(0)) % 360

    return rot


def plot_frame_simple(frame, play_data, spsp_prob, spsp_rolling_avg, receiver_id, qb_id, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))

    frame = rotate_frame_90ccw(frame)

    ax.set_facecolor('#FFFFFF')#mediumseagreen, #4CCE87

    # Field settings
    offset_backfield = 20#15
    offset_upfield = 20#25
    plt.ylim(constants.MIDFIELD - offset_backfield, constants.MIDFIELD + offset_upfield)
    plt.xlim(0, constants.FIELD_WIDTH)

    possession_team = play_data['possessionTeam']
    defensive_team = play_data['defensiveTeam']

    off_color = '#FFFFFF'#team_colors[play_data['possessionTeam']], #434AFF'
    def_color = '#000000'#team_colors[play_data['defensiveTeam']]

    line_color = '#DCDCDC'

    # Draw yard lines every 5 yards (horizontal now)
    for y in range(int(constants.MIDFIELD - offset_backfield), int(constants.MIDFIELD + offset_upfield), 5):
        ax.axhline(y=y, color=line_color, linewidth=2, zorder=2)

    # Draw hashmarks every yard
    hashmark_width = 0.5
    for y in range(int(constants.MIDFIELD - offset_backfield), int(constants.MIDFIELD + offset_upfield), 1):
        # Left hashmarks
        ax.plot([constants.SIDELINE_TO_HASH - hashmark_width/2, constants.SIDELINE_TO_HASH + hashmark_width/2],
                [y, y], color=line_color, linewidth=2, zorder=2)

        # Right hashmarks
        ax.plot([constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH - hashmark_width/2, constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH + hashmark_width/2],
                [y, y], color=line_color, linewidth=2, zorder=2)
        
        # Left sideline hashmarks
        ax.plot([1, hashmark_width], [y, y], color=line_color, linewidth=2, zorder=2)
        
        # Right sideline hashmarks
        ax.plot([constants.FIELD_WIDTH - hashmark_width, constants.FIELD_WIDTH - 1], [y, y], color=line_color, linewidth=2, zorder=2)


    # 1st down marker (vertical field)
    ax.axhline(y=constants.MIDFIELD + play_data['yardsToGo'], color="#f2d627", linewidth=2.5, zorder=2.2)

    # Line of scrimmage
    ax.axhline(y=constants.MIDFIELD, color="#384bf6", linewidth=2.5, zorder=2.2)

    teams = frame['team'].unique().tolist() if ('club' not in frame.columns or frame['club'].isna().all()) else frame['club'].unique().tolist()
    teams.remove('football')
    color_map = {}
    color_map[possession_team] = off_color
    color_map[defensive_team] = def_color 

    # Plot football
    football = frame[frame['team'] == 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] == 'football']
    ax.scatter(football['x'], football['y'], c="#a87a2f", s=60, marker='o', zorder=6)
    
    players = frame[frame['team'] != 'football'] if ('club' not in frame.columns or frame['club'].isna().all()) else frame[frame['club'] != 'football']
    off_players = players[(players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'] == possession_team) &
                          (players['nflId'] != receiver_id) &
                          (players['nflId'] != qb_id)]
    def_players = players[players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'] == defensive_team]
    receiver = players[players['nflId'] == receiver_id]
    qb = players[players['nflId'] == qb_id]

    # Plot offensive players
    ax.scatter(
        off_players['x'], 
        off_players['y'], 
        c=off_players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map), 
        s=225,
        edgecolors='black',
        linewidths=1.5, 
        zorder=4
    )

    # Plot defensive players
    ax.scatter(
        def_players['x'], 
        def_players['y'], c=def_players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map), 
        s=225,
        zorder=3.8
    )
        
    marker_radius = 0.6
    arrow_length = 0.75

    # Offensive Arrows
    ########################################################
    # Convert angles to radians
    angles = np.deg2rad(off_players['o'].fillna(0))
    dx = np.sin(angles)   
    dy = np.cos(angles)   
    
    x_offset = off_players['x'] + dx * marker_radius
    y_offset = off_players['y'] + dy * marker_radius

    ax.quiver(
        x_offset, y_offset,   # Offset starting points
        dx, dy,               # Arrow directions
        angles='xy',
        scale_units='xy',
        scale=1 / arrow_length,
        color=off_players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map),
        edgecolor='black',
        linewidth=1.5,
        width=0.005,
        zorder=3.9
    )

    # Defensive arrows
    ########################################################
    # Convert angles to radians
    angles = np.deg2rad(def_players['o'].fillna(0))
    dx = np.sin(angles)   
    dy = np.cos(angles)   

    x_offset = def_players['x'] + dx * marker_radius
    y_offset = def_players['y'] + dy * marker_radius

    ax.quiver(
        x_offset, y_offset,   # Offset starting points
        dx, dy,               # Arrow directions
        angles='xy',
        scale_units='xy',
        scale=1 / arrow_length,
        color=def_players['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map),
        width=0.005,
        zorder=3.9
    )


    if not receiver.empty:
        if spsp_rolling_avg >= 0.7:
            indicator_color = "#59FF4A"
        elif spsp_rolling_avg < 0.4 and spsp_rolling_avg > 0:
            indicator_color = "#FF6060"
        elif spsp_rolling_avg >= 0.4 and spsp_rolling_avg < 0.7:
            indicator_color = "#FFA83F"
        else:
            indicator_color = "none"

        ax.scatter(receiver['x'], receiver['y'], s=400,
                   facecolors='none', edgecolors=indicator_color,
                   linewidths=5, zorder=2.9)
        
    # Plot QB
    ax.scatter(
        qb['x'], 
        qb['y'], 
        c=qb['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map), 
        s=225,
        edgecolors='black',
        linewidths=1.5, 
        zorder=5
    )

    # Plot receiver
    ax.scatter(
        receiver['x'], 
        receiver['y'], 
        c=receiver['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map), 
        s=225,
        edgecolors='black',
        linewidths=1.5, 
        zorder=5
    )

    # QB Arrow
    ########################################################
    # Convert angles to radians
    angles = np.deg2rad(qb['o'].fillna(0))
    dx = np.sin(angles)   
    dy = np.cos(angles)   
    
    x_offset = qb['x'] + dx * marker_radius
    y_offset = qb['y'] + dy * marker_radius

    ax.quiver(
        x_offset, y_offset,   # Offset starting points
        dx, dy,               # Arrow directions
        angles='xy',
        scale_units='xy',
        scale=1 / arrow_length,
        color=qb['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map),
        edgecolor='black',
        linewidth=1.5,
        width=0.005,
        zorder=4.9
    )

    # Receiver Arrow
    ########################################################
    # Convert angles to radians
    angles = np.deg2rad(receiver['o'].fillna(0))
    dx = np.sin(angles)   
    dy = np.cos(angles)   
    
    x_offset = receiver['x'] + dx * marker_radius
    y_offset = receiver['y'] + dy * marker_radius

    ax.quiver(
        x_offset, y_offset,   # Offset starting points
        dx, dy,               # Arrow directions
        angles='xy',
        scale_units='xy',
        scale=1 / arrow_length,
        color=receiver['team' if ('club' not in frame.columns or frame['club'].isna().all()) else 'club'].map(color_map),
        edgecolor='black',
        linewidth=1.5,
        width=0.005,
        zorder=4.9
    )


    # Plot dotted line (solid if pass_forward frame) from QB to receiver
    plt.plot(
        [receiver['x'], qb['x']], 
        [receiver['y'], qb['y']], 
        linestyle='solid' if str(frame.iloc[0]['event']) in constants.PASS_FORWARD else ':', 
        color=indicator_color, 
        linewidth=4,
        zorder=4.8
    )



    # for _, row in frame.iterrows():
    #     if (row['x'] > ball_x - zoom_offset_x and row['x'] <= ball_x + zoom_offset_x) and (row['y'] > ball_y - zoom_offset_y and row['y'] <= ball_y + zoom_offset_y) or not zoom:
    #         label = '' if math.isnan(row['jerseyNumber']) else int(row['jerseyNumber'])
    #         ax.text(row['x'] + (0.6 if zoom else 0.5), row['y'], label, fontsize=16 if zoom else 8, zorder=4)

    

    suffixes = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th'}

    if str(play_data['gameId']).startswith('2021'):
        yards_gained = play_data['prePenaltyPlayResult']
    elif str(play_data['gameId']).startswith('2022'):
        yards_gained = play_data['yardsGained']

    # play_state = f"{possession_team} vs. {defensive_team}, Q{play_data['quarter']} {play_data['gameClock']}, {play_data['down']}{suffixes[play_data['down']]} & {play_data['yardsToGo']}"
    # play_state += f", yardsGained: {yards_gained}, {spsp_prob*100:.2f}% SPSP ({spsp_rolling_avg*100:.2f}% rolling)"
    title = f"{play_data['down']}{suffixes[play_data['down']]} & {play_data['yardsToGo']}, {spsp_prob*100:.2f}% SPSP ({spsp_rolling_avg*100:.2f}% rolling)"
    title += f"yardsGained:{yards_gained}, Actual Success:{yards_gained >= play_data['yardsToGo']}"
    fig.suptitle(title, fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"play_frames/{file_name}/{file_name}_{frame['frameId'].iloc[0]:04d}.png")
    plt.close()



team_colors = {
    'ARI': '#97233F',
    'ATL': '#A71930',
    'BAL': '#241773',
    'BUF': '#00338D',
    'CAR': '#0085CA',
    'CHI': '#0B162A',
    'CIN': '#FB4F14',
    'CLE': '#311D00',
    'DAL': '#003594',
    'DEN': '#FB4F14',
    'DET': '#0076B6',
    'GB':  '#203731',
    'HOU': '#03202F',
    'IND': '#002C5F',
    'JAX': '#006778',
    'KC':  '#E31837',
    'LV':  '#000000',
    'LAC': '#2472ca',
    'LA': '#003594',
    'MIA': '#008E97',
    'MIN': '#4F2683',
    'NE':  '#002244',
    'NO':  '#D3BC8D',
    'NYG': '#0B2265',
    'NYJ': '#125740',
    'PHI': '#004a50',
    'PIT': '#FFB612',
    'SEA': '#69BE28',
    'SF':  '#AA0000',
    'TB':  '#D50A0A',
    'TEN': '#4B92DB',
    'WAS': '#773141',
    'home': '#000000',
    'away': '#FFFFFF'
}