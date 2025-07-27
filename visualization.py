import matplotlib.pyplot as plt
import constants
import math
from pandas import DataFrame
import os
import imageio.v2 as imageio
import shutil
import numpy as np

def plot_frame(frame, play_data, spsp_prob, spsp_rolling_avg, receiver_id, file_name, zoom):
    fig, ax = plt.subplots(figsize=(12, 7.5 if zoom else 6.5))

    ball = frame[frame['team'] == 'football'].iloc[0] if frame['club'].isna().all() else frame[frame['club'] == 'football'].iloc[0]
    ball_x = ball['x']
    ball_y = ball['y']

    zoom_offset_x = 15
    zoom_offset_y = 8

    # Set green background for the field
    ax.set_facecolor('mediumseagreen')

    off_color = team_colors[play_data['possessionTeam']]
    def_color = team_colors[play_data['defensiveTeam']]

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
    teams = frame['team'].unique().tolist() if frame['club'].isna().all() else frame['club'].unique().tolist()
    teams.remove('football')
    color_map = {teams[0]: team_colors[teams[0]], teams[1]: team_colors[teams[1]], 'football': '#dec000'}

    # Add ball
    football = frame[frame['team'] == 'football'] if frame['club'].isna().all() else frame[frame['club'] == 'football']
    ax.scatter(football['x'], football['y'], c='#dec000', s=500 if zoom else 25, marker='o',zorder=3.1)

    # Add players
    players = frame[frame['team'] != 'football'] if frame['club'].isna().all() else frame[frame['club'] != 'football']
    ax.scatter(players['x'], players['y'], c=players['team' if frame['club'].isna().all() else 'club'].map(color_map), s=1000 if zoom else 60, zorder=3)

    # Add SPSP indicator around receiver
    receiver_row = frame[frame['nflId'] == receiver_id]
    if not receiver_row.empty:
        if spsp_rolling_avg >= 0.6:
            indicator_color = '#15FF00'
        elif spsp_rolling_avg < 0.4:
            indicator_color = '#FF2D2D'
        elif spsp_rolling_avg >= 0.4 and spsp_rolling_avg < 0.6:
            indicator_color = "#FFA02D"
        else:
            indicator_color = "#696969"

        ax.scatter(
            receiver_row['x'], 
            receiver_row['y'], 
            s=1100 if zoom else 80, 
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
    # Adjust scaling for zoom vs. full field
    marker_radius = 1.2 if zoom else 0.4  # tweak until it looks right

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
        color=players['team' if frame['club'].isna().all() else 'club'].map(color_map),
        width=0.005,#0.01 if zoom else 0.0015,
        zorder=4
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
    play_state = f"{play_data['possessionTeam']} vs. {play_data['defensiveTeam']}, Q{play_data['quarter']} {play_data['gameClock']}, {play_data['down']}{suffixes[play_data['down']]} & {play_data['yardsToGo']}"
    play_state += f", yardsGained: {play_data['playResult'] if '2021' in str(play_data['gameId']) else play_data['yardsGained']}, {spsp_prob*100:.2f}% SPSP ({spsp_rolling_avg*100:.2f}% rolling)"
    fig.text(0.5, 0.90, play_state, ha='center', fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"plots/{file_name}/{file_name}_{frame['frameId'].iloc[0]:04d}{'_zoomed' if zoom else ''}.png")
    plt.close()


def create_play_gif(play_data, frames: DataFrame, spsp_prob_per_frame, receiver_id, file_name, zoom=False, loop=True):
    frame_start = frames['frameId'].min()
    frame_end = frames['frameId'].max()

    print('frame_start:', frame_start)
    print('frame_end:', frame_end)

    # Create new folder for frame plots
    folder_name = f'plots/{file_name}'
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

        plot_frame(frame, play_data, spsp_prob_per_frame[prob_count], prob_rolling_avg, receiver_id, file_name, zoom=zoom)
        prob_count += 1
    
    frames_folder = f'plots/{file_name}'
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
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
        print(f'Deleted folder: {frames_folder}')
    else:
        print(f'Folder not found: {frames_folder}')

    print('gif created')


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
}