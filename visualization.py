import matplotlib.pyplot as plt
import constants
import math
from pandas import DataFrame
import os
import imageio.v2 as imageio
import shutil

def plot_frame(frame, play_data, file_name, zoom):#frames, play_data, file_name, zoom=True):
    # frame_id = frames['frameId'].iloc[-1]
    # print('plotting frame_id:', frame_id)
    # frame = frames[frames['frameId'] == frame_id]

    fig, ax = plt.subplots(figsize=(12, 7.5 if zoom else 6))

    ball = frame[frame['displayName'] == 'football'].iloc[0]
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

    # Draw Center Field and Goalines
    ax.axvline(x=constants.CENTER_FIELD, color='white', linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.OFF_GOALLINE, color='white', linewidth=6 if zoom else 2, zorder=2.1)
    ax.axvline(x=constants.DEF_GOALLINE, color='white', linewidth=6 if zoom else 2, zorder=2.1)

    # Draw hash marks
    ax.axhline(y=constants.SIDELINE_TO_HASH, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)
    ax.axhline(y=constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH, color='white', linestyle='dotted', linewidth=6 if zoom else 2, zorder=0)

    # Handle team colors
    teams = frame['club'].unique().tolist()
    teams.remove('football')
    color_map = {teams[0]: team_colors[teams[0]], teams[1]: team_colors[teams[1]], 'football': '#dec000'}

    # Add ball
    football = frame[frame['displayName'] == 'football']
    ax.scatter(football['x'], football['y'], c='#dec000', s=500 if zoom else 25, marker='o',zorder=3.1)

    # Add players
    players = frame[frame['displayName'] != 'football']
    ax.scatter(players['x'], players['y'], c=players['club'].map(color_map), s=1000 if zoom else 60, zorder=3)

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
    fig.text(0.5, 0.90, play_state, ha='center', fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"plots/{file_name}/{file_name}_{frame['frameId'].iloc[0]:04d}{'_zoomed' if zoom else ''}.png")
    plt.close()


def create_play_gif(play_data, frames: DataFrame, file_name, zoom=False, loop=True):
    # print('play_data:', play_data)
    # print('frames:\n', frames)

    frame_start = frames['frameId'].min()
    frame_end = frames['frameId'].max()

    print('frame_start:', frame_start)
    print('frame_end:', frame_end)

    # Create new folder for frame plots
    folder_name = f'plots/{file_name}'
    os.makedirs(folder_name, exist_ok=True)

    print('creating gif...')

    # Create a plot for every frame in the range
    for frame_id in range(frame_start, frame_end+1):
        frame = frames[frames['frameId'] == frame_id]
        plot_frame(frame, play_data, file_name, zoom=zoom)
    
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