# # Filter to include only plays that contain a 'handoff' event
# handoff_plays = all_tracking_df[all_tracking_df['event'] == 'handoff'][['gameId', 'playId']]
# handoff_play_ids = set(zip(handoff_plays['gameId'], handoff_plays['playId']))
# run_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in handoff_play_ids, axis=1)]
# run_tracking_data = data_processing.filter_tracking_data(all_tracking_data, run_play_data)
# # run_tracking_data = data_processing.normalize_field_direction(run_tracking_data)

# # Filter to include only plays that contain a 'play_action' event
# play_action_plays = all_tracking_df[all_tracking_df['event'] == 'play_action'][['gameId', 'playId']]
# play_action_play_ids = set(zip(play_action_plays['gameId'], play_action_plays['playId']))
# play_action_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in play_action_play_ids, axis=1)]
# play_action_tracking_data = data_processing.filter_tracking_data(all_tracking_data, play_action_play_data)
# play_action_tracking_data = data_processing.normalize_field_direction(play_action_tracking_data)

# # Filter to include only RPO plays
# rpo_play_data = all_play_data[all_play_data['pff_runPassOption'] == 1]
# rpo_tracking_data = data_processing.filter_tracking_data(all_tracking_data, rpo_play_data)
# rpo_tracking_data = data_processing.normalize_field_direction(rpo_tracking_data)

# # Filter to include only TRADITIONAL dropback pass plays that have at least 2 seconds of timeInTackleBox
# passing_play_data = all_play_data[(all_play_data['passResult'].notna()) & 
#                                   (all_play_data['timeInTackleBox'] >= 2.0) &
#                                   (all_play_data['dropbackType'] == 'TRADITIONAL')]

# print('ALL RUSH EVENTS:', run_tracking_data[0]['event'].value_counts())
# print('ALL RUSH O FORMATIONS:', run_play_data['offenseFormation'].value_counts())
# print('ALL RUSH D FORMATIONS:', run_play_data['pff_passCoverage'].value_counts())
# print('ALL RUSH rushLocationType:', run_play_data['rushLocationType'].value_counts())
# print('ALL RUSH pff_runConceptPrimary:', run_play_data['pff_runConceptPrimary'].value_counts())
# print('ALL RUSH pff_runConceptSecondary:', run_play_data['pff_runConceptSecondary'].value_counts())
# print('ALL PA EVENTS:', play_action_tracking_data[0]['event'].value_counts())
# print('ALL PA O FORMATIONS:', play_action_play_data['offenseFormation'].value_counts())
# print('ALL PA D FORMATIONS:', play_action_play_data['pff_passCoverage'].value_counts())
# print('ALL RPO EVENTS:', rpo_tracking_data[0]['event'].value_counts())
# print('ALL RPO O FORMATIONS:', rpo_play_data['offenseFormation'].value_counts())
# print('ALL RPO D FORMATIONS:', rpo_play_data['pff_passCoverage'].value_counts())
# print('ALL PASSING DROPBACK TYPES:\n', passing_play_data['dropbackType'].value_counts())
# print('AVG TIME TO THROW:', passing_play_data['timeToThrow'].mean())
# print('AVG timeToPressureAsPassRusher:', all_player_play_data['timeToPressureAsPassRusher'].mean())
# print('AVG getOffTimeAsPassRusher:', all_player_play_data['getOffTimeAsPassRusher'].mean())

# # Calculate how "hot" a pocket is
# def calculate_pocket_heat(plays_to_process, all_player_data, all_player_play_data, all_tracking_df):

#     # EXTREMELY CLEAN POCKET: (2022091113, 630), (2022091101, 1879), (2022091104, 2952)
#             # IMMEDIATE PRESSURE FROM 1 DEFENDER: (2022091109, 1041)
#             # TOTAL COLLAPSE: (2022091108, 614)
#             # Scored very low: (2022091101, 1492)
#             # Starts off completely clean, but then has an unblocked rusher (not labelled): (2022091101, 1166)
#             # Moderate pressure?: (2022091105, 2817)
#             # Scored average, but is very good protection: (2022091108, 2799)
#             # Avg score but feels like it should be higher (2022091107, 1642)
#             # Dont know why it statistically has a high score: (2022091106, 3050)
#             # Very high score: (2022091111, 336)
#             # Moderate pressure: (2022091105, 2817)
#     defense_rush_positions = ['CB', 'OLB', 'DE', 'DT', 'ILB', 'FS', 'SS', 'NT', 'MLB', 'DB', 'LB']
#     all_def_players = all_player_data[all_player_data['position'].isin(defense_rush_positions)]['nflId'].unique()
#     for i,passing_play in plays_to_process.iterrows():
#         game_id = passing_play['gameId']
#         play_id = passing_play['playId']

#         play_tracking = all_tracking_df[(all_tracking_df['gameId'] == game_id) & (all_tracking_df['playId'] == play_id)]
#         players_in_play = play_tracking['nflId'].dropna().unique()
#         def_players_in_play = np.intersect1d(players_in_play, all_def_players)

#         # Calculate the time it took from ball-snap to either timeToThrow or timeToSack
#         time_to_pass_result = passing_play['timeToThrow'] if not math.isnan(passing_play['timeToThrow']) else passing_play['timeToSack']

#         # Extract "pressure" features
#         rushing_defenders = []
#         rushing_defenders_pressure_caused = []
#         total_times_to_pressure = []
#         total_times_to_getoff = []

#         for def_id in def_players_in_play:
#             player= all_player_data[all_player_data['nflId'] == def_id].iloc[0]

#             def_play_data = all_player_play_data[
#                 (all_player_play_data['gameId'] == game_id) &
#                 (all_player_play_data['playId'] == play_id) &
#                 (all_player_play_data['nflId'] == def_id)
#             ].iloc[0]

#             caused_pressure = def_play_data['causedPressure']
#             time_to_pressure_as_rusher = def_play_data['timeToPressureAsPassRusher']
#             get_off_as_rusher = def_play_data['getOffTimeAsPassRusher']

#             # Count number of rushers (who have a non-NaN value for 'getOffTimeAsPassRusher')
#             if not math.isnan(get_off_as_rusher):
#                 rushing_defenders.append(def_id)
#                 total_times_to_getoff.append(get_off_as_rusher)

#                 if caused_pressure:
#                     rushing_defenders_pressure_caused.append(def_id)
#                     total_times_to_pressure.append(time_to_pressure_as_rusher)

#             # print(f"{player['displayName']} play data:")
#             # print(f'\tcaused_pressure: {caused_pressure}')
#             # print(f'\ttime_to_pressure_as_rusher: {time_to_pressure_as_rusher}')
#             # print(f'\tget_off_as_rusher: {get_off_as_rusher}')


#         # If there was no pressure on the play, assign arbitrary large value
#         if len(total_times_to_pressure) == 0:
#             total_times_to_pressure = [5]


#         # Ratio of rushers to rushers that caused pressure
#         # pc_heat_val = len(rushing_defenders_pressure_caused) / len(rushing_defenders) * 10
#         pc_heat_val = 10 * len(rushing_defenders_pressure_caused)

#         avg_time_to_pressure = np.round(np.sum(total_times_to_pressure) / len(rushing_defenders), 2)
#         thresh = 2 # to represent 'quick' throw
#         # ttp_heat_val = np.round(1 - (avg_time_to_pressure / max_time_to_pass), 2)
#         ttp_heat_val = 0 #np.round(np.clip(1 / (1 + np.exp(avg_time_to_pressure - thresh)), 0, 1), 2)


#         # The max amount of time it would take to cross LoS as rusher
#         # TODO: try and estimate this number
#         max_getoff_time = 2
#         avg_time_to_getoff = np.round(np.median(total_times_to_getoff) / len(rushing_defenders), 2) #median instead of mean for robustness to outliers (e.g., one slow rusher)
#         ttg_heat_val = np.round(1 - (avg_time_to_getoff / max_getoff_time), 2) * 10

#         # Time to throw
#         # Did QB have to bail from the pocket?
#         # ttt_heat_val = np.round(1 - (passing_play['timeInTackleBox'] / time_to_pass_result), 2)
#         ttt_heat_val = np.round(1 - np.clip(passing_play['timeInTackleBox'] / time_to_pass_result, 0, 1) ** 0.5, 2) * 10 # square root emphasizes short times

        

#         # Outcome
#         # Max time limit for a clean pocket
#         # TODO: try and estimate this number
#         # max_time_to_pass = 5
#         # # outcome_heat_val = np.round(1 - (passing_play['timeToThrow'] if not math.isnan(passing_play['timeToThrow']) else passing_play['timeToSack'] / max_time_to_pass), 2)
#         # # outcome_heat_val = np.clip(outcome_heat_val, 0, 1)
#         # if not math.isnan(passing_play['timeToSack']):
#         #     outcome_heat_val = np.round(passing_play['timeToSack'] / max_time_to_pass, 2)
#         #     outcome_heat_val += 0.2 # boosted for sacks
#         # else:
#         #     outcome_heat_val = np.round(passing_play['timeToThrow'] / max_time_to_pass, 2)
#         #     outcome_heat_val *= 0.5 # half-weight for non-sacks, as quick throws indicate indirect heat
#         # outcome_heat_val = np.clip(outcome_heat_val, 0, 1)

#         # TODO: Add penalty for # of defenders within 1-2 yards at the time of the league average timeToThrow (2.7)

#         penalties = 0
#         if passing_play['unblockedPressure']:
#             print('Unblocked pressure! (10 heat val)')
#             penalties += 10

        
#         # TODO: Add bonus for time over league average (2.7) while remaining in the Tackle Box

#         # TODO: Alternate idea: Make the "pocket_heat" value per frame, so the model can see the pocket_heat change over the course of the play
#         #       Use the distance of the QB to every rusher at every frame to assign score

        


        
#         print('==========================================================================================')
#         print(f'# OF RUSHERS ON ({game_id}, {play_id}): {len(rushing_defenders)} ({len(rushing_defenders_pressure_caused)} caused pressure) ({pc_heat_val} heat val)')
#         print(f'Time to pass result: {time_to_pass_result}')
#         print(f'Avg time to pressure for rushing defenders: {avg_time_to_pressure} ({ttp_heat_val} heat val)')
#         print(f'Avg time to getoff for rushing defenders: {avg_time_to_getoff} ({ttg_heat_val} heat val)')
#         print(f"QB time in tackle box: {passing_play['timeInTackleBox']} ({ttt_heat_val} heat val)")
#         print(f"Penalties ({penalties} heat val)")
#         # print(f"Outcome: {'SACK' if not math.isnan(passing_play['timeToSack']) else 'NO SACK'} ({outcome_heat_val})")
#         print(f'TOTAL HEAT VAL: {np.round(pc_heat_val + ttp_heat_val + ttg_heat_val + ttt_heat_val + penalties, 2)}')


# # Input: 11 defenders, Output: defenders that are rushers
# # Method: Defenders that are within 3-4 yards of the LoS that move towards the QB in the first ~1.5 seconds
# def detect_rushers(all_def_players, tracking_data, ball_coord, qb_coord):
#     print('PROCESSING RUSHERS')

#     ball_x, ball_y = ball_coord
#     qb_x, qb_y = qb_coord

#     # Observe difference in defenders positions/velocity from the snap and 1.5 seconds later
#     time_delay = 15 # 1.5 sec * 10 frames/sec
#     start_frame = tracking_data['frameId'].min()
#     end_frame = min(tracking_data['frameId'].max(), start_frame+time_delay)
#     print(f'start:{start_frame}, end:{end_frame}')

#     # Filter play tracking data to only include defenders
#     frame_defenders = tracking_data[tracking_data['nflId'].isin(all_def_players)]
#     print(frame_defenders)

#     # Get defender tracking data at the ball snap
#     starting_positions = frame_defenders[frame_defenders['frameId'] == start_frame]
#     print('starting_positions:\n', starting_positions)

#     # Get defender tracking data 1.5 seconds after ball snap
#     ending_positions = frame_defenders[frame_defenders['frameId'] == end_frame]
#     print('ending_positions:\n', ending_positions)

#     los_dist_thresh = 5.0
#     close_to_los = starting_positions[np.abs(starting_positions['x'] - ball_x) <= los_dist_thresh]
#     print('close_to_los:\n', close_to_los)