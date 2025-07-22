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