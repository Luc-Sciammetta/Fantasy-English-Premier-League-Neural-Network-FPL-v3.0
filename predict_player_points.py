import requests
import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn

from points_predictors.oneGW_points_predictor import FPLModelOneGW
from points_predictors.total_points_predictor import FPLModel

import pprint


def getPlayersFromAPI():
    """
    Gets all the players for the season and their total/general season stats from the API and returns them as a dataframe
    """
    url = 'https://fantasy.premierleague.com/api/'
    P = requests.get(url+'bootstrap-static/').json()

    players = P['elements']
    players = sorted(players, key=lambda x: x['id'])

    df = pd.DataFrame(players)

    features = [
        'minutes', 'starts', 'starts_per_90',
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'own_goals',
        'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
        'expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90',
        'saves', 'saves_per_90', 'penalties_saved', 'penalties_missed',
        'bonus', 'bps', 'total_points', 'points_per_game',

        'form', 'ep_next', 'ep_this',
        'selected_by_percent',
        'transfers_in_event', 'transfers_out_event',
        'now_cost',
        'status', 'chance_of_playing_next_round', 'chance_of_playing_this_round',

        'id', 'element_type', 'team', 'first_name', 'second_name',

        'ict_index',
    ]  

    # for _, player in df.iterrows():
    #     print(player['first_name'], player['second_name'], player['team'])

    df = df[features]
    return df


def getPlayerStatFromAPI(player_id):
    """
    Gets the data from a specific player for each gameweek and returns it as a dataframe
    """
    url = 'https://fantasy.premierleague.com/api/'
    P = requests.get(url+f'element-summary/{player_id}/').json()

    history = P['history']
    history = sorted(history, key=lambda x: x['round'])

    df = pd.DataFrame(history)

    features = [
        'element', 'fixture', 'opponent_team', 'total_points', 'was_home', 'team_h_score', 'team_a_score', 'round', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'clearances_blocks_interceptions', 'recoveries', 'tackles', 'defensive_contribution', 'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded', 'value', 'selected', 'transfers_in', 'transfers_out'
        ]

    df = df[features]
    return df


def getFixturesFromAPI():
    """
    Gets all the fixtures for the season from the API and returns them as a dataframe
    """
    url = 'https://fantasy.premierleague.com/api/'
    F = requests.get(url+'fixtures/').json()

    fixtures = F
    fixtures = sorted(fixtures, key=lambda x: x['id'])

    df = pd.DataFrame(fixtures)

    features = [
        'id', 'event', 'finished', 'team_h', 'team_a', 'team_h_score', 'team_a_score',
        'team_a_difficulty', 'team_h_difficulty', 'stats'
    ]

    df = df[features]
    return df


def getPlayerFromName(first, last):
    """
    Gets a player's season and gamebygame stats from their first and last name
    """
    df_season = getPlayersFromAPI()
    player = df_season[(df_season['first_name'] == first) & (df_season['second_name'] == last)]
    if len(player) == 0:
        return None
    else:
        player = player.iloc[0]

    df_gameweek = getPlayerStatFromAPI(player['id'])
    return player, df_gameweek #returns a tuple of a series, and a dataframe


def getPlayerFromID(player_id):
    """
    Gets a player's season and gamebygame stats from their player ID
    """
    df_season = getPlayersFromAPI()
    player = df_season[df_season['id'] == player_id]
    if len(player) == 0:
        return None
    else:
        player = player.iloc[0]

    df_gameweek = getPlayerStatFromAPI(player['id'])
    return player, df_gameweek #returns a tuple of a series, and a dataframe


def getPlayerNextGWFeatures(player_id, current_gw): #note current_gw has "not" happened yet
    points_last_5 = 0
    minutes_per_game_last_5 = 0
    goals_last_5 = 0
    assists_last_5 = 0
    clean_sheets_last_5 = 0
    bonus_points_last_5 = 0
    goals_conceeded_last_5 = 0
    influence_last_5 = 0
    creativity_last_5 = 0
    threat_last_5 = 0
    # ict_index_last_5 = 0
    yellow_cards_last_5 = 0
    red_cards_last_5 = 0
    starts_last_5 = 0
    transfers_in_last_5 = 0
    transfers_out_last_5 = 0

    saves_last_5 = 0

    player_price_diff_last_5 = -1 #holds the difference in the player's price in the last 5 games

    player_position = -1
    player_price = -1

    # xG_per_90 = -1
    # xA_per_90 = -1
    # xGoals_Conceeded_per_90 = -1

    xG_per_90_last_5 = 0
    xA_per_90_last_5 = 0
    xGoals_Conceeded_per_90_last_5 = 0

    #home/away in next 7
    home_away_current = -1 #home = 1, away = 0

    #FDR for next 7
    fdr_current = -1

    player_season_stats, player_gw_stats = getPlayerFromID(player_id)
    
    player_position = player_season_stats['element_type']
    player_price = player_season_stats['now_cost']/10
    xG_per_90 = player_season_stats['expected_goals_per_90']
    xA_per_90 = player_season_stats['expected_assists_per_90']
    xGoals_Conceeded_per_90 = player_season_stats['expected_goals_conceded_per_90']

    before_price = None
    games_played = 0
    for i in range(current_gw-1, current_gw-6, -1):
        gw_stats = player_gw_stats[player_gw_stats['round'] == i]
        if len(gw_stats) == 0:
            continue #blank gw
            # raise Exception(f"No stats found for player {player_id} in gameweek {i}")
        else:
            gw_stats = gw_stats.iloc[0]
            points_last_5 += gw_stats['total_points']
            minutes_per_game_last_5 += gw_stats['minutes']
            goals_last_5 += gw_stats['goals_scored']
            assists_last_5 += gw_stats['assists']
            clean_sheets_last_5 += gw_stats['clean_sheets']
            bonus_points_last_5 += gw_stats['bonus']
            goals_conceeded_last_5 += gw_stats['goals_conceded']
            influence_last_5 += float(gw_stats['influence'])
            creativity_last_5 += float(gw_stats['creativity'])
            threat_last_5 += float(gw_stats['threat'])
            # ict_index_last_5 += float(gw_stats['ict_index'])
            yellow_cards_last_5 += gw_stats['yellow_cards']
            red_cards_last_5 += gw_stats['red_cards']
            starts_last_5 += gw_stats['starts']
            transfers_in_last_5 += gw_stats['transfers_in']
            transfers_out_last_5 += gw_stats['transfers_out']

            xG_per_90_last_5 += float(gw_stats['expected_goals'])
            xA_per_90_last_5 += float(gw_stats['expected_assists'])
            xGoals_Conceeded_per_90_last_5 += float(gw_stats['expected_goals_conceded']) 

            saves_last_5 += gw_stats['saves']


            if gw_stats['minutes'] > 0:
                games_played += 1

        before_price = gw_stats['value']/10 #divide by 10 to get 45 to be 4.5

        if before_price is not None and i == current_gw-5:
            player_price_diff_last_5 = player_price - before_price

    if games_played > 0:
        minutes_per_game_last_5 = minutes_per_game_last_5 / games_played
    else:
        minutes_per_game_last_5 = 0

    #get home/away and FDR for next gameweek
    fixtures = getFixturesFromAPI()
    next_gw_fuxture = fixtures[(fixtures['event'] == current_gw) & ((fixtures['team_h'] == player_season_stats['team']) | (fixtures['team_a'] == player_season_stats['team']))] #get the next fixture for the player
    if len(next_gw_fuxture) == 0:
        raise Exception(f"No fixture found for player {player_id} in gameweek {current_gw}")
    else:
        next_gw_fuxture = next_gw_fuxture.iloc[0]
        if next_gw_fuxture['team_h'] == player_season_stats['team']:
            home_away_current = 1
            fdr_current = next_gw_fuxture['team_a_difficulty']
        else:
            home_away_current = 0
            fdr_current = next_gw_fuxture['team_h_difficulty']

    return {
        'points_last_5': points_last_5,
        'minutes_per_game_last_5': round(minutes_per_game_last_5, 2),
        'goals_last_5': goals_last_5,
        'assists_last_5': assists_last_5,
        'clean_sheets_last_5': clean_sheets_last_5,
        'bonus_points_last_5': bonus_points_last_5,
        'goals_conceded_last_5': goals_conceeded_last_5,
        'yellow_cards_last_5': yellow_cards_last_5,
        'red_cards_last_5': red_cards_last_5,
        'starts_last_5': starts_last_5,
        'transfers_in_last_5': transfers_in_last_5,
        'transfers_out_last_5': transfers_out_last_5,
        'saves_last_5': saves_last_5,

        'influence_last_5': influence_last_5,
        'creativity_last_5': creativity_last_5,
        'threat_last_5': threat_last_5,
        # 'ict_index_last_5': round(ict_index_last_5, 2),
        'player_price_diff_last_5': round(player_price_diff_last_5, 2),
        'player_price': player_price,


        # 'xG_per_90': round(xG_per_90, 2),
        # 'xA_per_90': round(xA_per_90, 2),
        # 'xG_conceded_per_90': round(xGoals_Conceeded_per_90, 2),
        
        'xG_per_90_last_5': round(xG_per_90_last_5 / games_played, 2) if games_played > 0 else 0,
        'xA_per_90_last_5': round(xA_per_90_last_5 / games_played, 2) if games_played > 0 else 0,
        'xG_conceded_per_90_last_5': round(xGoals_Conceeded_per_90_last_5 / games_played, 2) if games_played > 0 else 0,

        'home_away_current': home_away_current,
        'fdr_current': fdr_current,
        'position_1': 1 if player_position == 2 else 0,  # position_1 (DEF)
        'position_2': 1 if player_position == 3 else 0,  # position_2 (MID)
        'position_3': 1 if player_position == 4 else 0,  # position_3 (FWD)
    }


def predictPlayerNextGWPoints(player_id, current_gw, model_value):
    """
    Predicts the points for a player in the next gameweek using the trained model and scaler
    Args:
      player_id: the player's ID in the FPL API
      current_gw: the current gameweek (the gameweek we want to predict for, which has not happened yet)
      model_value: the value of the model we want to use (the value is the MAE of the model on the validation set, which is used in the filename of the saved model and scaler)
    Returns: 
        The predicted points for the next gameweek as a float
    """
    torch.manual_seed(42) #set the seed for reproducibility

    model = FPLModelOneGW(input_size=26) #number of inputs from getHistoricalData_OneGW.py
    model.load_state_dict(torch.load(f'points_predictors/one_gw_{str(model_value)}_best_model.pth'))
    scaler = joblib.load(f'points_predictors/one_gw_{str(model_value)}_scaler.pkl')

    data = getPlayerNextGWFeatures(player_id, current_gw)
    x = pd.DataFrame([data])

    #seperate into binary/numerical columns to scale only numerical columns
    binary_columns = [col for col in x.columns if col.startswith('position_')] + ['home_away_current']
    numerical_columns = [col for col in x.columns if col not in binary_columns]

    x[numerical_columns] = scaler.transform(x[numerical_columns])
    x_input = torch.tensor(x.values, dtype=torch.float32)

    with torch.no_grad():
        predictions_log = model(x_input)

    actual_prediction = np.expm1(predictions_log)

    return round(actual_prediction.item(), 2) #returns the predicted points for the next gameweek as a float


def getPlayerNext7GWFeatures(player_id, current_gw): #note current_gw has "not" happened yet
    points_last_5 = 0
    minutes_per_game_last_5 = 0
    goals_last_5 = 0
    assists_last_5 = 0
    clean_sheets_last_5 = 0
    bonus_points_last_5 = 0
    goals_conceeded_last_5 = 0
    influence_last_5 = 0
    ict_index_last_5 = 0
    yellow_cards_last_5 = 0
    red_cards_last_5 = 0
    starts_last_5 = 0
    transfers_in_last_5 = 0
    transfers_out_last_5 = 0

    player_price_diff_last_5 = -1 #holds the difference in the player's price in the last 5 games

    player_position = -1
    player_price = -1

    xG_per_90 = -1
    xA_per_90 = -1
    xGoals_Conceeded_per_90 = -1

    #home/away in next 7
    home_away = [] #home = 1, away = 0

    #FDR for next 7
    fdr = []

    player_season_stats, player_gw_stats = getPlayerFromID(player_id)
    
    player_position = player_season_stats['element_type']
    player_price = player_season_stats['now_cost']/10
    xG_per_90 = player_season_stats['expected_goals_per_90']
    xA_per_90 = player_season_stats['expected_assists_per_90']
    xGoals_Conceeded_per_90 = player_season_stats['expected_goals_conceded_per_90']

    before_price = None
    games_played = 0
    for i in range(current_gw-1, current_gw-6, -1):
        gw_stats = player_gw_stats[player_gw_stats['round'] == i]
        if len(gw_stats) == 0:
            continue #blank gw
            # raise Exception(f"No stats found for player {player_id} in gameweek {i}")
        else:
            gw_stats = gw_stats.iloc[0]
            points_last_5 += gw_stats['total_points']
            minutes_per_game_last_5 += gw_stats['minutes']
            goals_last_5 += gw_stats['goals_scored']
            assists_last_5 += gw_stats['assists']
            clean_sheets_last_5 += gw_stats['clean_sheets']
            bonus_points_last_5 += gw_stats['bonus']
            goals_conceeded_last_5 += gw_stats['goals_conceded']
            influence_last_5 += float(gw_stats['influence'])
            ict_index_last_5 += float(gw_stats['ict_index'])
            yellow_cards_last_5 += gw_stats['yellow_cards']
            red_cards_last_5 += gw_stats['red_cards']
            starts_last_5 += gw_stats['starts']
            transfers_in_last_5 += gw_stats['transfers_in']
            transfers_out_last_5 += gw_stats['transfers_out']

            if gw_stats['minutes'] > 0:
                games_played += 1

        before_price = gw_stats['value']/10 #divide by 10 to get 45 to be 4.5

        if before_price is not None and i == current_gw-5:
            player_price_diff_last_5 = player_price - before_price

    if games_played > 0:
        minutes_per_game_last_5 = minutes_per_game_last_5 / games_played
    else:
        minutes_per_game_last_5 = 0

    #get home/away and FDR for next gameweek
    fixtures = getFixturesFromAPI()
    for i in range(current_gw, current_gw+7):
        next_fuxture = fixtures[(fixtures['event'] == i) & ((fixtures['team_h'] == player_season_stats['team']) | (fixtures['team_a'] == player_season_stats['team']))] #get the next fixture for the player
        if len(next_fuxture) == 0:
            # print(f"No fixture found for player {player_id} in gameweek {i}")
            home_away.append(-1) #use -1 to indicate no fixture
            fdr.append(-1) #use -1 to indicate no fixture
            continue #blank gw
        else:
            next_fuxture = next_fuxture.iloc[0]
            if next_fuxture['team_h'] == player_season_stats['team']:
                home_away.append(1)
                fdr.append(next_fuxture['team_a_difficulty'])
            else:
                home_away.append(0)
                fdr.append(next_fuxture['team_h_difficulty'])

    return {
        'points_last_5': points_last_5,
        'minutes_per_game_last_5': minutes_per_game_last_5,
        'goals_last_5': goals_last_5,
        'assists_last_5': assists_last_5,
        'clean_sheets_last_5': clean_sheets_last_5,
        'bonus_points_last_5': bonus_points_last_5,
        'goals_conceded_last_5': goals_conceeded_last_5,
        'yellow_cards_last_5': yellow_cards_last_5,
        'red_cards_last_5': red_cards_last_5,
        'starts_last_5': starts_last_5,
        'transfers_in_last_5': transfers_in_last_5,
        'transfers_out_last_5': transfers_out_last_5,
        # 'influence_last_5': influence_last_5,
        'ict_index_last_5': ict_index_last_5,
        'player_price_diff_last_5': player_price_diff_last_5,
        # 'ict_index': ict_index,
        'player_price': player_price,
        'xG_per_90': xG_per_90,
        'xA_per_90': xA_per_90,
        'xG_conceded_per_90': xGoals_Conceeded_per_90,
        'home_away_current': home_away[0],
        'home_away_current_plus_1': home_away[1],
        'home_away_current_plus_2': home_away[2],
        'home_away_current_plus_3': home_away[3],
        'home_away_current_plus_4': home_away[4],
        'home_away_current_plus_5': home_away[5],
        'home_away_current_plus_6': home_away[6],
        'fdr_current': fdr[0],
        'fdr_current_plus_1': fdr[1],
        'fdr_current_plus_2': fdr[2],
        'fdr_current_plus_3': fdr[3],
        'fdr_current_plus_4': fdr[4],
        'fdr_current_plus_5': fdr[5],
        'fdr_current_plus_6': fdr[6] , 
        'position_1': 1 if player_position == 2 else 0,  # position_1 (DEF)
        'position_2': 1 if player_position == 3 else 0,  # position_2 (MID)
        'position_3': 1 if player_position == 4 else 0,  # position_3 (FWD)
    }


def predictPlayerNext7GWPoints(player_id, current_gw, model_value):
    """
    Predicts the points for a player in the next 7 gameweeks using the trained model and scaler
    Args:
      player_id: the player's ID in the FPL API
      current_gw: the current gameweek (the gameweek we want to predict for, which has not happened yet)
        model_value: the value of the model we want to use (the value is the MAE of the model on the validation set, which is used in the filename of the saved model and scaler)
    Returns: 
        The predicted points for the next 7 gameweeks as a float
    """
    torch.manual_seed(42) #set the seed for reproducibility

    model = FPLModel(input_size=35) #number of inputs from getHistoricalData_OneGW.py
    model.load_state_dict(torch.load(f'points_predictors/{str(model_value)}_best_model.pth'))
    scaler = joblib.load(f'points_predictors/{str(model_value)}_scaler.pkl')

    data = getPlayerNext7GWFeatures(player_id, current_gw)
    x = pd.DataFrame([data])

    #seperate into binary/numerical columns to scale only numerical columns
    binary_columns = [col for col in x.columns if col.startswith('position_') or col.startswith('home_away_')]
    numerical_columns = [col for col in x.columns if col not in binary_columns]

    x[numerical_columns] = scaler.transform(x[numerical_columns])
    x_input = torch.tensor(x.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_input)

    return round(predictions.item(), 2) #returns the predicted points for the next gameweek as a float


def getTopPlayersForGW(gameweek):
    players = getPlayersFromAPI()
    next_gw_results = []
    next_7_results = []
    for _, player in players.iterrows():
        player_id = player['id']
        print("checking player ", player_id)
        player_name = f"{player['first_name']} {player['second_name']}"
        next_gw_results.append({"name": player_name, "points": predictPlayerNextGWPoints(player_id, gameweek, 4.9505), 'stuff': player})
        next_7_results.append({"name": player_name, "points": predictPlayerNext7GWPoints(player_id, gameweek, 7.208), 'stuff': player})

    next_gw_results.sort(key=lambda x: x['points'], reverse=True)
    next_7_results.sort(key=lambda x: x['points'], reverse=True)

    print()
    print("=== Top Players for Next GW ===")
    for i, entry in enumerate(next_gw_results[:20], 1):
        print(f"{i}. {entry['name']}: {entry['points']} pts")

    print("\n=== Top Players for Next 7 GWs ===")
    for i, entry in enumerate(next_7_results[:20], 1):
        print(f"{i}. {entry['name']}: {entry['points']} pts")

    return next_gw_results, next_7_results


if __name__ == "__main__":

    getPlayersFromAPI()
    # print(getPlayerFromName('Bukayo', 'Saka'))
    # print(predictPlayerNextGWPoints(16, 12, 1.5594))
    # print(predictPlayerNext7GWPoints(16, 12, 7.208))

    # getTopPlayersForGW(10)