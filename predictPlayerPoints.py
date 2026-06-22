import os
import pickle
import numpy as np
import requests
import xgboost as xgb
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time

from predictExpectedGoalsConceded.getCleanSheetDataset import flatten_lags, getLast5Matches, getNextMatch
from predictGoals.getGoalsDataset import computeOpponentxGCLookup

from predictFuturePoints.total_points_predictor import FPLModel

from apiFunctions import getPlayerStatFromAPI, getFixturesFromAPI, getPlayerFromID, getPlayersFromAPI, buildOpponentXGCLookup, determineBlankDoubleGWs

SEASON = os.environ.get('SEASON', 2526)  # Default to 2526 if not set
ALPHA = float(os.environ.get('ALPHA', 0.15))  #weight for the next 7 gameweek points in the overall expected points calculation

BLANK_MULTIPLIER = 0
DOUBLE_MULTIPLIER = 1.0

P_D = { #points distribution
    "minutes": [2, 2, 2, 2],
    "goals_scored": [10, 6, 5, 4], 
    "assists": [3, 3, 3, 3],
    "clean_sheets": [4, 4, 1, 0],
}

total_FPL_players = { #total number of FPL teams in each season (roughly)
    2122: 9170000.0,
    2223: 11450000.0,
    2324: 10910000.0,
    2425: 11500000.0,
    2526: 13100000.0,
}

minutes_model = xgb.XGBClassifier()
minutes_model.load_model('predictMinutes/minutes_model.json')

goals_model = xgb.XGBRegressor()
goals_model.load_model('predictGoals/goals_model.json')

assists_model = xgb.XGBRegressor()
assists_model.load_model('predictAssists/assists_model.json')

cleansheet_model = xgb.XGBRegressor()
cleansheet_model.load_model('predictExpectedGoalsConceded/cleansheet_model.json')

#----------------------Prediction functions---------------------
def predictMinutes(player_stats, gw, season):
    """Predicts the probability that a player will play at least 60 minutes in a game"""
    features = getMinutesFeatures(player_stats, gw, season)
    features = features[minutes_model.feature_names_in_] #enforces column order

    # for _, feature in features.iterrows():
    #     print(feature)
    prob = minutes_model.predict_proba(features)[0, 1] #predict the probability of playing
    cls = int(minutes_model.predict(features)[0]) #hard 1/0 decision

    return prob, cls

def getMinutesFeatures(df, gw, season):
    """Gets the features for the minutes prediction model for a given player and gameweek."""
    position_dict = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

    started_last_match = -1 #whether the player started in the last match (1 gw behind)
    starts_last_5 = -1 #how many times the player started in the last 5 games
    player_position = -1
    days_since_last_match = -1 #how many days since the player's last match
    days_since_last_team_match = -1 #how many days since the player's team last played a match
    days_till_next_match = -1 #how many days until the player's next match (after the current gw)

    minutes_last_match = -1 #1 gw behind
    minutes_last_match_plus_1 = -1 #2 gw behind
    minutes_last_match_plus_2 = -1 #3 gw behind
    minutes_last_match_plus_3 = -1 #4 gw behind
    minutes_last_match_plus_4 = -1 #5 gw behind

    games_played_last_5 = -1 #how many games the player has played in the last 5 games


    row = df[df['round'] == gw].iloc[0] #current gw row
    idx = df.index[df['round'] == gw][0] #past id

    player_position = position_dict[row['position']]

    started_last_match = 1 if idx > 0 and df.iloc[idx-1]['starts'] == 1 else 0

    minutes_last_match = df.iloc[idx-1]['minutes'] if idx > 0 else np.nan
    minutes_last_match_plus_1 = df.iloc[idx-2]['minutes'] if idx > 1 else np.nan
    minutes_last_match_plus_2 = df.iloc[idx-3]['minutes'] if idx > 2 else np.nan
    minutes_last_match_plus_3 = df.iloc[idx-4]['minutes'] if idx > 3 else np.nan
    minutes_last_match_plus_4 = df.iloc[idx-5]['minutes'] if idx > 4 else np.nan

    #days since the team last played a match
    days_since_last_team_match = (pd.to_datetime(row['kickoff_time']) - pd.to_datetime(df.iloc[idx-1]['kickoff_time'])).days if idx > 0 else np.nan #current gw kickoff time - last gw kickoff time

    #days since the player last played a match
    gw_last_played = np.nan #holds the gw of the player's last match
    for i in range(idx -1, -1, -1): #look back until the player played in a match
        if df.iloc[i]['minutes'] > 0:
            gw_last_played = df.iloc[i]['round']
            break
    days_since_last_match = (pd.to_datetime(row['kickoff_time']) - pd.to_datetime(df[df['round'] == gw_last_played].iloc[0]['kickoff_time'])).days if not pd.isna(gw_last_played) else np.nan #current gw kickoff time - last played gw kickoff time

    games_played_last_5 = sum(1 for i in range(idx-1, idx-6, -1) if i >= 0 and df.iloc[i]['minutes'] > 0)
    starts_last_5 = sum(1 for i in range(idx-1, idx-6, -1) if i >= 0 and df.iloc[i]['starts'] == 1)

    days_till_next_match = (pd.to_datetime(df.iloc[idx+1]['kickoff_time']) - pd.to_datetime(row['kickoff_time'])).days if idx < len(df) - 1 else np.nan #next gw kickoff time - current gw kickoff time
        
    features = {
        'season': season,
        'gameweek': gw,
        
        'player_position': player_position,
        'started_last_match': started_last_match,

        'minutes_last_match': minutes_last_match,
        'minutes_last_match_plus_1': minutes_last_match_plus_1,
        'minutes_last_match_plus_2': minutes_last_match_plus_2,
        'minutes_last_match_plus_3': minutes_last_match_plus_3,
        'minutes_last_match_plus_4': minutes_last_match_plus_4,

        'games_played_last_5': games_played_last_5,
        'starts_last_5': starts_last_5,

        'days_since_last_match': days_since_last_match,
        'days_since_last_team_match': days_since_last_team_match,
        'days_till_next_match': days_till_next_match,
    }

    return pd.DataFrame([features])



def predictGoals(player_stats, gw, season, opponent_xGC):
    """Predicts the number of goals a player will score in a game"""
    features = getGoalsFeatures(player_stats, gw, season, opponent_xGC)
    features = features[goals_model.feature_names_in_] #enforces column order

    # print(goals_model.feature_names_in_)

    # for _, feature in features.iterrows():
    #     print(feature)
    expected_goals = float(goals_model.predict(features)[0]) #predict the expected goals for the current gw

    return expected_goals


def getGoalsFeatures(df, gw, season, opponent_xGC):
    """Gets the features for the goals prediction model for a given player and gameweek."""
    position_dict = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    player_position = -1

    lag_features = ['expected_goals', 'expected_assists', 'influence', 'threat', 'creativity', 'goals_scored', 'bps', 'minutes']
    lag_data = {f'{feat}_lag_{i}': [] for feat in lag_features for i in range(1, 6)} #lag features for the last 5 matches (1 gw behind, 2 gws behind, etc.)

    goals_per_90_season_to_date = -1 #the player's number of goals per 90 minutes in the season to date (up to the current gw)
    xGoals_per_90_season_to_date = -1 #the player's number of xGoals per 90 minutes in the season to date (up to the current gw)

    is_home = -1 #whether the player's team is playing at home in the current gw

    row = df[df['round'] == gw].iloc[0] #current gw row
    idx = df.index[df['round'] == gw][0] #index of the current gw row

    player_position = position_dict[row['position']]
        
    for feat in lag_features:
        for lag in range(1, 6):
            lag_data[f'{feat}_lag_{lag}'] = df.iloc[idx - lag][feat] if idx >= lag else np.nan
            

    #calculate goals per 90 season to date
    minutes_played_season_to_date = df.iloc[:idx]['minutes'].sum()

    goals_scored_season_to_date = df.iloc[:idx]['goals_scored'].sum()
    xGoals_season_to_date = df.iloc[:idx]['expected_goals'].sum()

    goals_per_90_season_to_date = (goals_scored_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan
    xGoals_per_90_season_to_date = (xGoals_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan

    is_home = 1 if row['was_home'] == True else 0
            
    features = {
        'season': season,
        'gameweek': gw,
        
        'player_position': player_position,
        
        **lag_data,

        'goals_per_90_season_to_date': goals_per_90_season_to_date,
        'xGoals_per_90_season_to_date': xGoals_per_90_season_to_date,
        
        'is_home': is_home,

        'opponent_xGC_per_game': opponent_xGC
    }

    return pd.DataFrame([features])

def predictAssists(player_stats, gw, season, opponent_xGC):
    """Predicts the number of assists a player will score in a game"""
    features = getAssistsFeatures(player_stats, gw, season, opponent_xGC)
    features = features[assists_model.feature_names_in_] #enforces column order

    # print(assists_model.feature_names_in_)

    # for _, feature in features.iterrows():
    #     print(feature)
    expected_assists = float(assists_model.predict(features)[0]) #predict the expected assists for the current gw

    return expected_assists


def getAssistsFeatures(df, gw, season, opponent_xGC):
    """Gets the features for the assists prediction model for a given player and gameweek."""
    position_dict = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    player_position = -1

    lag_features = ['expected_goals', 'expected_assists', 'influence', 'threat', 'creativity', 'assists', 'bps', 'minutes']
    lag_data = {f'{feat}_lag_{i}': [] for feat in lag_features for i in range(1, 6)} #lag features for the last 5 matches (1 gw behind, 2 gws behind, etc.)

    assists_per_90_season_to_date = -1 #the player's number of assists per 90 minutes in the season to date (up to the current gw)
    xAssists_per_90_season_to_date = -1 #the player's number of xAssists per 90 minutes in the season to date (up to the current gw)

    is_home = -1 #whether the player's team is playing at home in the current gw

    row = df[df['round'] == gw].iloc[0] #current gw row
    idx = df.index[df['round'] == gw][0] #index of the current gw row

    player_position = position_dict[row['position']]
        
    for feat in lag_features:
        for lag in range(1, 6):
            lag_data[f'{feat}_lag_{lag}'] = df.iloc[idx - lag][feat] if idx >= lag else np.nan
            

    #calculate assists per 90 season to date
    minutes_played_season_to_date = df.iloc[:idx]['minutes'].sum()

    assists_scored_season_to_date = df.iloc[:idx]['assists'].sum()
    xAssists_season_to_date = df.iloc[:idx]['expected_assists'].sum()

    assists_per_90_season_to_date = (assists_scored_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan
    xAssists_per_90_season_to_date = (xAssists_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan

    is_home = 1 if row['was_home'] == True else 0
            
    features = {
        'season': season,
        'gameweek': gw,
        
        'player_position': player_position,
        
        **lag_data,

        'assists_per_90_season_to_date': assists_per_90_season_to_date,
        'xAssists_per_90_season_to_date': xAssists_per_90_season_to_date,
        
        'is_home': is_home,

        'opponent_xGC_per_game': opponent_xGC
    }

    return pd.DataFrame([features])

def predictExpectedGoalsConceded(team_id, fixtures_df, gw, season):
    """Predicts the number of goals a team will concede in a game"""
    team_fixtures = fixtures_df[
        (fixtures_df['event'] == gw) &
        ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
    ]
    if team_fixtures.empty:
        return np.nan  #blank gameweek for this team
    fixture = team_fixtures.iloc[0]
    

    features = getExpectedGoalsConcededFeatures(team_id, fixture, fixtures_df, gw, season)
    features = features[cleansheet_model.feature_names_in_] #enforces column order

    # print(cleansheet_model.feature_names_in_)

    # for _, feature in features.iterrows():
    #     print(feature)
    expected_goals_conceded = float(cleansheet_model.predict(features)[0]) #predict the expected goals conceded for the current gw

    return expected_goals_conceded

def getExpectedGoalsConcededFeatures(team_id, fixture, fixtures_df, gw, season):
    """
    Args:
        team_id (int): The ID of the team for which to predict expected goals conceded.
        fixture (pandas.Series): A Series containing the fixture for which to predict expected goals conceded. Must contain 'team_h', 'team_a', and 'kickoff_time' columns.
        fixtures_df (pandas.DataFrame): A DataFrame containing all fixtures for the season. Must contain 'team_h', 'team_a', 'kickoff_time', 'team_h_score', and 'team_a_score' columns.
        gw (int): The gameweek for which to predict expected goals conceded.
        season (int): The season for which to predict expected goals conceded.
    Returns:
        pandas.DataFrame: A DataFrame containing the features for the expected goals conceded prediction model.
    """
    player_team = team_id
    is_home = -1
    if fixture['team_h'] == team_id:
        opponent_team = fixture['team_a']
        is_home = 1
    else:        
        opponent_team = fixture['team_h']
        is_home = 0

    player_lagged_GC = []
    player_lagged_GS = []
    player_days_rest = 0
    player_days_till_next = 0

    opp_lagged_GC = []
    opp_lagged_GS = []
    opp_days_rest = 0
    opp_days_till_next = 0

    player_past_fixtures = getLast5Matches(player_team, fixture['kickoff_time'], fixtures_df)
    if len(player_past_fixtures) > 0:
        last_match = player_past_fixtures.tail(1).iloc[0]
        player_days_rest = (fixture['kickoff_time'] - last_match['kickoff_time']).days
    else:
        player_days_rest = np.nan

    next_match = getNextMatch(player_team, fixture['kickoff_time'], fixtures_df)
    if not next_match.empty:
        player_days_till_next = (next_match.iloc[0]['kickoff_time'] - fixture['kickoff_time']).days
    else:
        player_days_till_next = np.nan


    for _, past_fix in player_past_fixtures.iterrows():
        if past_fix['team_h'] == player_team:
            player_lagged_GC.append(past_fix['team_a_score']) #the opponent's goals scored = team's goals conceded
            player_lagged_GS.append(past_fix['team_h_score'])


        else:
            player_lagged_GC.append(past_fix['team_h_score'])
            player_lagged_GS.append(past_fix['team_a_score'])


    #calculate metrics for the opponent team
    opp_past_fixtures = getLast5Matches(opponent_team, fixture['kickoff_time'], fixtures_df)
    if len(opp_past_fixtures) > 0:
        last_match = opp_past_fixtures.tail(1).iloc[0]
        opp_days_rest = (fixture['kickoff_time'] - last_match['kickoff_time']).days
    else:
        opp_days_rest = np.nan

    next_match = getNextMatch(opponent_team, fixture['kickoff_time'], fixtures_df)
    if not next_match.empty:
        opp_days_till_next = (next_match.iloc[0]['kickoff_time'] - fixture['kickoff_time']).days
    else:
        opp_days_till_next = np.nan

    for _, past_fix in opp_past_fixtures.iterrows():
        if past_fix['team_h'] == opponent_team:
            opp_lagged_GC.append(past_fix['team_a_score']) #the opponent's goals scored = team's goals conceded
            opp_lagged_GS.append(past_fix['team_h_score'])
        else:
            opp_lagged_GC.append(past_fix['team_h_score'])
            opp_lagged_GS.append(past_fix['team_a_score'])


    #reverse the lagged lists so that the most recent match is first
    player_GC_reversed = player_lagged_GC[::-1]
    player_GS_reversed = player_lagged_GS[::-1]
    opp_GC_reversed = opp_lagged_GC[::-1]
    opp_GS_reversed = opp_lagged_GS[::-1]

    features = {
        'season': season,
        'gameweek': fixture['event'],
        'team': player_team,
        'opponent_team': opponent_team,
        'is_home': is_home,
        **flatten_lags(player_GC_reversed, 'lagged_GC'),
        **flatten_lags(player_GS_reversed, 'lagged_GS'),
        'days_rest': player_days_rest,
        'days_till_next': player_days_till_next,
        **flatten_lags(opp_GC_reversed, 'opponent_lagged_GC'),
        **flatten_lags(opp_GS_reversed, 'opponent_lagged_GS'),
        'opponent_days_rest': opp_days_rest,
        'opponent_days_till_next': opp_days_till_next,
        'goals_conceded': fixture['team_a_score'],
    }

    return pd.DataFrame([features])

def getPlayerNext7GWFeatures(player_id, current_gw): #note current_gw has "not" happened yet
    """Gets the features for a player's next 7 gameweeks, including their stats from the last 5 gameweeks, their position and price, and the home/away and FDR for their next 7 fixtures."""
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

    # % selected for past 5
    selected_by = []

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
        if len(gw_stats) == 0: #TODO: This is where we'll add things for a cold start at the beginning of the gw, but also need to distingush between blank gw or beggning of the season
            selected_by.append(0)
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

            selected_by.append(float(gw_stats['selected'] / total_FPL_players[SEASON] * 100)) #convert to percentage of teams that selected the player in that gw

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
        if len(next_fuxture) == 0: #^blank gw
            home_away.append(-1) #use -1 to indicate no fixture
            fdr.append(-1) #use -1 to indicate no fixture
            continue 
        else:
            #?just do an average of the home/away/fdr and that'll be our single/double gw value
            home_away_tally = 0
            fdr_tally = 0
            num_games = len(next_fuxture)
            for _, fixture in next_fuxture.iterrows():
                if fixture['team_h'] == player_season_stats['team']:
                    home_away_tally += 1
                    fdr_tally += fixture['team_a_difficulty']
                else:
                    home_away_tally += 0
                    fdr_tally += fixture['team_h_difficulty']
            home_away.append(float(home_away_tally / num_games))
            fdr.append(float(fdr_tally / num_games))

            
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
        'fdr_current_plus_6': fdr[6],
        'selected_by': selected_by[0],
        'selected_by_minus_1': selected_by[1],
        'selected_by_minus_2': selected_by[2],
        'selected_by_minus_3': selected_by[3],
        'selected_by_minus_4': selected_by[4],
        'position_1': 1 if player_position == 2 else 0,  # position_1 (DEF)
        'position_2': 1 if player_position == 3 else 0,  # position_2 (MID)
        'position_3': 1 if player_position == 4 else 0,  # position_3 (FWD)
    }


def predictPlayerNext7GWPointsTorch(player_id, current_gw, model_value, player_stats, double_blanks):
    """
    Predicts the points for a player in the next 7 gameweeks using the trained model and scaler
    Args:
        player_id: the player's ID in the FPL API
        current_gw: the current gameweek (the gameweek we want to predict for, which has not happened yet)
        model_value: the value of the model we want to use (the value is the MAE of the model on the validation set, which is used in the filename of the saved model and scaler)
        player_stats: the stats of the player
        double_blanks: list of the number of games a player has for each gw in the next 7 games
    Returns: 
        The predicted points for the next 7 gameweeks as a float
    """
    torch.manual_seed(42) #set the seed for reproducibility

    model = FPLModel(input_size=40) #number of inputs from getHistoricalData_OneGW.py
    model.load_state_dict(torch.load(f'predictFuturePoints/{str(model_value)}_best_model.pth'))
    scaler = joblib.load(f'predictFuturePoints/{str(model_value)}_scaler.pkl')

    data = getPlayerNext7GWFeatures(player_id, current_gw)
    x = pd.DataFrame([data])

    #seperate into binary/numerical columns to scale only numerical columns
    binary_columns = [col for col in x.columns if col.startswith('position_') or col.startswith('home_away_')]
    numerical_columns = [col for col in x.columns if col not in binary_columns]

    x[numerical_columns] = scaler.transform(x[numerical_columns])
    x_input = torch.tensor(x.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_input)

    prediction = round(predictions.item(), 2)

    recent = player_stats[player_stats['round'] < current_gw].tail(10)
    playing_rate_last_10 = (recent['minutes'] > 0).mean() if len(recent) > 0 else 1

    double_blanks_multiplier = 0
    for game in double_blanks:
        if game == 0:
            double_blanks_multiplier += BLANK_MULTIPLIER #should be 0 but i dont want it to be 0 for reasons i cant explain in a comment
        elif game > 1:
            double_blanks_multiplier += (game * DOUBLE_MULTIPLIER) #small extra bonus for double gws
        else:
            double_blanks_multiplier += 1

    #calculations for final predicted next7 points
    total = prediction * playing_rate_last_10 #multiply by the playing rate of the player in the last 10 games
    total = float(total / 7) #7 gws ahead
    total *= double_blanks_multiplier #multiplier to benifit players w/ double gw and hurt those w/ blanks

    return total


def convertExpectedGoalsConcededToCleanSheetProb(expected_goals_conceded):
    """
    Converts the expected goals conceded to a clean sheet probability using a Poisson distribution.
    """
    # P(X=0) = e^(-lambda) where lambda is the expected goals conceded
    clean_sheet_prob = np.exp(-expected_goals_conceded)
    return clean_sheet_prob


def calculatePlayerExpectedStats(player_id, gameweek, season, full_player_id_list, fixtures_df, xgc_lookup):
    """Calculates the expected stats for a player in a given gameweek using the trained models
    Args:
        player_id: the player's ID in the FPL API
        gameweek: the gameweek we want to predict for (the gameweek has not happened yet)
        season: the season we want to predict for
        full_player_id_list: a dataframe containing all the players and their overall stats for the season, obtained from getPlayersFromAPI()
        fixtures_df: a dataframe containing all the fixtures for the season, obtained from getFixturesFromAPI()
        xgc_lookup: a dictionary mapping (team_id, gameweek) to xGC-per-game, obtained from buildOpponentXGCLookup()
    Returns:
        A dictionary containing the expected stats for the player in the given gameweek, including:
        - minutes_prob: the probability that the player will play at least 60 minutes in the game
        - minutes_class: the predicted class for minutes played (1 if the player is predicted to play at least 60 minutes, 0 otherwise)
        - expected_goals: the expected number of goals the player will score in the game
        - expected_assists: the expected number of assists the player will score in the game
        - expected_goals_conceded: the expected number of goals the player's team will concede in the game
        - clean_sheet_prob: the probability that the player's team will keep a clean sheet in the game
    """
    # print(full_player_id_list[full_player_id_list['id'] == player_id][['first_name', 'second_name', 'team']])
    player_stat = getPlayerStatFromAPI(player_id) #gets the player's stats for each gameweek
    time.sleep(0.02) #to avoid hitting the API rate limit

    #convert player position from id to string
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    element_type = full_player_id_list[full_player_id_list['id'] == player_id].iloc[0]['element_type']
    player_stat['position'] = pos_map[element_type]
    
    player_team_id = full_player_id_list[full_player_id_list['id'] == player_id].iloc[0]['team'] #gets the player's team id

    gw_rows = player_stat[player_stat['round'] == gameweek]
    if gw_rows.empty: #for when there is no gw
        return None, player_stat
    
    opp_xgc_tally = 0
    for _, row in gw_rows.iterrows():
        opponent_team_id = row['opponent_team']
        val = xgc_lookup.get((opponent_team_id, gameweek), np.nan)
        opp_xgc_tally += 1.0 if pd.isna(val) else val
    opp_xgc = float(opp_xgc_tally / len(gw_rows))

    minutes_prob, minutes_cls = predictMinutes(player_stat, gameweek, season)

    expected_goals = predictGoals(player_stat, gameweek, season, opp_xgc)

    expected_assists = predictAssists(player_stat, gameweek, season, opp_xgc)

    expected_goals_conceded = predictExpectedGoalsConceded(player_team_id, fixtures_df, gameweek, season)
    clean_sheet_prob = convertExpectedGoalsConcededToCleanSheetProb(expected_goals_conceded)

    # diagnoseCleanSheetSpread(fixtures_df, gameweek, season)
    # df, per_team = diagnoseAcrossGameweeks(fixtures_df, season, range(1, 39))

    return ({
        'minutes_prob': minutes_prob,
        'minutes_class': minutes_cls,
        'expected_goals': expected_goals,
        'expected_assists': expected_assists,
        'expected_goals_conceded': expected_goals_conceded,
        'clean_sheet_prob': clean_sheet_prob,
    }, player_stat)

def getExpectedPoints(stats, player_position):
    """Calculates the expected points for a player in a given gameweek using the predicted stats and the average points per stat for the player's position."""
    indexer = player_position - 1 #to get the lookup for the points value
    return P_D['minutes'][indexer]*stats['minutes_prob'] + P_D['goals_scored'][indexer]*stats['expected_goals'] + P_D['assists'][indexer]*stats['expected_assists'] + P_D['clean_sheets'][indexer]*stats['clean_sheet_prob']

def getTopPlayersForGameweek(gameweek, season):
    """Gets the top players for a given gameweek based on their expected points, as well as their predicted points for the next 7 gameweeks."""
    full_player_id_list = getPlayersFromAPI() #gets all the players and their overall stats
    fixtures_df = getFixturesFromAPI() #gets all the fixtures for the season

    # print("building xgc lookup")
    if os.path.exists(f'xgcLookup/xgc_lookup_{gameweek}.pkl'):
        xgc_lookup = loadxgc(f'xgcLookup/xgc_lookup_{gameweek}.pkl')
    else:
        xgc_lookup = buildOpponentXGCLookup(gameweek) #once per gw
        savexgc(xgc_lookup, f'xgcLookup/xgc_lookup_{gameweek}.pkl')

    player_xp = []
    player_next_7_points = []

    # print("calculating xp stats")
    for _, player in full_player_id_list.iterrows():
        double_blanks = determineBlankDoubleGWs(player['id'], gameweek, gameweek+7, fixtures_df)
        stats, player_stat = calculatePlayerExpectedStats(player['id'], gameweek, season, full_player_id_list, fixtures_df, xgc_lookup)
        
        #*next 7 points calculation
        next_7_points = round(predictPlayerNext7GWPointsTorch(player['id'], gameweek, 7.0657, player_stat, double_blanks), 4) #PyTorch model
        player_next_7_points.append([player['first_name'], player['second_name'], next_7_points, player['team'], player['element_type'], player['now_cost'], player['id'], player['total_points']])

        #*single points gw calculation
        if double_blanks[0] == 0: #^this gw is a blank
            #! the stats variable would be a dict with all values at -1
            xp = 0
            player_xp.append([player['first_name'], player['second_name'], player['team'], xp, stats, player['element_type'], player['now_cost'], player['id'], player['total_points']])
        elif stats is None:                              # fixtures say plays, but no history row — no usable stats
            xp = 0
            player_xp.append([player['first_name'], player['second_name'], player['team'], xp, stats, player['element_type'], player['now_cost'], player['id'], player['total_points']])
        elif double_blanks[0] == 1: #^normal gw
            xp = getExpectedPoints(stats, player['element_type']) + ALPHA * next_7_points #combine with expected long term points to promote consistency (a player who has a higher x7p will probably do well, so include that)
            player_xp.append([player['first_name'], player['second_name'], player['team'], round(xp, 4), stats, player['element_type'], player['now_cost'], player['id'], player['total_points']])
        else: #^multiply our xp by the amount of games we'll play
            xp = double_blanks[0] * getExpectedPoints(stats, player['element_type']) + (ALPHA * next_7_points) #combine with expected long term points to promote consistency (a player who has a higher x7p will probably do well, so include that)
            player_xp.append([player['first_name'], player['second_name'], player['team'], round(xp, 4), stats, player['element_type'], player['now_cost'], player['id'], player['total_points']])

    player_xp = sorted(player_xp, key=lambda p: p[3], reverse=True)
    player_next_7_points = sorted(player_next_7_points, key=lambda p: p[2], reverse=True)

    # print(f"-------Top Players for GW {gameweek}-------")
    # for i in range(0, 20):
    #     print(f"{i+1}. ", player_xp[i][0], player_xp[i][1], player_xp[i][3])

    # print(f"\n-------Top Players for Next 7 GWs (from GW {gameweek} to GW {gameweek+6})-------")
    # for i in range(0, 20):
    #     print(f"{i+1}. ", player_next_7_points[i][0], player_next_7_points[i][1], player_next_7_points[i][2])

    return player_xp, player_next_7_points

def savexgc(xgc_lookup, filename):
    with open(filename, 'wb') as f:         
        pickle.dump(xgc_lookup, f)

def loadxgc(filename):
    with open(filename, 'rb') as f:        
        return pickle.load(f)

def main():
    season = 2526
    gameweek = 15

    # players = getPlayersFromAPI()
    # print(players[players['second_name'].str.contains('Semenyo', case=False)][['id', 'first_name', 'second_name']])

    getPlayerNext7GWFeatures(82, 31)

    # print(getPlayerNext7GWFeatures(1, 12))

    # getTopPlayersForGameweek(gameweek, season)


if __name__ == "__main__":
    main()

#------ Clean Sheets Plots ------
import matplotlib.pyplot as plt

def diagnoseCleanSheetSpread(fixtures_df, gameweek, season):
    for gw in range(1, 39):
        preds = {}
        for team_id in range(1, 21):
            try:
                preds[team_id] = predictExpectedGoalsConceded(team_id, fixtures_df, gw, season)
            except Exception as e:
                print(f"skip team {team_id}: {e}")
        s = pd.Series(preds, name='xGC').dropna()

        print(s.sort_values())
        print(s.describe())            # min, max, mean, std
        print(f"range: {s.max() - s.min():.3f}, std: {s.std():.3f}")

        s.hist(bins=15)
        plt.xlabel('predicted expected goals conceded')
        plt.ylabel('count')
        plt.title(f'Clean sheet model output spread — GW{gw}')
        plt.show()

def diagnoseAcrossGameweeks(fixtures_df, season, gw_range):
    rows = []
    for gw in gw_range:
        for team_id in range(1, 21):
            try:
                rows.append({'gw': gw, 'team': team_id,
                             'xGC': predictExpectedGoalsConceded(team_id, fixtures_df, gw, season)})
            except Exception:
                continue
    df = pd.DataFrame(rows).dropna()

    # 1. global spread of every prediction
    print("ALL PREDICTIONS:")
    print(df['xGC'].describe())
    print(f"global std: {df['xGC'].std():.3f}  (real-world ~0.35-0.40)\n")

    # 2. per-team: does a team get a CONSISTENT level, or random noise?
    per_team = df.groupby('team')['xGC'].agg(['mean', 'std', 'min', 'max'])
    per_team = per_team.sort_values('mean')
    print("PER-TEAM (sorted by mean xGC):")
    print(per_team)
    print(f"\nspread of team means: {per_team['mean'].std():.3f}")
    print(f"avg within-team std:  {per_team['std'].mean():.3f}")

    df['xGC'].hist(bins=30)
    return df, per_team
