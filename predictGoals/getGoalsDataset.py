# https://github.com/vaastav/Fantasy-Premier-League/tree/master

import pandas as pd
import numpy as np
import urllib.parse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seasons = [2223, 2324, 2425, 2526]

def encodeName(name):
    """
    This function puts the player's name into a format that can be used in the URL
    Args: 
        name (str): The player's name
    Returns:
        str: The URL-encoded player's name
    """
    return urllib.parse.quote(name)


def getPlayerIDsForSeason(season):
    """
    This function gets all of the player IDs for a given season.
    Args:
        season (int): The season for which to get player IDs
    Returns:
        pandas.DataFrame: A DataFrame containing the player IDs
    """
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/player_idlist.csv"
    df = pd.read_csv(url)
    df = df.sort_values('id')
    df = df.reset_index(drop=True)
    return df


def getPlayerTeam(playerid, season):
    """
    This function gets the team that the player is on based on the season.
    Args:
        playerid (int): The player's ID
        season (int): The season for which to get the player's team
    Returns:
        int: The player's team ID
    """
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/players_raw.csv"
    df = pd.read_csv(url)
    return df[df['id'] == playerid].iloc[0]['team']


def getPlayerStatsForAllGameweeks(playerid, allPlayersDf, season):
    """
    This function gets all of the stats for a given player for all gameweeks in a given season.
    Args:
        playerid (int): The player's ID
        allPlayersDf (pandas.DataFrame): A DataFrame containing all players
        season (int): The season for which to get the player's stats
    Returns:
        pandas.DataFrame: A DataFrame containing the player's stats
    """
    playerRow = allPlayersDf[allPlayersDf['id'] == playerid].iloc[0]
    first_name = encodeName(playerRow['first_name'])
    second_name = encodeName(playerRow['second_name'])
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/players/{first_name}_{second_name}_{playerid}/gw.csv"
    df = pd.read_csv(url)

    positionDf = getPlayersAndTheirPosition(season)
    player_position = positionDf[(positionDf['first_name'] == playerRow['first_name']) & (positionDf['second_name'] == playerRow['second_name'])].iloc[0]['element_type']

    print(f"Player: {playerRow['first_name']} {playerRow['second_name']}, Position: {player_position}")

    df['position'] = player_position #adds the player's position to the df

    return df


def getPlayersAndTheirPosition(season):
    """
    This function gets all of the players and their positions for a given season.
    Args:
        season (int): The season for which to get players and their positions
    Returns:
        pandas.DataFrame: A DataFrame containing the players and their positions
    """
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/cleaned_players.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['goals_scored', 'assists', 'total_points', 'minutes', 'goals_conceded', 'creativity', 'influence', 'threat', 'bonus', 'bps', 'ict_index', 'clean_sheets', 'red_cards', 'yellow_cards', 'selected_by_percent', 'now_cost'])
    return df


def getFDRForPlayerInGameweek(playerTeamID, opponentTeamID, wasHome, gameweek, fixturesDf):
    """
    This function gets the Fixture Difficulty Rating (FDR) for a player in a given gameweek.
    Args:
        playerTeamID (int): The player's team ID
        opponentTeamID (int): The opponent's team ID
        wasHome (bool): True if the player's team is at home, False otherwise
        gameweek (int): The gameweek
        fixturesDf (pandas.DataFrame): A DataFrame containing the fixtures
    Returns:
        int: The FDR for the player in the given gameweek
    """
    if wasHome:
        fixture = fixturesDf[(fixturesDf['team_h'] == playerTeamID) & (fixturesDf['team_a'] == opponentTeamID) & (fixturesDf['event'] == gameweek)].iloc[0]
        return int(fixture['team_h_difficulty'])
    else:
        fixture = fixturesDf[(fixturesDf['team_a'] == playerTeamID) & (fixturesDf['team_h'] == opponentTeamID) & (fixturesDf['event'] == gameweek)].iloc[0]
        return int(fixture['team_a_difficulty'])


def getFixturesForSeason(season):
    """
    This function gets all of the fixtures for a given season.
    Args:
        season (int): The season for which to get fixtures
    Returns:
        pandas.DataFrame: A DataFrame containing the fixtures
    """
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/fixtures.csv"
    df = pd.read_csv(url)
    return df   


#this code below is from Claude. I have no clue how it works
def computeOpponentxGCLookup(all_player_data_for_season):
    """
    all_player_data_for_season: DataFrame with rows from all players' gw.csv for one season,
    containing at minimum: opponent_team, round, expected_goals
    
    Returns: dict mapping (team_id, gameweek) -> xGC per 90 by that team UP TO (not including) gameweek
    """
    # xG conceded by team T in gameweek G = sum of expected_goals where opponent_team == T, round == G
    xgc_per_gw = (
        all_player_data_for_season
        .groupby(['opponent_team', 'round'])['expected_goals']
        .sum()
        .reset_index()
        .rename(columns={'opponent_team': 'team', 'expected_goals': 'xgc'})
    )
    
    # sort and compute cumulative (excluding current row, to avoid leakage)
    xgc_per_gw = xgc_per_gw.sort_values(['team', 'round'])
    xgc_per_gw['cum_xgc_before'] = xgc_per_gw.groupby('team')['xgc'].cumsum().shift(1)
    xgc_per_gw['games_before'] = xgc_per_gw.groupby('team').cumcount()  # 0, 1, 2, ...
    
    xgc_per_gw['xgc_per_game_before'] = np.where(
        xgc_per_gw['games_before'] > 0,
        xgc_per_gw['cum_xgc_before'] / xgc_per_gw['games_before'],
        np.nan
    )
    
    # build the lookup dict
    lookup = {
        (row['team'], row['round']): row['xgc_per_game_before']
        for _, row in xgc_per_gw.iterrows()
    }
    return lookup


def cleanPlayerDataframe(df, season):
    position_dict = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    gameweek = []
    kickoff_time = []
    player_position = []

    lag_features = ['expected_goals', 'expected_assists', 'influence', 'threat', 'creativity', 'goals_scored', 'bps', 'minutes']
    lag_data = {f'{feat}_lag_{i}': [] for feat in lag_features for i in range(1, 6)} #lag features for the last 5 matches (1 gw behind, 2 gws behind, etc.)

    goals_per_90_season_to_date = [] #the player's number of goals per 90 minutes in the season to date (up to the current gw)
    xGoals_per_90_season_to_date = [] #the player's number of xGoals per 90 minutes in the season to date (up to the current gw)

    is_home = [] #whether the player's team is playing at home in the current gw

    opponent_team_id = [] #the ID of the opponent team in the current gw

    #target metric: 
    goals_scored = []

    for index, row in df.iterrows(): #for each gameweek for the player
        player_position.append(position_dict[row['position']])
        gameweek.append(row['round'])
        kickoff_time.append(row['kickoff_time'])
        
        for feat in lag_features:
            for lag in range(1, 6):
                lag_data[f'{feat}_lag_{lag}'].append(
                    df.iloc[index - lag][feat] if index >= lag else np.nan
                )

        #calculate goals per 90 season to date
        minutes_played_season_to_date = df.iloc[:index]['minutes'].sum()

        goals_scored_season_to_date = df.iloc[:index]['goals_scored'].sum()
        xGoals_season_to_date = df.iloc[:index]['expected_goals'].sum()

        goals_per_90_season_to_date.append((goals_scored_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan)
        xGoals_per_90_season_to_date.append((xGoals_season_to_date / minutes_played_season_to_date * 90) if minutes_played_season_to_date > 0 else np.nan)

        is_home.append(1 if row['was_home'] == True else 0)
        opponent_team_id.append(row['opponent_team'])

        goals_scored.append(row['goals_scored'])
            
    df_dict = {
        'season': season,
        'gameweek': gameweek,
        'kickoff_time': kickoff_time,
        
        'player_position': player_position,
        
        **lag_data,

        'goals_per_90_season_to_date': goals_per_90_season_to_date,
        'xGoals_per_90_season_to_date': xGoals_per_90_season_to_date,
        
        'is_home': is_home,
        'opponent_team_id': opponent_team_id,

        'goals_scored': goals_scored
    }

    ret_df = pd.DataFrame(df_dict)

    return ret_df
    

def getFullDataset():
    """
    This function gets the full dataset for all seasons.
    Returns:
        pandas.DataFrame: The full dataset.
    """
    all_data = []
    all_raw_data_by_season = {}
    for season in seasons:
        print("Working on season ", season)
        season_raw = []

        full_player_id_list = getPlayerIDsForSeason(season)
        players_raw = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/players_raw.csv")
                
        for _, playerRow in players_raw.iterrows():
            playerid = playerRow['id']
            try: 
                player_stat = getPlayerStatsForAllGameweeks(playerid, full_player_id_list, season)

                season_raw.append(player_stat) #keep track of the raw data for the season so that we can calculate opponent xGC per game later
                
                cleaned_player_df = cleanPlayerDataframe(player_stat, season)
                if len(cleaned_player_df) > 0:
                    all_data.append(cleaned_player_df)
            except Exception as e:
                print(f"Skipping player {playerid} season {season}: {e}")
                continue
        
        all_raw_data_by_season[season] = pd.concat(season_raw, ignore_index=True)
    
    final_df = pd.concat(all_data, ignore_index=True)

    #now add the opponent's team xGoals Conceeded per 90 minutes for the season to date as a feature
    final_df['opponent_xGC_per_game'] = np.nan
    for season, raw in all_raw_data_by_season.items():
        lookup = computeOpponentxGCLookup(raw)
        
        mask = final_df['season'] == season
        keys = list(zip(final_df.loc[mask, 'opponent_team_id'], final_df.loc[mask, 'gameweek']))
        final_df.loc[mask, 'opponent_xGC_per_game'] = [lookup.get(key, np.nan) for key in keys]


    final_df.to_csv('predictGoals/goals_training_data.csv', index=False)
    print(f"Done. {len(final_df)} rows saved")
    return final_df


def main():
    getFullDataset()


if __name__ == '__main__':
    main()