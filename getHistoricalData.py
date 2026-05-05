# https://github.com/vaastav/Fantasy-Premier-League/tree/master

import pandas as pd
import urllib.parse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seasons = [2223, 2324, 2425] #1819, 1920, 2021, 2122, 2526

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


def getFutureValue(df, index, offset, column, default):
    """
    This function gets the value of a column in a DataFrame at a future index.
    Args:
        df (pandas.DataFrame): The DataFrame to get the value from
        index (int): The current index
        offset (int): The number of rows to move forward
        column (str): The column to get the value from
        default: The default value to return if the future index is out of bounds
    Returns:
        The value at the future index, or the default value if the future index is out of bounds.
    """
    future_index = index + offset
    if future_index >= len(df):
        return default
    else:
        return df.iloc[future_index][column]


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


def cleanPlayerDataframe(df, fixturesdf, playerTeamID):
    """
    This function cleans the player's data by adding additional features.
    Args:
        df (pandas.DataFrame): The player's data
        fixturesdf (pandas.DataFrame): A DataFrame containing the fixtures
        playerTeamID (int): The player's team ID
    Returns:
        pandas.DataFrame: The cleaned player's data
    """
    position_dict = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

    points_last_5 = []
    minutes_last_5 = []
    goals_last_5 = []
    assists_last_5 = []
    clean_sheets_last_5 = []
    bonus_points_last_5 = []

    ict_index = []
    player_position = []
    player_price = []

    xG_per_90 = []
    xA_per_90 = []

    #home/away in next 7
    home_away_current_plus_1 = [] #home = 1, away = 0
    home_away_current_plus_2 = []
    home_away_current_plus_3 = []
    home_away_current_plus_4 = []
    home_away_current_plus_5 = []
    home_away_current_plus_6 = []
    home_away_current_plus_7 = []

    #FDR for next 7
    fdr_current_plus_1 = []
    fdr_current_plus_2 = []
    fdr_current_plus_3 = []
    fdr_current_plus_4 = []
    fdr_current_plus_5 = []
    fdr_current_plus_6 = []
    fdr_current_plus_7 = []

    total_points_plus_7 = [] #target metric

    for index, row in df.iterrows():
        player_position.append(position_dict[row['position']])

        if index >= 1:
            ict_index.append(df.iloc[index - 1]['ict_index'])
            player_price.append(df.iloc[index - 1]['value']/10) #divide by 10 to get 45 to be 4.5
            xG_per_90.append(df.iloc[index - 1]['expected_goals'])
            xA_per_90.append(df.iloc[index - 1]['expected_assists'])
        else:
            ict_index.append(0)
            player_price.append(0)
            xG_per_90.append(0)
            xA_per_90.append(0)

        points = 0
        minutes = 0
        goals = 0
        assists = 0
        clean_sheets = 0
        bonus = 0
        for i in range(index - 1, index - 6, -1):
            if i < 0: #no negative rows
                break
            
            past_row = df.iloc[i]
            points += past_row['total_points']
            minutes += past_row['minutes']
            goals += past_row['goals_scored']
            assists += past_row['assists']
            clean_sheets += past_row['clean_sheets']
            bonus += past_row['bonus']

        points_last_5.append(points)
        minutes_last_5.append(minutes)
        goals_last_5.append(goals)
        assists_last_5.append(assists)
        clean_sheets_last_5.append(clean_sheets)
        bonus_points_last_5.append(bonus)


        home_away_current_plus_1.append(int(getFutureValue(df, index, 1, 'was_home', -1)))
        home_away_current_plus_2.append(int(getFutureValue(df, index, 2, 'was_home', -1)))
        home_away_current_plus_3.append(int(getFutureValue(df, index, 3, 'was_home', -1)))
        home_away_current_plus_4.append(int(getFutureValue(df, index, 4, 'was_home', -1)))
        home_away_current_plus_5.append(int(getFutureValue(df, index, 5, 'was_home', -1)))
        home_away_current_plus_6.append(int(getFutureValue(df, index, 6, 'was_home', -1)))
        home_away_current_plus_7.append(int(getFutureValue(df, index, 7, 'was_home', -1)))

        for offset in range(1, 8):
            future_opponent = getFutureValue(df, index, offset, 'opponent_team', -1)
            future_was_home = getFutureValue(df, index, offset, 'was_home', -1)
            
            if future_opponent == -1 or future_was_home == -1:
                fdr = -1  # blank gameweek or end of season
            else:
                fdr = getFDRForPlayerInGameweek(playerTeamID, future_opponent, future_was_home, getFutureValue(df, index, offset, 'round', -1), fixturesdf)
            
            [fdr_current_plus_1, fdr_current_plus_2, fdr_current_plus_3, 
            fdr_current_plus_4, fdr_current_plus_5, fdr_current_plus_6, 
            fdr_current_plus_7][offset - 1].append(fdr)

        total_points = 0
        for i in range(index + 1, index + 8):
            if i >= len(df): #no negative rows
                break
            ahead_row = df.iloc[i]
            total_points += ahead_row['total_points']

        total_points_plus_7.append(total_points)
            
    df_dict = {
        'points_last_5': points_last_5,
        'minutes_last_5': minutes_last_5,
        'goals_last_5': goals_last_5,
        'assists_last_5': assists_last_5,
        'clean_sheets_last_5': clean_sheets_last_5,
        'bonus_points_last_5': bonus_points_last_5,
        'ict_index': ict_index,
        'player_position': player_position,
        'player_price': player_price,
        'xG_per_90': xG_per_90,
        'xA_per_90': xA_per_90,
        'home_away_current_plus_1': home_away_current_plus_1,
        'home_away_current_plus_2': home_away_current_plus_2,
        'home_away_current_plus_3': home_away_current_plus_3,
        'home_away_current_plus_4': home_away_current_plus_4,
        'home_away_current_plus_5': home_away_current_plus_5,
        'home_away_current_plus_6': home_away_current_plus_6,
        'home_away_current_plus_7': home_away_current_plus_7,
        'fdr_current_plus_1': fdr_current_plus_1,
        'fdr_current_plus_2': fdr_current_plus_2,
        'fdr_current_plus_3': fdr_current_plus_3,
        'fdr_current_plus_4': fdr_current_plus_4,
        'fdr_current_plus_5': fdr_current_plus_5,
        'fdr_current_plus_6': fdr_current_plus_6,
        'fdr_current_plus_7': fdr_current_plus_7 , 
        'total_points_plus_7': total_points_plus_7 
    }

    ret_df = pd.DataFrame(df_dict)

    ret_df = ret_df.iloc[5:] #drop the first 5 rows since points_last, etc are not available and there would be undercounting
    
    #drop rows where any FDR is -1 (blank gameweeks or end of season)
    fdr_cols = ['fdr_current_plus_1', 'fdr_current_plus_2', 'fdr_current_plus_3',
                'fdr_current_plus_4', 'fdr_current_plus_5', 'fdr_current_plus_6', 'fdr_current_plus_7']
    ret_df = ret_df[~(ret_df[fdr_cols] == -1).any(axis=1)]

    ret_df = ret_df.reset_index(drop=True)

    return ret_df
    

def getFullDataset():
    """
    This function gets the full dataset for all seasons.
    Returns:
        pandas.DataFrame: The full dataset.
    """
    all_data = []
    for season in seasons:
        print("Working on season ", season)
        full_player_id_list = getPlayerIDsForSeason(season)
        fixtures = getFixturesForSeason(season)
        players_raw = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/players_raw.csv")
        
        #only keep players who played 450+ minutes (roughly 5 full games)
        active_players = players_raw[players_raw['minutes'] >= 450]

        print("Looking at", len(active_players), "players")

        for _, playerRow in active_players.iterrows():
            playerid = playerRow['id']
            try:
                player_team_id = players_raw[players_raw['id'] == playerid].iloc[0]['team']
                player_stat = getPlayerStatsForAllGameweeks(playerid, full_player_id_list, season)
                cleaned_player_df = cleanPlayerDataframe(player_stat, fixtures, player_team_id)
                if len(cleaned_player_df) > 0:
                    all_data.append(cleaned_player_df)
            except Exception as e:
                print(f"Skipping player {playerid} season {season}: {e}")
                continue
    
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv('fpl_training_data.csv', index=False)
    print(f"Done. {len(final_df)} rows saved")
    return final_df


def main():
    # getFullDataset()
    pass


main()