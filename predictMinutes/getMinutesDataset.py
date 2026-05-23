# https://github.com/vaastav/Fantasy-Premier-League/tree/master

import pandas as pd
import numpy as np
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


def cleanPlayerDataframe(df, season):
    position_dict = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    gameweek = []
    kickoff_time = []

    started_last_match = [] #whether the player started in the last match (1 gw behind)
    starts_last_5 = [] #how many times the player started in the last 5 games
    player_position = []
    days_since_last_match = [] #how many days since the player's last match
    days_since_last_team_match = [] #how many days since the player's team last played a match
    days_till_next_match = [] #how many days until the player's next match (after the current gw)

    minutes_last_match = [] #1 gw behind
    minutes_last_match_plus_1 = [] #2 gw behind
    minutes_last_match_plus_2 = [] #3 gw behind
    minutes_last_match_plus_3 = [] #4 gw behind
    minutes_last_match_plus_4 = [] #5 gw behind

    games_played_last_5 = [] #how many games the player has played in the last 5 games

    #target metric: 
    # 0 = played < 60 mins
    # 1 = played 60+ mins
    did_player_play = []

    for index, row in df.iterrows(): #for each gameweek for the player
        player_position.append(position_dict[row['position']])
        gameweek.append(row['round'])
        kickoff_time.append(row['kickoff_time'])

        started_last_match.append(1 if index > 0 and df.iloc[index-1]['starts'] == 1 else 0)

        minutes_last_match.append(df.iloc[index-1]['minutes'] if index > 0 else np.nan)
        minutes_last_match_plus_1.append(df.iloc[index-2]['minutes'] if index > 1 else np.nan)
        minutes_last_match_plus_2.append(df.iloc[index-3]['minutes'] if index > 2 else np.nan)
        minutes_last_match_plus_3.append(df.iloc[index-4]['minutes'] if index > 3 else np.nan)
        minutes_last_match_plus_4.append(df.iloc[index-5]['minutes'] if index > 4 else np.nan)

        #days since the team last played a match
        days_since_last_team_match.append((pd.to_datetime(row['kickoff_time']) - pd.to_datetime(df.iloc[index-1]['kickoff_time'])).days if index > 0 else np.nan) #current gw kickoff time - last gw kickoff time

        #days since the player last played a match
        gw_last_played = np.nan #holds the gw of the player's last match
        for i in range(index -1, -1, -1): #look back until the player played in a match
            if df.iloc[i]['minutes'] > 0:
                gw_last_played = df.iloc[i]['round']
                break
        days_since_last_match.append((pd.to_datetime(row['kickoff_time']) - pd.to_datetime(df[df['round'] == gw_last_played].iloc[0]['kickoff_time'])).days if not pd.isna(gw_last_played) else np.nan) #current gw kickoff time - last played gw kickoff time

        games_played_last_5.append(sum(1 for i in range(index-1, index-6, -1) if i >= 0 and df.iloc[i]['minutes'] > 0))
        starts_last_5.append(sum(1 for i in range(index-1, index-6, -1) if i >= 0 and df.iloc[i]['starts'] == 1))

        days_till_next_match.append((pd.to_datetime(df.iloc[index+1]['kickoff_time']) - pd.to_datetime(row['kickoff_time'])).days if index < len(df) - 1 else np.nan) #next gw kickoff time - current gw kickoff time

        #taregt matric
        if row['minutes'] < 60:
            did_player_play.append(0)
        else:
            did_player_play.append(1)
            


    df_dict = {
        'season': season,
        'gameweek': gameweek,
        'kickoff_time': kickoff_time,
        
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

        'did_player_play': did_player_play
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
    for season in seasons:
        print("Working on season ", season)
        full_player_id_list = getPlayerIDsForSeason(season)
        players_raw = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/players_raw.csv")
        
        for _, playerRow in players_raw.iterrows():
            playerid = playerRow['id']
            try:
                player_stat = getPlayerStatsForAllGameweeks(playerid, full_player_id_list, season)
                cleaned_player_df = cleanPlayerDataframe(player_stat, season)
                if len(cleaned_player_df) > 0:
                    all_data.append(cleaned_player_df)
            except Exception as e:
                print(f"Skipping player {playerid} season {season}: {e}")
                continue
    
    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_csv('predict2Minutes/minutes_training_data.csv', index=False)
    print(f"Done. {len(final_df)} rows saved")
    return final_df


def main():
    getFullDataset()

if __name__ == '__main__':
    main()