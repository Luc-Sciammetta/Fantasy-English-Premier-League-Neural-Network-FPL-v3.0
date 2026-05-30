import numpy as np
import pandas as pd
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


def getLast5Matches(team_id, current_kickoff, fixtures_df):
    """
    Returns the team's last 5 matches before current_kickoff, sorted oldest to newest.
    """
    current_kickoff = pd.to_datetime(current_kickoff)
    fixtures_df = fixtures_df.copy()
    fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
    
    team_matches = fixtures_df[
        ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
        (fixtures_df['kickoff_time'] < current_kickoff) &
        (fixtures_df['finished'] == True)
    ]
    
    return team_matches.sort_values('kickoff_time').tail(5)

def getNextMatch(team_id, current_kickoff, fixtures_df):
    """
    Returns the team's next match after current_kickoff.
    """
    current_kickoff = pd.to_datetime(current_kickoff)
    fixtures_df = fixtures_df.copy()
    fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
    
    team_matches = fixtures_df[
        ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
        (fixtures_df['kickoff_time'] > current_kickoff)
    ]
    
    return team_matches.sort_values('kickoff_time').head(1)

def flatten_lags(values, name, n=5):
    """Returns dict like {'name_1': v1, 'name_2': v2, ...} padded with NaN."""
    return {
        f'{name}_{i+1}': values[i] if i < len(values) else np.nan
        for i in range(n)
    }

#Features: is_home, lagged team performance (goals conceeded, goals scored, ), opponent's lagged stats, days rest, days till next, 
def populate_team_gw_data(team_gw_data, gameweek_fixtures, fixtures_df, season):
    for _, fixture in gameweek_fixtures.iterrows():
        if pd.isna(fixture['team_h_score']) or pd.isna(fixture['team_a_score']):
            continue  # skip unplayed fixtures
        home_team = fixture['team_h']
        away_team = fixture['team_a']

        home_lagged_GC = []
        home_lagged_GS = []
        home_lagged_xGC = [] #TODO: not implemented
        home_lagged_xGS = [] #TODO: not implemented
        home_days_rest = 0
        home_days_till_next = 0

        away_lagged_GC = []
        away_lagged_GS = []
        away_lagged_xGC = [] #TODO: not implemented
        away_lagged_xGS = [] #TODO: not implemented
        away_days_rest = 0
        away_days_till_next = 0

        home_past_fixtures = getLast5Matches(home_team, fixture['kickoff_time'], fixtures_df)
        if len(home_past_fixtures) > 0:
            last_match = home_past_fixtures.tail(1).iloc[0]
            home_days_rest = (fixture['kickoff_time'] - last_match['kickoff_time']).days
        else:
            home_days_rest = np.nan

        next_match = getNextMatch(home_team, fixture['kickoff_time'], fixtures_df)
        if not next_match.empty:
            home_days_till_next = (next_match.iloc[0]['kickoff_time'] - fixture['kickoff_time']).days
        else:
            home_days_till_next = np.nan


        for _, past_fix in home_past_fixtures.iterrows():
            if past_fix['team_h'] == home_team:
                home_lagged_GC.append(past_fix['team_a_score']) #the opponent's goals scored = team's goals conceded
                home_lagged_GS.append(past_fix['team_h_score'])


            else:
                home_lagged_GC.append(past_fix['team_h_score'])
                home_lagged_GS.append(past_fix['team_a_score'])
            

        #calculate metrics for the away team
        away_past_fixtures = getLast5Matches(away_team, fixture['kickoff_time'], fixtures_df)
        if len(away_past_fixtures) > 0:
            last_match = away_past_fixtures.tail(1).iloc[0]
            away_days_rest = (fixture['kickoff_time'] - last_match['kickoff_time']).days
        else:
            away_days_rest = np.nan

        next_match = getNextMatch(away_team, fixture['kickoff_time'], fixtures_df)
        if not next_match.empty:
            away_days_till_next = (next_match.iloc[0]['kickoff_time'] - fixture['kickoff_time']).days
        else:
            away_days_till_next = np.nan

        for _, past_fix in away_past_fixtures.iterrows():
            if past_fix['team_h'] == away_team:
                away_lagged_GC.append(past_fix['team_a_score']) #the opponent's goals scored = team's goals conceded
                away_lagged_GS.append(past_fix['team_h_score'])
            else:
                away_lagged_GC.append(past_fix['team_h_score'])
                away_lagged_GS.append(past_fix['team_a_score'])


        #reverse the lagged lists so that the most recent match is first
        home_GC_reversed = home_lagged_GC[::-1]
        home_GS_reversed = home_lagged_GS[::-1]
        away_GC_reversed = away_lagged_GC[::-1]
        away_GS_reversed = away_lagged_GS[::-1]

        home_gw_data = {
            'season': season,
            'gameweek': fixture['event'],
            'team': home_team,
            'opponent_team': away_team,
            'is_home': 1,
            **flatten_lags(home_GC_reversed, 'lagged_GC'),
            **flatten_lags(home_GS_reversed, 'lagged_GS'),
            'days_rest': home_days_rest,
            'days_till_next': home_days_till_next,
            **flatten_lags(away_GC_reversed, 'opponent_lagged_GC'),
            **flatten_lags(away_GS_reversed, 'opponent_lagged_GS'),
            'opponent_days_rest': away_days_rest,
            'opponent_days_till_next': away_days_till_next,
            'goals_conceded': fixture['team_a_score'],
        }

        away_gw_data = {
            'season': season,
            'gameweek': fixture['event'],
            'team': away_team,
            'opponent_team': home_team,
            'is_home': 0,
            **flatten_lags(away_GC_reversed, 'lagged_GC'),
            **flatten_lags(away_GS_reversed, 'lagged_GS'),
            'days_rest': away_days_rest,
            'days_till_next': away_days_till_next,
            **flatten_lags(home_GC_reversed, 'opponent_lagged_GC'),
            **flatten_lags(home_GS_reversed, 'opponent_lagged_GS'),
            'opponent_days_rest': home_days_rest,
            'opponent_days_till_next': home_days_till_next,
            'goals_conceded': fixture['team_h_score'],
        }

        team_gw_data.append(home_gw_data)
        team_gw_data.append(away_gw_data)

def getDataset():
    all_data = [] #list of dataframes

    #get all players for a season
    #loop through those players, keeping track of their team. 
    #every time we see a player for a team, we increase the team's metrics

    for season in seasons:
        print("Working on season ", season)
        fixtures_df = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20{int(round(season, -2)/100)}-{season % 100}/fixtures.csv")
        fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])  #cleanup kickoff_time column for easier calculations later
        for gameweek in range(1, 39):
            print(f"Working on gameweek {gameweek}")
            gameweek_fixtures = fixtures_df[fixtures_df['event'] == gameweek]
            populate_team_gw_data(all_data, gameweek_fixtures, fixtures_df, season)

    final_df = pd.DataFrame(all_data)
    final_df.to_csv('predictCleanSheets/cleansheet_training_data.csv', index=False)
    print(f"Done. {len(final_df)} rows saved")
    return final_df
            
            
if __name__ == "__main__":
    df = getDataset()