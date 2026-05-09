#compiles a dataset of basically the same features as the getHistoricalData.py file, but just for one gw in advance

import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from get_historical_data.getHistoricalData import encodeName, getPlayerIDsForSeason, getPlayerTeam, getFixturesForSeason, getPlayerStatsForAllGameweeks, getPlayersAndTheirPosition, getFutureValue, getFDRForPlayerInGameweek
from get_historical_data.getHistoricalData import seasons

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
    minutes_per_game_last_5 = []
    goals_last_5 = []
    assists_last_5 = []
    clean_sheets_last_5 = []
    bonus_points_last_5 = []
    goals_conceeded_last_5 = []
    influence_last_5 = []
    ict_index_last_5 = []
    player_price_diff_last_5 = [] #holds the difference in the player's price in the last 5 games
    yellow_cards_last_5 = []
    red_cards_last_5 = []
    starts_last_5 = []
    transfers_in_last_5 = []
    transfers_out_last_5 = []

    # ict_index = []
    player_position = []
    player_price = []

    xG_per_90 = []
    xA_per_90 = []
    xGoals_Conceeded_per_90 = []

    #home/away in next 7
    home_away_current = [] #home = 1, away = 0

    #FDR for next 7
    fdr_current = []

    actual_gw_points = [] #target metric

    for index, row in df.iterrows():
        player_position.append(position_dict[row['position']])

        if index >= 1:
            # ict_index.append(df.iloc[index - 1]['ict_index'])
            player_price.append(df.iloc[index - 1]['value']/10) #divide by 10 to get 45 to be 4.5
            xG_per_90.append(round(df.iloc[index - 1]['expected_goals'], 2))
            xA_per_90.append(round(df.iloc[index - 1]['expected_assists'], 2))
            xGoals_Conceeded_per_90.append(round(df.iloc[index - 1]['expected_goals_conceded'], 2))
        else:
            # ict_index.append(0)
            player_price.append(0)
            xG_per_90.append(0)
            xA_per_90.append(0)
            xGoals_Conceeded_per_90.append(0)

        points = 0
        minutes = 0
        goals = 0
        assists = 0
        clean_sheets = 0
        bonus = 0
        goals_conceeded = 0
        influence = 0
        ict_index = 0
        yellow_cards = 0
        red_cards = 0
        starts = 0
        transfers_in = 0
        transfers_out = 0

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
            goals_conceeded += past_row['goals_conceded']
            influence += past_row['influence']
            ict_index += round(past_row['ict_index'], 2)
            yellow_cards += past_row['yellow_cards']
            red_cards += past_row['red_cards']
            starts += past_row['starts']
            transfers_in += past_row['transfers_in']
            transfers_out += past_row['transfers_out']

        games_played = sum(1 for i in range(index-1, index-6, -1) if i >= 0 and df.iloc[i]['minutes'] > 0)

        points_last_5.append(points)
        minutes_per_game_last_5.append(round(minutes / games_played, 2) if games_played > 0 else 0)
        goals_last_5.append(goals)
        assists_last_5.append(assists)
        clean_sheets_last_5.append(clean_sheets)
        bonus_points_last_5.append(bonus)
        goals_conceeded_last_5.append(goals_conceeded)
        influence_last_5.append(influence)
        ict_index_last_5.append(ict_index)
        yellow_cards_last_5.append(yellow_cards)
        red_cards_last_5.append(red_cards)
        starts_last_5.append(starts)
        transfers_in_last_5.append(transfers_in)
        transfers_out_last_5.append(transfers_out)

        if index - 6 >= 0:
            #players with a +price_diff means their value increased, players with a -price_diff, means their value decreased
            player_price_diff = df.iloc[index-1]['value']/10 - df.iloc[index-6]['value']/10
            player_price_diff_last_5.append(round(player_price_diff, 2))
        else:
            player_price_diff_last_5.append(0)

        home_away_current.append(int(getFutureValue(df, index, 0, 'was_home', -1)))
        

        future_opponent = getFutureValue(df, index, 0, 'opponent_team', -1)
        future_was_home = getFutureValue(df, index, 0, 'was_home', -1)
            
        if future_opponent == -1 or future_was_home == -1:
            fdr = -1  # blank gameweek or end of season
        else:
            fdr = getFDRForPlayerInGameweek(playerTeamID, future_opponent, future_was_home, getFutureValue(df, index, 0, 'round', -1), fixturesdf)

        fdr_current.append(fdr)   

        actual_gw_points.append(getFutureValue(df, index, 0, 'total_points', 0))
            
    df_dict = {
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
        'player_position': player_position,
        'player_price': player_price,
        'xG_per_90': xG_per_90,
        'xA_per_90': xA_per_90,
        'xG_conceded_per_90': xGoals_Conceeded_per_90,
        'home_away_current': home_away_current,
        'fdr_current': fdr_current,
        'actual_gw_points': actual_gw_points
    }

    ret_df = pd.DataFrame(df_dict)

    ret_df = ret_df.iloc[5:] #drop the first 5 rows since points_last, etc are not available and there would be undercounting
    
    #drop rows where any FDR is -1 (blank gameweeks or end of season)
    fdr_cols = ['fdr_current']
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

    #split player position into separate columns
    final_df['player_position'] = final_df['player_position'].astype(str)
    final_df = pd.get_dummies(final_df, columns=['player_position'], prefix='position', drop_first=True)
    dummy_cols = [col for col in final_df.columns if col.startswith('position_')] #now convert the true/false to ints
    final_df[dummy_cols] = final_df[dummy_cols].astype(int)

    final_df.to_csv('oneGW_fpl_training_data.csv', index=False)
    print(f"Done. {len(final_df)} rows saved")
    return final_df


def main():
    getFullDataset()


if __name__ == '__main__':
    for i in range(0, 10):
        main()