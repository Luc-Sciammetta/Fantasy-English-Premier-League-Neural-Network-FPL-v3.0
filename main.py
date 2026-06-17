import os

import requests
import pandas as pd
import pprint
import time
import matplotlib.pyplot as plt

from optimize_team import optimizeFullTeam, optimizeTeamFormation, determine_transfers
from predictPlayerPoints import getTopPlayersForGameweek

SEASON = os.environ.get('SEASON', 2526)  # Default to 2526 if not set
TRANSFER_POINTS_THRESHOLD = 30 #if the transfers that we make improve the team by x points in the next 7 gameweeks

def plot_list(lst, title, xlabel, ylabel):
    plt.plot(lst)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def getCurrentGameweek():
    """Gets the current gw from the FPL API. Returns None if no current gameweek could be found."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url).json()

    events = pd.DataFrame(response['events'])

    for _, event in events.iterrows():
        if event['is_current'] == True:
            return event['id'] #the gameweek it currently is
        
    return None

def getBudget():
    """Gets the current budget from the budget.txt file."""
    budget_file = open(r"teamInfo/budget.txt", "r")
    budgets = budget_file.readlines()
    return int(budgets[-1]) #the last value in the file is the current budget

def getTeam(players_next_gw, player_next_7_points):
    """Gets the current team from the team.txt file."""
    team_file = open(r"teamInfo/team.txt", "r")
    teams = team_file.readlines()
    team = populateTeam(teams[-1].split(",")[:-1], players_next_gw, player_next_7_points)
    return team

def populateTeam(player_ids, players_next_gw, player_next_7_points):
    """Populates the current team with player info from the FPL API.
    Args: 
        player_ids: list of player ids in the team
        players_next_gw: list of tuples (first_name, second_name, predicted_points_next_gw) for all players
        player_next_7_points: list of tuples (first_name, second_name, predicted_points_next_7_gw) for all players"""
    elements = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()['elements']
    by_id = {e['id']: e for e in elements}

    team = []
    for player in player_ids:
        player_info = by_id[int(player)]      # ids come back as strings from split(",")

        next_gw_points = next(
            (p[3] for p in players_next_gw
             if p[0] == player_info['first_name'] and p[1] == player_info['second_name']),
            0.0
        )
        next_7_points = next(
            (p[2] for p in player_next_7_points
             if p[0] == player_info['first_name'] and p[1] == player_info['second_name']),
            0.0
        )

        team.append({
            "first_name": player_info['first_name'],
            "second_name": player_info['second_name'],
            "team": player_info['team'],
            "element_type": player_info['element_type'],
            "points": next_7_points,
            "points_next_gw": next_gw_points,
            "cost": player_info['now_cost'],
            "id": player_info['id'],
        })
    return team

def getTransfers():
    """Gets the current number of transfers from the transfers.txt file."""
    transfers_file = open(r"teamInfo/transfers.txt", "r")
    transfers = transfers_file.readlines()
    return (int(transfers[-1]) + 1) if int(transfers[-1]) < 5 else 5 

def saveBudget(val):
    """Saves the current budget to the budget.txt file."""
    budget_file = open(r"teamInfo/budget.txt", "a")
    budget_file.write(str(val) + "\n")
    budget_file.close()

def saveTransfers(val):
    """Saves the current number of transfers to the transfers.txt file."""
    transfers_file = open(r"teamInfo/transfers.txt", "a")
    transfers_file.write(str(val) + "\n")
    transfers_file.close()

def saveTeam(team):
    """Saves the current team to the team.txt file."""
    team_file = open(r"teamInfo/team.txt", "a")
    for player in team:
        team_file.write(str(player['id']) + ",")
    team_file.write("\n")
    team_file.close()

def getTeamExpected7GWPoints(team):
    total_points = 0
    for player in team:
        total_points += player['points']
    return total_points

def determineWhoChanged(old_team, new_team):
    """Determines which players were transferred in and out by comparing the old team and the new team.
    Args:
        old_team: list of player dicts in the old team
        new_team: list of player dicts in the new team
    Returns:        
        out_t: list of player dicts that were transferred out
        in_t: list of player dicts that were transferred in
    """
    in_t = []
    out_t = []
    for player in old_team:
        in_team = False
        for np in new_team:
            if (player['first_name'], player['second_name']) == (np['first_name'], np['second_name']): #match
                in_team = True
        if not in_team:
            out_t.append(player)

    for player in new_team:
        in_team = False
        for np in old_team:
            if (player['first_name'], player['second_name']) == (np['first_name'], np['second_name']): #match
                in_team = True
        if not in_team:
            in_t.append(player)

    return out_t, in_t


def run(gw):
    """Main loop"""
    # print("------------ Welcome to the Fantasy EPL AI ------------\n")
    gameweek = getCurrentGameweek()
    gameweek = gw 
    # print("Initializing. Doing pre-stuff...")
    players_next_gw, player_next_7_points = getTopPlayersForGameweek(gameweek, SEASON)
    # print("\nDone. Now Optimizing Team.")

    print("-------- Current Gameweek:", gameweek, "--------")

    if gameweek == 1:
        print("Making a new team...")
        team, budget = optimizeFullTeam(players_next_gw, player_next_7_points)
        transfers = 0
    else:
        budget = getBudget()
        team = getTeam(players_next_gw, player_next_7_points)
        transfers = getTransfers()

        print("Current Budget:", budget)
        print("Current Transfers Amount:", transfers)
        # print("\nCurrent Team:")
        # for x in team:
        #     print(x['first_name'], x['second_name'])

        team, budget, transfers, old_team = determine_transfers(team, budget, transfers, players_next_gw, player_next_7_points) #the new team, the remaining budget, the remaining transfers
        old_team_value = getTeamExpected7GWPoints(old_team)
        new_team_value = getTeamExpected7GWPoints(team)
        if new_team_value - old_team_value > TRANSFER_POINTS_THRESHOLD: #if the transfers that we make improve the team by x points in the next 7 gameweeks
            #make the changes
            print("\nTransfers will improve the team by:", new_team_value - old_team_value, "points")
            print("Old Team Expected 7GW Points:", old_team_value)
            print("New Team Expected 7GW Points:", new_team_value)
            out_t, in_t = determineWhoChanged(old_team, team)
            
            print("\nPlayers Transfered Out:")
            for x in out_t:
                print(x['first_name'], x['second_name'])
            print("\nPlayers Transfered In:")
            for x in in_t:
                print(x['first_name'], x['second_name'])

        else: #revert the team back and bank the free transfer for next week
            print("\nNot worth making transfers, banking them for next week.")
            print("Old Team Expected 7GW Points:", old_team_value)
            print("New Team Expected 7GW Points:", new_team_value)
            team = old_team
            transfers = getTransfers() #we want the amount of transfers we have available this week

    starters, bench, value = optimizeTeamFormation(team)
    capitain = starters[0]
    vice_captain = starters[1]
    print("\nCaptain:", capitain['first_name'], capitain['second_name'])
    print("Vice Captain:", vice_captain['first_name'], vice_captain['second_name'])

    # print(f"\nOptimized Team for Gameweek {gameweek}:")
    # print("Starters:")
    # for starter in starters:
    #     print(starter["first_name"], starter["second_name"], starter["team"], "Expected Points:", starter["points_next_gw"])

    # print("\nBench:")
    # for benched in bench:
    #     print(benched["first_name"], benched["second_name"], benched["team"], "Expected Points:", benched["points_next_gw"])
    
    saveTeam(team)
    saveBudget(budget)
    saveTransfers(transfers)

    # print("Saved Team!")
    # print("\nNew Budget:", budget)
    # print("New Transfers Amount:", transfers)
    # print("Bye!")

    actual_points = calculateActualPoints(starters, bench, gameweek, SEASON)
    return actual_points

def calculateActualPoints(starters, bench, gameweek, season=SEASON):
    """Sums the actual points the starting XI scored in `gameweek`, applying
    captain (top predicted) and vice-captain (2nd predicted) logic."""
    live = requests.get(
        f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    ).json()
    points_by_id  = {e['id']: e['stats']['total_points'] for e in live['elements']}
    minutes_by_id = {e['id']: e['stats']['minutes']      for e in live['elements']}

    # captain = highest predicted in the XI, vice = second highest
    ranked = sorted(starters, key=lambda p: p['points_next_gw'], reverse=True)
    captain = ranked[0]
    vice = ranked[1]

    total = 0
    print(f"\nSTARTERS breakdown (predicted -> actual -> next_7):")
    for p in starters:
        actual = points_by_id.get(p['id'], 0)
        total += actual
        print(f"  {p['first_name']} {p['second_name']:<25} "
              f"pred {p['points_next_gw']:>5.2f} -> actual {actual} -> next_7 {p['points']:.2f}")
        
    print(f"\nBENCH breakdown (predicted -> actual -> next_7):")
    for p in bench:
        actual = points_by_id.get(p['id'], 0)
        print(f"  {p['first_name']} {p['second_name']:<25} "
              f"pred {p['points_next_gw']:>5.2f} -> actual {actual} -> next_7 {p['points']:.2f}")

    # captain doubles; if captain played 0 minutes, the armband falls to the VC
    if minutes_by_id.get(captain['id'], 0) > 0:
        cap_used = captain
    else:
        cap_used = vice
    total += points_by_id.get(cap_used['id'], 0)

    print(f"Captain: {captain['first_name']} {captain['second_name']}, "
          f"VC: {vice['first_name']} {vice['second_name']} "
          f"(double applied to {cap_used['first_name']} {cap_used['second_name']})")
    print(f"Actual points for GW{gameweek}: {total}")
    return total


if __name__ == "__main__":
    start = time.time()
    season_total = 0
    points_each_gw = []
    for i in range(1, 39): #simulate the season
        start_run = time.time()
        gw_points = run(i)
        season_total += gw_points
        points_each_gw.append(gw_points)
        print(f"\nTotal points after GW{i}: {season_total}")
        print(f"Execution time for GW{i}: {float((time.time() - start_run)/60)} minutes")
    end = time.time()
    print(f"\nExecution time: {float((end - start)/60)} minutes")
    print(f"Total points for the season: {season_total}")
    plot_list(points_each_gw, title="Points Each Gameweek", xlabel="Gameweek", ylabel="Points")
