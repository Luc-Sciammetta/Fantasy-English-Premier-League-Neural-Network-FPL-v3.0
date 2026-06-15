import os
import requests
import pandas as pd
import pprint
import time

from optimize_team import optimizeFullTeam, optimizeTeamFormation, determine_transfers
from predictPlayerPoints import getTopPlayersForGameweek

SEASON = 2526

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
    budget_file = open(r"teamInfo/budget.txt", "r")
    budgets = budget_file.readlines()
    return int(budgets[-1]) #the last value in the file is the current budget

def getTeam(players_next_gw, player_next_7_points):
    team_file = open(r"teamInfo/team.txt", "r")
    teams = team_file.readlines()
    team = populateTeam(teams[-1].split(",")[:-1], players_next_gw, player_next_7_points)
    return team

def populateTeam(player_ids, players_next_gw, player_next_7_points):
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
    transfers_file = open(r"teamInfo/transfers.txt", "r")
    transfers = transfers_file.readlines()
    return (int(transfers[-1]) + 1) if int(transfers[-1]) < 5 else 5 

def saveBudget(val):
    budget_file = open(r"teamInfo/budget.txt", "a")
    budget_file.write(str(val) + "\n")
    budget_file.close()

def saveTransfers(val):
    transfers_file = open(r"teamInfo/transfers.txt", "a")
    transfers_file.write(str(val) + "\n")
    transfers_file.close()

def saveTeam(team):
    team_file = open(r"teamInfo/team.txt", "a")
    for player in team:
        team_file.write(str(player['id']) + ",")
    team_file.write("\n")
    team_file.close()

def run():
    gameweek = getCurrentGameweek()
    gameweek = 10
    players_next_gw, player_next_7_points = getTopPlayersForGameweek(gameweek, SEASON) 

    if gameweek == 1:
        team, budget = optimizeFullTeam(players_next_gw, player_next_7_points)
        transfers = 0
    else:
        budget = getBudget()
        team = getTeam(players_next_gw, player_next_7_points)
        transfers = getTransfers()

        print("Current team:")
        for player in team:
            print(player['first_name'], player['second_name'], player['team'], "Expected Points:", player['points_next_gw'])
            

        team, budget, transfers = determine_transfers(team, budget, transfers, players_next_gw, player_next_7_points, gameweek) #the new team, the remaining budget, the remaining transfers
        print(f"Remaining budget: {budget}, Remaining transfers: {transfers}")


    starters, bench, value = optimizeTeamFormation(team)

    print(f"-----Gameweek {gameweek}-----")
    print("Starters:")
    for starter in starters:
        print(starter["first_name"], starter["second_name"], starter["team"], "Expected Points:", starter["points_next_gw"])

    print()
    print("Bench:")
    for benched in bench:
        print(benched["first_name"], benched["second_name"], benched["team"], "Expected Points:", benched["points_next_gw"])
    
    saveTeam(team)
    saveBudget(budget)
    saveTransfers(transfers)


if __name__ == "__main__":
    start = time.time()
    run()
    end = time.time()
    print(f"Execution time: {float((end - start)/60)} minutes")
   