import os
import pulp

PENALTY = 8  #done so that the optimization will prefer to make free transfers unless its absolutely sure that it will be a good trade
            #so we double the actual cost of making an extra costly transfer

def optimizeFullTeam(players_next_gw, players):
    """Optimizes and gets a full team of 15 players based on player predicted points for the next 7 gameweeks
    Args:
        players_next_gw: list of tuples (first_name, second_name, predicted_points_next_gw) for all players
        players: list of tuples (first_name, second_name, predicted_points_next_7_gw, team, element_type, cost, id) for all players
    Returns:
        team: list of dicts with player info for the optimized team
    """
    problem = pulp.LpProblem("FantasyTeamOptimization", pulp.LpMaximize)

    #variables made (in a dict) for each possible player. They can be 0 or 1, where 1 means they made the team and 0 means they didnt
    x = pulp.LpVariable.dict("player", range(0, len(players)), 0, 1, cat=pulp.LpInteger)

    problem += pulp.lpSum(players[i][2] * x[i] for i in range(0, len(players))) #what we want to maximize (total points)

    problem += sum(x[i] for i in range(0, len(players))) == 15 #15 players in the team
    problem += pulp.lpSum(players[i][5] * x[i] for i in range(0, len(players))) <= 1000 #total cost must be less than or equal to 100

    problem += sum(x[i] for i in range(0, len(players)) if players[i][4] == 1) == 2 #2 goalkeepers
    problem += sum(x[i] for i in range(0, len(players)) if players[i][4] == 2) == 5 #5 defenders
    problem += sum(x[i] for i in range(0, len(players)) if players[i][4] == 3) == 5 #5 midfielders
    problem += sum(x[i] for i in range(0, len(players)) if players[i][4] == 4) == 3 #3 forwards

    for team in range(1, 21): #for every team in the league
        problem += sum(x[i] for i in range(0, len(players)) if players[i][3] == team) <= 3 #max 3 players from the same team

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    team = []
    spent = 0

    # print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    # print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    # print("Selected Players:")
    for i in range(0, len(players)):
        if x[i].varValue == 1: #if the player is selected in the team
            # print(players[i][0], players[i][1], players[i][3], "Expected Points:", players[i][2])
            
            spent += players[i][5]

            next_gw_points = next(
                (p[3] for p in players_next_gw
                if p[0] == players[i][0] and p[1] == players[i][1]),
                0.0   # default: player not in next-GW list -> 0, not None
            )

            team.append({
                "first_name": players[i][0],
                "second_name": players[i][1],
                "team": players[i][3],
                "element_type": players[i][4],
                "points": players[i][2],        #7GW predicted points
                "points_next_gw": next_gw_points,       #next GW predicted points
                "cost": players[i][5],
                "id": players[i][6]
            })

    return team, (1000 - spent) #return the team and the remaining budget

def optimizeTeamFormation(team):
    """Optimizes the team formation for the next gameweek based on predicted points for the next gameweek.
    Args:
        team: list of dicts with player info for the current team
    Returns:
        starters: list of dicts with player info for the optimized starting 11
        bench: list of dicts with player info for the optimized bench
        total_points: total predicted points for the starting 11 for the next gameweek
    """
    problem = pulp.LpProblem("FantasyFormationOptimization", pulp.LpMaximize)

    x = pulp.LpVariable.dict("player", range(0, len(team)), 0, 1, cat=pulp.LpInteger)

    problem += pulp.lpSum(team[i]['points_next_gw'] * x[i] for i in range(0, len(team)))

    problem += sum(x[i] for i in range(0, len(team))) == 11

    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 1) == 1 #1 goalkeeper
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 2) >= 3 #at least 3 defenders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 3) >= 2 #at least 2 midfielders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 4) >= 1 #at least 1 forward

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    starters = []
    bench = []

    # print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    # print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    for i in range(len(team)):
        if x[i].varValue == 1:
            starters.append(team[i])
        else:
            bench.append(team[i])
    
    return sorted(starters, key=lambda x: x['points_next_gw'], reverse=True), sorted(bench, key=lambda x: x['points_next_gw'], reverse=True), pulp.value(problem.objective)

def determine_transfers(team, budget, free_transfers, next_gw, next_7):
    """Determines the optimal transfers to make for the next gameweek based on player predicted points for the next 7 gameweeks and the next gameweek.
    Args:
        team: list of dicts with player info for the current team
        budget: current budget available for transfers
        free_transfers: number of free transfers available
        next_gw: list of tuples (first_name, second_name, predicted_points_next_gw) for all players
        next_7: list of tuples (first_name, second_name, predicted_points_next_7_gw, team, element_type, cost, id) for all players
    Returns:
        new_team: list of dicts with player info for the new team after transfers
        new_budget: remaining budget after transfers
        free_left: remaining free transfers after transfers
        old_team: list of dicts with player info for the old team (before transfers)
    """
    problem = pulp.LpProblem("FantasyTeamTransferOptimization", pulp.LpMaximize)

    x = pulp.LpVariable.dict("player", range(0, len(next_7)), 0, 1, cat=pulp.LpInteger)

    owned_names = {(p['first_name'], p['second_name']) for p in team}
    owned = [1 if (next_7[i][0], next_7[i][1]) in owned_names else 0 for i in range(len(next_7))]

    paid_transfers = pulp.LpVariable("paid_transfers", lowBound=0)
    problem += paid_transfers >= (15 - pulp.lpSum(owned[i] * x[i] for i in range(len(next_7)))) - free_transfers #number of paid transfers must be at least the number of new players in the team minus the free transfers available

    problem += (pulp.lpSum(next_7[i][2] * x[i] for i in range(len(next_7))) - (4+PENALTY) * paid_transfers) #maximize points of new team minus 4 points for each paid transfer

    squad_value = sum(next_7[i][5] for i in range(len(next_7)) if owned[i]) #value of the players in the new team that are already owned
    total_funds = budget + squad_value 
    problem += pulp.lpSum(next_7[i][5] * x[i] for i in range(len(next_7))) <= total_funds #total cost of the new team must be less than or equal to the total funds available

    problem += sum(x[i] for i in range(0, len(next_7))) == 15 #15 players in the team

    problem += sum(x[i] for i in range(0, len(next_7)) if next_7[i][4] == 1) == 2 #2 goalkeepers
    problem += sum(x[i] for i in range(0, len(next_7)) if next_7[i][4] == 2) == 5 #5 defenders
    problem += sum(x[i] for i in range(0, len(next_7)) if next_7[i][4] == 3) == 5 #5 midfielders
    problem += sum(x[i] for i in range(0, len(next_7)) if next_7[i][4] == 4) == 3 #3 forwards

    for club in range(1, 21): #for every team in the league
        problem += sum(x[i] for i in range(0, len(next_7)) if next_7[i][3] == club) <= 3 #max 3 players from the same team

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # print(paid_transfers.varValue)

    new_team = []
    spent = 0

    # print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    # print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    # print("Selected Players:")
    for i in range(0, len(next_7)):
        if x[i].varValue == 1: #if the player is selected in the team
            # print(next_7[i][0], next_7[i][1], next_7[i][3], "Expected Points:", next_7[i][2])
            
            spent += next_7[i][5]

            next_gw_points = next(
                (p[3] for p in next_gw
                if p[0] == next_7[i][0] and p[1] == next_7[i][1]),
                0.0   # default: player not in next-GW list -> 0, not None
            )

            new_team.append({
                "first_name": next_7[i][0],
                "second_name": next_7[i][1],
                "team": next_7[i][3],
                "element_type": next_7[i][4],
                "points": next_7[i][2],        #7GW predicted points
                "points_next_gw": next_gw_points,       #next GW predicted points
                "cost": next_7[i][5],
                "id": next_7[i][6]
            })

    new_budget = total_funds - spent
    transfers_made = sum(1 for i in range(len(next_7)) if x[i].varValue == 1 and owned[i] == 0) #number of new players in the team
    free_left = max(0, free_transfers - transfers_made)

    if pulp.LpStatus[problem.status] != "Optimal": #the optimization didnt find a solution
        return team, budget, free_transfers, team 
    
    return new_team, new_budget, free_left, team


if __name__ == "__main__":
    gameweek = 30
    season = 2526
    team = optimizeFullTeam(gameweek, season)
    optimizeTeamFormation(team)