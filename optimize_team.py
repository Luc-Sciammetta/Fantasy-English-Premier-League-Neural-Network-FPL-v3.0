import pulp

from predict_player_points import getTopPlayersForGW

def optimizeFullTeam(gw):
    _, players = getTopPlayersForGW(gw) 

    problem = pulp.LpProblem("Fantasy Team Optimization", pulp.LpMaximize)

    #variables made (in a dict) for each possible player. They can be 0 or 1, where 1 means they made the team and 0 means they didnt
    x = pulp.LpVariable.dict("player", range(0, len(players)), 0, 1, cat=pulp.LpInteger)

    problem += pulp.lpSum(players[i]["points"] * x[i] for i in range(0, len(players))) #what we want to maximize (total points)

    problem += sum(x[i] for i in range(0, len(players))) == 15 #15 players in the team
    problem += pulp.lpSum(players[i]['stuff']["now_cost"] * x[i] for i in range(0, len(players))) <= 1000 #total cost must be less than or equal to 100

    problem += sum(x[i] for i in range(0, len(players)) if players[i]['stuff']["element_type"] == 1) == 2 #2 goalkeepers
    problem += sum(x[i] for i in range(0, len(players)) if players[i]['stuff']["element_type"] == 2) == 5 #5 defenders
    problem += sum(x[i] for i in range(0, len(players)) if players[i]['stuff']["element_type"] == 3) == 5 #5 midfielders
    problem += sum(x[i] for i in range(0, len(players)) if players[i]['stuff']["element_type"] == 4) == 3 #3 forwards

    for team in range(1, 21): #for every team in the league
        problem += sum(x[i] for i in range(0, len(players)) if players[i]['stuff']["team"] == team) <= 3 #max 3 players from the same team

    problem.solve()

    print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    print("Selected Players:")
    for i in range(0, len(players)):
        if x[i].varValue == 1: #if the player is selected in the team
            print(players[i]['stuff']["first_name"], players[i]['stuff']["second_name"], players[i]['stuff']["team"], "Expected Points:", players[i]["points"])


optimizeFullTeam(37)