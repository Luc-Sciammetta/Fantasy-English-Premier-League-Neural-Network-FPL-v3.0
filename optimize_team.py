import pulp

from predict_player_points import getTopPlayersForGW

def optimizeFullTeam(gw):
    players_next_gw, players = getTopPlayersForGW(gw) 

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

    team = []

    print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    print("Selected Players:")
    for i in range(0, len(players)):
        if x[i].varValue == 1: #if the player is selected in the team
            print(players[i]['stuff']["first_name"], players[i]['stuff']["second_name"], players[i]['stuff']["team"], "Expected Points:", players[i]["points"])
            
            next_gw_points = next((p["points"] for p in players_next_gw if p['stuff']['id'] == players[i]['stuff']['id']), None)
        
            team.append({
                "first_name": players[i]['stuff']["first_name"],
                "second_name": players[i]['stuff']["second_name"],
                "stuff": players[i]['stuff'],
                "points": players[i]["points"],        #7GW predicted points
                "points_next_gw": next_gw_points       #next GW predicted points
            })

    return team

def optimizeTeamFormation(team):
    problem = pulp.LpProblem("Fantasy Formation Optimization", pulp.LpMaximize)

    x = pulp.LpVariable.dict("player", range(0, len(team)), 0, 1, cat=pulp.LpInteger)

    problem += pulp.lpSum(team[i]['points_next_gw'] * x[i] for i in range(0, len(team)))

    problem += sum(x[i] for i in range(0, len(team))) == 11

    problem += sum(x[i] for i in range(0, len(team)) if team[i]['stuff']["element_type"] == 1) == 1 #1 goalkeeper
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['stuff']["element_type"] == 2) >= 3 #at least 3 defenders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['stuff']["element_type"] == 3) >= 2 #at least 2 midfielders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['stuff']["element_type"] == 4) >= 1 #at least 1 forward

    problem.solve()

    starters = []
    bench = []

    print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    for i in range(len(team)):
        if x[i].varValue == 1:
            starters.append(team[i])
        else:
            bench.append(team[i])

    print("Starters:")
    for starter in starters:
        print(starter["first_name"], starter["second_name"], starter['stuff']["team"], "Expected Points:", starter["points_next_gw"])

    print()
    print("Bench:")
    for benched in bench:
        print(benched["first_name"], benched["second_name"], benched['stuff']["team"], "Expected Points:", benched["points_next_gw"])
    
    return starters, bench

if __name__ == "__main__":
    gameweek = 15
    team = optimizeFullTeam(gameweek)
    optimizeTeamFormation(team)