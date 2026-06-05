import pulp

from predictPlayerPoints import getTopPlayersForGameweek

def optimizeFullTeam(gw, season):
    players_next_gw, players = getTopPlayersForGameweek(gw, season) 

    problem = pulp.LpProblem("Fantasy Team Optimization", pulp.LpMaximize)

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

    problem.solve()

    team = []

    print("Status:", pulp.LpStatus[problem.status]) #determines if the optimization was successful
    print("Total Points:", pulp.value(problem.objective)) #total points of the optimized team
    print("Selected Players:")
    for i in range(0, len(players)):
        if x[i].varValue == 1: #if the player is selected in the team
            print(players[i][0], players[i][1], players[i][3], "Expected Points:", players[i][2])
            
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
                "points_next_gw": next_gw_points       #next GW predicted points
            })

    return team

def optimizeTeamFormation(team):
    problem = pulp.LpProblem("Fantasy Formation Optimization", pulp.LpMaximize)

    x = pulp.LpVariable.dict("player", range(0, len(team)), 0, 1, cat=pulp.LpInteger)

    problem += pulp.lpSum(team[i]['points_next_gw'] * x[i] for i in range(0, len(team)))

    problem += sum(x[i] for i in range(0, len(team))) == 11

    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 1) == 1 #1 goalkeeper
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 2) >= 3 #at least 3 defenders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 3) >= 2 #at least 2 midfielders
    problem += sum(x[i] for i in range(0, len(team)) if team[i]['element_type'] == 4) >= 1 #at least 1 forward

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
        print(starter["first_name"], starter["second_name"], starter["team"], "Expected Points:", starter["points_next_gw"])

    print()
    print("Bench:")
    for benched in bench:
        print(benched["first_name"], benched["second_name"], benched["team"], "Expected Points:", benched["points_next_gw"])
    
    return starters, bench

if __name__ == "__main__":
    gameweek = 27
    season = 2526
    team = optimizeFullTeam(gameweek, season)
    optimizeTeamFormation(team)