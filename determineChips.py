from optimize_team import optimizeFreeHitTeam, optimizeTeamFormation, optimizeFullTeam

FREE_HIT_THRESH = 30
TC_THRESH = 14.5
BENCH_THRESH = 26
WC_THRESH = 50

THRESH_DECREASE_FACTOR = 0.90
RAMP_GWS = 6

def lastChipWindow(current_gw):
    """Returns when the deadline to play all chips are based on what gameweek it is currently."""
    return 18 if current_gw < 19 else 38

def getTeamGamesInGW(player_team, fixturesdf, current_gw, end_gw = 0):
    """Returns the number of games that a team has in a given gameweek(s)."""
    fixtures = fixturesdf[((fixturesdf['event'] >= current_gw) & (fixturesdf['event'] <= current_gw+end_gw)) & ((fixturesdf['team_a'] == player_team) | (fixturesdf['team_h'] == player_team))]
    return len(fixtures)

def calculateTeamPoints(team, metric):
    total = 0
    for player in team:
        total += player[metric]
    return total

def getBudget():
    """Gets the current budget from the budget.txt file."""
    budget_file = open(r"teamInfo2/budget.txt", "r")
    budgets = budget_file.readlines()
    return int(budgets[-1]) #the last value in the file is the current budget

def getTeamValue(team, budget):
    value = budget
    for player in team:
        value += player['cost']
    return value

def getFormation(team):
    starters, bench, value = optimizeTeamFormation(team)
    capitain = starters[0]
    vice_captain = starters[1]

    return starters, bench, capitain, vice_captain

def determineFreeHit(team, current_gw, fixturesdf, starters, players_next_gw, fc_thresh, team_budget):
    """Loop for every gw till the chip reset window, calculate how many points we'd loose from blanks, and then get then put that in a table."""
    last_gw = lastChipWindow(current_gw)
    table = {}
    for gw in range(current_gw, last_gw+1):
        total_score = 0
        for player in team:
            player_team = player['team']
            games = getTeamGamesInGW(player_team, fixturesdf, gw)
            if games != 0:
                continue #not a blank

            games_next_7 = getTeamGamesInGW(player_team, fixturesdf, current_gw, 7)
            avg_score = float(player['points'] / games_next_7) #average amount of points they'll score in this gw if they had one (based of off next_7)
            total_score += avg_score
        
        table[gw] = total_score

    if any(value > 0 for value in table.values()): #if any value in the dict is >0, then we have at least one blank gw
        return table, 0 #here is the list of gws to play, normal flag
    

    # ----- if we get here, then we look at calculating the optimal team vs our team -----
    current_team_budget = getTeamValue(team, team_budget)
    current_team_value = calculateTeamPoints(starters, 'points_next_gw')
    opt_team, _ = optimizeFreeHitTeam(players_next_gw, current_team_budget)
    opt_starters, _, _, _ = getFormation(opt_team)
    opt_team_value = calculateTeamPoints(opt_starters, 'points_next_gw')

    if opt_team_value - current_team_value > fc_thresh:
        return {current_gw: opt_team_value - current_team_value}, 1 #could play free hit this week, but "flag" this gw
    else:
        return {}, 0 #Not playing free hit this week
    

def determineTripleCaptain(current_gw, starters, tc_thresh):
    candidates = [starters[0]]
    for player in candidates:
        if player['points_next_gw'] < tc_thresh:
            return {}, 0 #not playing TC this week
    
    #if players are above the threshold, we can play triple captain
    return {current_gw: (candidates[0]['points_next_gw'])}, 0 #dont multiply by 3 since the chip will get us another ['points_next_gw'] points, so thats how much its worth

def determineBenchBoost(current_gw, bench, bb_thresh):
    total = 0
    for player in bench:
        total += player['points_next_gw']
    
    if total < bb_thresh:
        return {}, 0
    return {current_gw: total}, 0

def determineWildCard(team, current_gw, players_next_7, players_next_gw, wc_thresh, team_budget):
    if current_gw < 4: #dont play wildcard in the first 4 gameweeks, since we can make free transfers and we dont have enough data to optimize a team yet
        return {}, 0

    current_team_budget = getTeamValue(team, team_budget)
    current_team_value = calculateTeamPoints(team, 'points')
    opt_team, _ = optimizeFullTeam(players_next_gw, players_next_7, current_team_budget)
    opt_team_value = calculateTeamPoints(opt_team, 'points')

    if opt_team_value - current_team_value > wc_thresh:
        return {current_gw: opt_team_value - current_team_value}, 0 #play wild card
    
    return {}, 0 #dont play wild card

def changeThesholds(current_gw):
    last_gw = lastChipWindow(current_gw)
    if last_gw - current_gw <= RAMP_GWS: #decrease the thresholds
        print("Getting close to end of chip window, decreasing thresholds.")
        fc_thresh = FREE_HIT_THRESH * (THRESH_DECREASE_FACTOR ** (RAMP_GWS+1 - (last_gw - current_gw)))
        tc_thresh = TC_THRESH * (THRESH_DECREASE_FACTOR ** (RAMP_GWS+1 - (last_gw - current_gw)))
        bb_thresh = BENCH_THRESH * (THRESH_DECREASE_FACTOR ** (RAMP_GWS+1 - (last_gw - current_gw)))
        wc_thresh = WC_THRESH * (THRESH_DECREASE_FACTOR ** (RAMP_GWS+1 - (last_gw - current_gw)))
        return fc_thresh, tc_thresh, bb_thresh, wc_thresh
    return FREE_HIT_THRESH, TC_THRESH, BENCH_THRESH, WC_THRESH

def whatToPlay(current_gw, team, starters, bench, fixturesdf, players_next_7, players_next_gw, team_budget, chips):
    fc_thresh, tc_thresh, bb_thresh, wc_thresh = changeThesholds(current_gw)

    if 'free hit' in chips:
        fh_table, fh_flag = determineFreeHit(team, current_gw, fixturesdf, starters, players_next_gw, fc_thresh, team_budget)
    else:
        fh_table, fh_flag = {}, 0

    if 'x3 capitain' in chips:
        tc_table, _ = determineTripleCaptain(current_gw, starters, tc_thresh)
    else:
        tc_table = {}

    if 'bench boost' in chips:
        bb_table, _ = determineBenchBoost(current_gw, bench, bb_thresh)
    else:
        bb_table = {}

    if 'wildcard' in chips:
        wc_table, _ = determineWildCard(team, current_gw, players_next_7, players_next_gw, wc_thresh, team_budget)
    else:
        wc_table = {}

    best_fh_gw = 0
    best_fh_score = 0
    for key, value in fh_table.items():
        if value > best_fh_score:
            best_fh_gw = key
            best_fh_score = value

    if best_fh_gw == current_gw and fh_flag == 0: #do free hit now
        return 1
    
    scores = {}
    if tc_table:
        scores[2] = tc_table[current_gw]
    if bb_table:
        scores[3] = bb_table[current_gw]
    if fh_table:
        scores[1] = fh_table[current_gw]
    if wc_table:
        scores[0] = wc_table[current_gw] / 7 #divide by 7 to get the average points per gameweek, since wild card affects all 7 gameweeks. This way we can compare it more fairly to the other chips which only affect one gameweek.
    
    print('\nTriple Captain Score:', tc_table)
    print('Bench Boost Score:', bb_table)
    print('Free Hit Score:', fh_table)
    print('Wildcard Score:', wc_table)

    if scores:
        max_chip = max(scores, key=scores.get)
        return max_chip
    
    return -1 #nothing to play now