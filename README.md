# Fantasy-English-Premier-League-Neural-Network-FPL-v3.0

This project builds and manages a fifteen-player Fantasy Premier League squad using machine learning, allowing team selection and transfers to be guided by data rather than guesswork.

It relies on two models that work in tandem. A PyTorch neural network predicts how many points a player is expected to score over the next seven gameweeks, giving a sense of their longer-term value. An XGBoost gradient-boosting regressor then predicts the details of a player's next gameweek, such as goals, assists, minutes, and clean sheets, which informs the more immediate week-to-week decisions.

Drawing on these predictions, the program selects the strongest possible squad of fifteen players to maximise expected points, while staying within budget and respecting the usual position and club limits. Once a squad is in place, it also determines the best transfers to make ahead of each gameweek.

Taken together, it forms a thorough tool for playing Fantasy Premier League, applying machine learning to bring a more considered and informed approach to team selection and transfer strategy.

## Features
- Collects historical player data from [this git repository](https://github.com/vaastav/Fantasy-Premier-League/tree/master) 
- Preprocesses and cleans the raw data ready for modelling
- Uses a PyTorch neural network to predict total points over the next seven gameweeks
- Uses an XGBoost gradient-boosting regressor to predict goals, assists, minutes, clean sheets, and similar statistics for the next gameweek
- Selects the optimal squad of fifteen players through a dedicated optimisation algorithm
- Determines the best eleven starters and four bench players for each gameweek
- Determines the optimal transfers to make ahead of each gameweek

## Limitations/Future Work
- The program does not currently account for injuries/suspensions, which can significantly impact player performance
- The program uses information about the player's past 5 games and future 7 games. If this information is not available (gameweeks 1-5 or 33-38), the program will not be as strong as it could be. 
- This only takes into account the Premier League. Players could be rotated in/out based on other leagues could affect whether they play or not.
- Chip support is currently not implemented (wildcard, free hit, bench boost, tripple captain, assistant manager).

# Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```
3. Run the main script to start the program --> i dont have a main scrit yet bc its not done :D