# Documentation: matchPredictions_aaran.py

## Competition Structure
### About the World Hockey League
Your group will be crunching the numbers from the most recent season of the World Hockey League.
You will receive one season of data, including 32 teams and 82 games per team, yielding 1,312 total
games. For each game, you will see multiple rows depicting game stats broken down by matchups of
home and away teams’ offensive lines (first and second) and defensive pairings (first and second). Note
that there are no changes in team or line quality over the course of the season, and the regular season
is representative of the playoffs.
See the educational modules for more information.

### Phase 1a: Team Performance Analysis
**Predict game outcomes and win probabilities**
For the first round of the World Hockey League tournament, predict the win probability for the home
team for 16 matchups.
Submission: Winning probabilities of home teams in 16 matchups submitted by Student Team Leader
via SurveyMonkey Apply

---

## Technical Details: matchPredictions_aaran.py
This script uses the Bradley-Terry coefficients generated in the power rankings step to predict the outcome of the 16 tournament matchups.

### Methodology:
1. **Model**: Uses the Bradley-Terry probability formula:
   $$P(\text{Home Win}) = \frac{1}{1 + e^{-(\beta_{\text{home}} - \beta_{\text{away}})}}$$
   where $\beta$ is the team's strength coefficient from the logistic regression.
2. **Input**: Reads the 16 matchups from `assets/WHSDSC_Rnd1_matchups.xlsx`.
3. **Output**: Generates a list of winning probabilities for the home team in each matchup.
