# Documentation: powerRankings_aaran.py

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
**Create team power rankings**
Based on season data, rank the 32 teams using the underlying team performance. But this isn’t just
about win-loss records. Your rankings should reflect the overall strength and quality of the teams.
Submission: Power ranking of all 32 teams submitted by Student Team Leader via SurveyMonkey
Apply

---

## Technical Details: powerRankings_aaran.py
This script implements a **Bradley-Terry Logistic Regression** model to rank the 32 teams in the WHL.

### Methodology:
1. **Game Aggregation**: Converts the raw row-level data into game-level outcomes (Home Win vs. Away Win).
2. **Feature Engineering**: Creates a feature matrix where each row represents a game, with +1 for the home team and -1 for the away team.
3. **Logistic Regression**: Fits a Logit model using `statsmodels` to estimate the relative strength ($\beta$) of each team.
4. **Ranking**: Teams are ranked based on their estimated strength coefficients. 
5. **Normalization**: To ensure the rankings are independent of the choice of reference team, we apply **Mean-Zero Normalization** to the final scores. This ensures that a team with a positive score is above average, and a negative score is below average.

### **Bias Verification**
We verified the model by running it with two different reference teams (Brazil and Vietnam). The resulting rankings and relative strength differences were identical (Correlation = 1.0), confirming the model's mathematical robustness against reference selection.
