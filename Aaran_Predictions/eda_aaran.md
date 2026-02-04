# Documentation: eda_aaran.py

## Competition Structure
### About the World Hockey League
Your group will be crunching the numbers from the most recent season of the World Hockey League.
You will receive one season of data, including 32 teams and 82 games per team, yielding 1,312 total
games. For each game, you will see multiple rows depicting game stats broken down by matchups of
home and away teams’ offensive lines (first and second) and defensive pairings (first and second). Note
that there are no changes in team or line quality over the course of the season, and the regular season
is representative of the playoffs.
See the educational modules for more information.

### Phase 1: Main Competition
The competition heats up as you prepare to dive into the action of a WHL tournament! Your mission: as
an analytics group, crunch the numbers to provide an outlook for the WHL tournament, using the
provided in-season data.
Your strategy and predictions will be submitted through the online platform SurveyMonkey Apply. This is
where the groundwork is laid, and the sharpest solutions will advance to the next stage.

### Phase 1a: Team Performance Analysis
Before diving into Phase 1a, first create a league table outlining teams’ overall standings for the season
following the online educational module using your method of choice (R, Python, Google Sheets,
GenAI/LLM). The league table will not be submitted on its own, but it is a necessary first step to
creating team power rankings and predicting game outcomes for your Phase 1a submission.

---

## Technical Details: eda_aaran.py
This script performs Exploratory Data Analysis on the WHL 2025 dataset.

### Key Features:
1. **Data Loading**: Loads `assets/whl_2025.csv` using pandas.
2. **Team Statistics**: Aggregates goals, xG, shots, and penalties for all 32 teams.
3. **Goalie Evaluation**: Calculates Goals Saved Above Expected (GSAx) and Save Percentage for all goalies.
4. **Efficiency Analysis**: Identifies top offensive lines and teams with the best finishing efficiency (Actual Goals - Expected Goals).
5. **Game Metrics**: Analyzes average goals per game and OT frequency.
