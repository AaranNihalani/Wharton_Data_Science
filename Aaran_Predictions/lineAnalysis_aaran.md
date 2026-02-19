# Documentation: lineAnalysis_aaran.py

## Competition Structure
### Phase 1b: Line Performance Analysis
**Quantify team offensive line quality disparity**
Hockey teams use multiple offensive lines and defensive pairings. The first offensive line is often more productive than the secondary lines — meaning they tend to generate more scoring opportunities. Your task is to quantify how large that disparity is for each team.

**Requirements:**
1.  **Offensive Performance Measure**: For each offensive line, form a measure based on expected goals (xG), accounting for:
    *   Differences in Time On Ice (TOI).
    *   Defensive matchups (tougher opponents).
2.  **Disparity Ratio**: Calculate the ratio of the first line's performance to the secondary line's performance.
3.  **Ranking**: Rank the top 10 WHL teams on offensive line quality disparity from largest to smallest.

---

## Technical Details: lineAnalysis_aaran.py
This script performs the line disparity analysis using a **Strength of Schedule (SOS) Adjusted xG Rate** method.

### Methodology:

1.  **Defensive Strength Calculation**:
    *   We first calculate the **xG Allowed per 60 minutes** for every unique defensive pairing (Team + Pairing) in the league.
    *   `Def_Rating = Sum(xG_Against) / Sum(TOI) * 3600`
    *   A higher rating indicates a weaker defense (allows more xG).
    *   We also calculate the global **League Average xG Allowed per 60**.

2.  **Offensive Performance Adjustment**:
    *   For every offensive line (`first_off` and `second_off`), we calculate their **Raw xG per 60**.
    *   We then calculate the weighted average defensive rating of the opponents they faced (`Avg_Opp_Def_Rating`), weighted by TOI.
    *   **Adjustment Formula**:
        $$ \text{Adj\_Rate} = \text{Raw\_Rate} \times \left( \frac{\text{League\_Avg\_Def}}{\text{Avg\_Opp\_Def\_Rating}} \right) $$
    *   *Logic*: If a line played against defenses that allow *more* goals than average (high rating), their raw stats are inflated, so we discount them. If they played against stingy defenses (low rating), we boost their stats.

3.  **Disparity Ratio**:
    *   $$ \text{Ratio} = \frac{\text{Adj\_xG\_Rate (First Line)}}{\text{Adj\_xG\_Rate (Second Line)}} $$
    *   Teams are ranked by this ratio in descending order.

### Output
*   **Console**: Prints the League Average xG/60 and the Top 10 Teams by Disparity Ratio.
*   **CSV**: Saves the full analysis to `line_disparity_analysis.csv`.
