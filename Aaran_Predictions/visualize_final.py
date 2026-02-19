import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

# Set seaborn style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# Load data
try:
    rankings_df = pd.read_csv('Aaran_Predictions/power_rankings_orlando.csv')
    line_analysis_df = pd.read_csv('Aaran_Predictions/line_analysis_full.csv')
except FileNotFoundError:
    # Fallback if running from within Aaran_Predictions folder
    rankings_df = pd.read_csv('power_rankings_orlando.csv')
    line_analysis_df = pd.read_csv('line_analysis_full.csv')

# Merge
merged_df = pd.merge(rankings_df, line_analysis_df, on='team')

# Sort by Rank (1 is best, so we want it at the top of the chart)
merged_df = merged_df.sort_values('rank')

# Prepare data for plotting
merged_df['off_disparity_centered'] = merged_df['off_disparity'] - 1

# Setup Plot
fig, ax = plt.subplots(figsize=(12, 14))

# Create positions
y_pos = np.arange(len(merged_df))

# Use a colormap based on the disparity value
norm = plt.Normalize(merged_df['off_disparity_centered'].min(), merged_df['off_disparity_centered'].max())
colors = plt.cm.RdBu_r(norm(merged_df['off_disparity_centered']))

# Plot Horizontal Bars
rects = ax.barh(y_pos, merged_df['off_disparity_centered'], align='center', color=colors, alpha=0.9, height=0.7)

# Add vertical line at 0 (Balance point)
ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)

# Formatting Y-Axis (Teams)
ax.set_yticks(y_pos)

# Create labels: "1. Team Name" with specific formatting
def format_team_name(team):
    team_map = {
        'uae': 'UAE',
        'usa': 'USA',
        'uk': 'UK',
        'new_zealand': 'New Zealand',
        'south_korea': 'South Korea',
        'saudi_arabia': 'Saudi Arabia'
    }
    if team.lower() in team_map:
        return team_map[team.lower()]
    return team.replace('_', ' ').title()

yticklabels = [f"{int(rank)}. {format_team_name(team)}" for rank, team in zip(merged_df['rank'], merged_df['team'])]
ax.set_yticklabels(yticklabels, fontsize=11)
ax.invert_yaxis()  # Put Rank 1 at the top

# Formatting X-Axis
ax.set_xlabel('Offensive Line Disparity (Centered at 1.0)\nPositive → 1st Line is Stronger | Negative → 2nd Line is Stronger', fontsize=12, labelpad=15, weight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Title
plt.suptitle('Offensive Line Quality Disparity vs. Team Power Ranking', fontsize=18, weight='bold', y=0.96)
ax.set_title('Teams ordered by Power Ranking (1 = Strongest)', fontsize=12, style='italic', pad=10, color='gray')

# Add values to bars
for rect, value in zip(rects, merged_df['off_disparity']):
    width = rect.get_width()
    offset = 0.005
    if width >= 0:
        x_pos = width + offset
        ha = 'left'
    else:
        x_pos = width - offset
        ha = 'right'
    
    ax.text(x_pos, rect.get_y() + rect.get_height()/2, f'{value:.2f}', ha=ha, va='center', fontsize=9, color='black')

# Trend Analysis with P-Value
rank_corr, p_value = pearsonr(merged_df['rank'], merged_df['off_disparity'])

if p_value > 0.05:
    interpretation = "No significant correlation.\nSuccess is not dependent on line structure."
elif rank_corr < 0:
    interpretation = "Statistically significant trend:\nStronger teams tend to be more top-heavy."
else:
    interpretation = "Statistically significant trend:\nStronger teams tend to be more balanced."

stats_text = (
    f"Trend Analysis:\n"
    f"• Correlation (r): {rank_corr:.2f}\n"
    f"• P-Value: {p_value:.3f}\n"
    f"• Insight: {interpretation}\n\n"
    f"Metric Key:\n"
    f"• Value > 1.0: 1st Line > 2nd Line (Top Heavy)\n"
    f"• Value ≈ 1.0: Balanced Attack\n"
    f"• Value < 1.0: 2nd Line > 1st Line (Depth)"
)

# Place text box
plt.text(0.72, 0.85, stats_text, transform=ax.transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Make room for suptitle

# Save
output_file = 'Nihalytical.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_file}")
