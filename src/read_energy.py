import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

energy_potential_profiles = np.load('data/total_percentage_PV_profiles.npy')

# labels
years = [2020, 2021, 2022, 2023]
seasons = ['Autumn', 'Spring', 'Summer', 'Winter']
technologies = ['EmergingPV', 'EmergingPV Potential', 'Silicon', 'Silicon w/ missmatch decrease']
hours = [f"{i}" for i in range(0, 24)]

# Step 3: Reshape the data for the DataFrame
reshaped_data = energy_potential_profiles.reshape(-1, 24)  # Combine first three dimensions, keep 24-hour data

# Step 4: Create a multi-index
index = pd.MultiIndex.from_product(
    [years, seasons, technologies],
    names=["Year", "Season", "Technology"]
)

# Step 5: Create the DataFrame
df = pd.DataFrame(reshaped_data, index=index, columns=hours)
season_tech_mean = {}

# Loop through each season and technology
for tech in technologies:
    for season in seasons:
        # Calculate mean for the current technology and season
        mean_value = df.loc[(slice(None), season, tech)].mean()
        
        # Store the result in a dictionary
        season_tech_mean[(season, tech)] = mean_value

# Convert results to a DataFrame 
mean_df = pd.DataFrame(season_tech_mean).T  # Transpose to have (season, tech) as rows
mean_df.columns = [f'Hour_{i}' for i in range(24)]  # Assign hour columns
mean_df.index.names = ["Season", "Technology"]

# Display the resulting DataFrame
print(mean_df)

# Step 7: Calculate the mean for all seasons across all years and technologies
season_mean = df.groupby(level="Season").mean()

print("\nMean Data for Each Season (Across All Years and Technologies):")
print(season_mean)

# Define line styles and colors
line_styles = ['-', '--', '-.', ':']  # Different line styles for technologies
technologies = mean_df.index.get_level_values('Technology').unique()
seasons = mean_df.index.get_level_values('Season').unique()
colors = ['blue', 'orange', 'green', 'red']  # Colors for seasons

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each technology's mean per season for 24 hours
for color, season in zip(colors, seasons):  # Assign colors to seasons
    for line_style, tech in zip(line_styles, technologies):  # Assign line styles to technologies
        hourly_data = mean_df.loc[(season, tech)]  # Get the hourly data
        ax.plot(
            range(24),  # Hourly data index
            hourly_data.values,
            label=f"{season} - {tech}",
            linestyle=line_style,
            color=color
        )

# Create custom legends
season_legend = [Line2D([0], [0], color=color, lw=2, label=season) for color, season in zip(colors, seasons)]
technology_legend = [Line2D([0], [0], color='black', lw=2, linestyle=line_style, label=tech) for line_style, tech in zip(line_styles, technologies)]

# Add legends
legend1 = ax.legend(handles=season_legend, title="Seasons", loc="upper left", bbox_to_anchor=(1, 1))
legend2 = ax.legend(handles=technology_legend, title="Technologies", loc="upper left", bbox_to_anchor=(1, 0.7))

# Add legends to the plot
ax.add_artist(legend1)

# Add labels, title, and grid
ax.set_title("Hourly Mean per Technology and Season", fontsize=16)
ax.set_xlabel("Hour", fontsize=14)
ax.set_ylabel("Mean Value", fontsize=14)
ax.grid()

plt.tight_layout()
plt.show()

mean_df.to_csv('data/mean_power_potential.csv')

