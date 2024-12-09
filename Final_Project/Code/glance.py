import os
# import ipdb
# ipdb.set_trace()
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('wind_gen_cf_2020.csv', parse_dates=['datetime'], index_col='datetime')

output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

def pre(data):
    print("Dataset Shape:", data.shape)
    print("\nFirst 5 Rows of Data:\n", data.head())
    print("\nData Info:")
    data.info()

    summary_stats = data.describe()
    print("\nSummary Statistics for Each Plant:\n", summary_stats)

    max_gen_plant = data.max().idxmax()
    min_gen_plant = data.min().idxmin()
    print(f"\nPlant with maximum generation: {max_gen_plant} (Max Value: {data[max_gen_plant].max()})")
    print(f"Plant with minimum generation: {min_gen_plant} (Min Value: {data[min_gen_plant].min()})")

    total_generation = data.sum(axis=1)

    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['week_of_year'] = data.index.isocalendar().week

    daily_avg_generation = data.iloc[:, :-5].resample('D').mean().sum(axis=1)
    monthly_avg_generation = data.iloc[:, :-5].resample('M').mean().sum(axis=1)

    correlation_matrix = data.corr()

    hourly_avg_generation = data.groupby("hour").mean().mean(axis=1)

    return {
        "total_generation": total_generation,
        "daily_avg_generation": daily_avg_generation,
        "monthly_avg_generation": monthly_avg_generation,
        "correlation_matrix": correlation_matrix,
        "hourly_avg_generation": hourly_avg_generation,
        "summary_stats": summary_stats,
        "max_gen_plant": max_gen_plant,
        "min_gen_plant": min_gen_plant,
    }

def visualize(plant_data, processed_data):
    sns.set(style="whitegrid")

    # Plant location scatter plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x="lon", y="lat", data=plant_data, hue="state", palette="tab20", s=50, legend="full"
    )
    plt.title("Wind Plant Locations by State")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wind_plant_locations.png"))
    plt.show()

    # Top 5 states by number of plants
    state_counts = plant_data['state'].value_counts()
    top_5_states = state_counts.head(5)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_5_states.index, y=top_5_states.values, palette="viridis")
    plt.title("Top 5 States by Number of Wind Plants")
    plt.xlabel("State")
    plt.ylabel("Number of Plants")
    plt.savefig(os.path.join(output_dir, "top_5_states_wind_plants.png"))
    plt.show()

    # Total wind generation over time (Hourly)
    plt.figure(figsize=(15, 6))
    plt.plot(processed_data["total_generation"], label="Total Wind Generation Across All Plants")
    plt.title("Total Wind Generation Over Time (Hourly)")
    plt.xlabel("Time")
    plt.ylabel("Total Generation (MW)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "total_generation_over_time.png"))
    plt.show()

    # Daily and Monthly Average Generation
    plt.figure(figsize=(10, 5))
    plt.plot(processed_data["daily_avg_generation"], label="Daily Average Generation (MW)")
    plt.plot(processed_data["monthly_avg_generation"], label="Monthly Average Generation (MW)", linestyle="--")
    plt.title("Daily and Monthly Average Generation Across All Plants")
    plt.xlabel("Date")
    plt.ylabel("Average Generation (MW)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "daily_monthly_avg_generation.png"))
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(processed_data["correlation_matrix"], cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Plants")
    plt.savefig(os.path.join(output_dir, "correlation_between_plants.png"))
    plt.show()

    # Hourly average generation pattern
    plt.figure(figsize=(10, 5))
    plt.plot(processed_data["hourly_avg_generation"], marker='o')
    plt.title("Average Generation by Hour of Day (Across All Plants)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Generation (MW)")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "hourly_avg_generation_pattern.png"))
    plt.show()

if __name__ == '__main__':
    # Process the data
    # processed_data = pre(data)

    # # Load the plant configuration data for location visualizations
    # plant_data = pd.read_csv("eia_wind_configs.csv")

    # # Visualize the data and save plots
    # visualize(plant_data, processed_data)
    data = pd.read_csv("Wind_capacity_99MW.csv")
    print(data.info())
