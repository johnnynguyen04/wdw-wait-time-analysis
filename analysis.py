import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

DATA_DIR = '/mnt/user-data/uploads'
OUT_DIR = '/home/claude/analysis_output'
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# STEP 1: Load all ride data
# =====================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

ride_files = {
    'Toy Story Mania': 'toy_story_mania.csv',
    'Rock n Roller Coaster': 'rock_n_rollercoaster.csv',
    'Slinky Dog Dash': 'slinky_dog.csv',
    'Alien Swirling Saucers': 'alien_saucers.csv',
    'Seven Dwarfs Mine Train': 'seven_dwarfs_train.csv',
    'Flight of Passage': 'flight_of_passage.csv',
    'Soarin': 'soarin.csv',
    'Pirates of the Caribbean': 'pirates_of_caribbean.csv',
}

ride_parks = {
    'Toy Story Mania': 'Hollywood Studios',
    'Rock n Roller Coaster': 'Hollywood Studios',
    'Slinky Dog Dash': 'Hollywood Studios',
    'Alien Swirling Saucers': 'Hollywood Studios',
    'Seven Dwarfs Mine Train': 'Magic Kingdom',
    'Flight of Passage': 'Animal Kingdom',
    'Soarin': 'EPCOT',
    'Pirates of the Caribbean': 'Magic Kingdom',
}

dfs = []
for ride_name, filename in ride_files.items():
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df['ride_name'] = ride_name
    df['park'] = ride_parks[ride_name]
    dfs.append(df)
    print(f"  {ride_name}: {len(df):,} rows")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df):,} rows across {len(ride_files)} rides")

# =====================================================
# STEP 2: Clean and parse
# =====================================================
print("\n" + "=" * 60)
print("STEP 2: CLEANING DATA")
print("=" * 60)

# Parse dates
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

# Rename columns for clarity
df = df.rename(columns={'SPOSTMIN': 'posted_wait', 'SACTMIN': 'actual_wait'})

print(f"Before cleaning: {len(df):,} rows")

# Remove rows with no posted wait time or invalid values
df = df.dropna(subset=['posted_wait'])
df = df[df['posted_wait'] > 0]
df = df[df['posted_wait'] <= 300]

print(f"After cleaning: {len(df):,} rows")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Years covered: {df['date'].dt.year.nunique()}")

# =====================================================
# STEP 3: Feature engineering
# =====================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name()
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

# Season
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'
df['season'] = df['month'].apply(get_season)

# Holiday periods
def get_holiday(date):
    m, d = date.month, date.day
    if (m == 12 and d >= 18) or (m == 1 and d <= 3): return 'Christmas/New Years'
    if (m == 3 and d >= 10) or (m == 4 and d <= 20): return 'Spring Break'
    if (m == 6 and d >= 15) or m == 7 or (m == 8 and d <= 15): return 'Summer Peak'
    if m == 11 and d >= 20: return 'Thanksgiving'
    return 'Regular'
df['holiday_period'] = df['date'].apply(get_holiday)
df['is_holiday'] = df['holiday_period'] != 'Regular'

print(f"Features added. Columns: {list(df.columns)}")
print(f"\nSeason distribution:\n{df['season'].value_counts().to_string()}")
print(f"\nHoliday distribution:\n{df['holiday_period'].value_counts().to_string()}")

# =====================================================
# STEP 4: ANALYSIS & VISUALIZATIONS
# =====================================================
print("\n" + "=" * 60)
print("STEP 4: ANALYSIS")
print("=" * 60)

# --- 4a: Overall stats by ride ---
print("\n--- Top Rides by Average Posted Wait ---")
ride_stats = df.groupby('ride_name')['posted_wait'].agg(['mean', 'median', 'std', 'count'])
ride_stats = ride_stats.sort_values('mean', ascending=False)
ride_stats.columns = ['Avg Wait', 'Median Wait', 'Std Dev', 'Data Points']
print(ride_stats.round(1).to_string())

# --- 4b: Hollywood Studios vs Other Parks ---
print("\n--- Average Wait by Park ---")
park_stats = df.groupby('park')['posted_wait'].agg(['mean', 'median']).sort_values('mean', ascending=False)
print(park_stats.round(1).to_string())

# --- 4c: Day of Week ---
print("\n--- Average Wait by Day of Week ---")
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_stats = df.groupby('day_of_week')['posted_wait'].mean().reindex(day_order)
print(day_stats.round(1).to_string())
best_day = day_stats.idxmin()
worst_day = day_stats.idxmax()
print(f"\nBest day to visit: {best_day} ({day_stats[best_day]:.0f} min avg)")
print(f"Worst day to visit: {worst_day} ({day_stats[worst_day]:.0f} min avg)")

# --- 4d: Holiday Impact ---
print("\n--- Holiday vs Regular Wait Times ---")
holiday_stats = df.groupby('holiday_period')['posted_wait'].mean().sort_values(ascending=False)
print(holiday_stats.round(1).to_string())
regular_avg = holiday_stats['Regular']
for period, avg in holiday_stats.items():
    if period != 'Regular':
        pct = ((avg - regular_avg) / regular_avg) * 100
        print(f"  {period}: {pct:+.0f}% vs Regular")

# --- 4e: Seasonal ---
print("\n--- Average Wait by Season ---")
season_order = ['Winter','Spring','Summer','Fall']
season_stats = df.groupby('season')['posted_wait'].mean().reindex(season_order)
print(season_stats.round(1).to_string())

# --- 4f: Hour of day ---
print("\n--- Average Wait by Hour ---")
hour_stats = df.groupby('hour')['posted_wait'].mean()
print(hour_stats.round(1).to_string())
peak_hour = hour_stats.idxmax()
quiet_hour = hour_stats[hour_stats.index >= 9].idxmin()  # after park opens
print(f"\nPeak hour: {peak_hour}:00 ({hour_stats[peak_hour]:.0f} min avg)")
print(f"Best hour (after 9am): {quiet_hour}:00 ({hour_stats[quiet_hour]:.0f} min avg)")

# =====================================================
# STEP 5: GENERATE CHARTS
# =====================================================
print("\n" + "=" * 60)
print("STEP 5: GENERATING CHARTS")
print("=" * 60)

# --- Chart 1: Average Wait by Ride ---
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#1a5276' if p == 'Hollywood Studios' else '#7f8c8d' for p in [ride_parks[r] for r in ride_stats.index]]
bars = ax.barh(ride_stats.index, ride_stats['Avg Wait'], color=colors)
ax.set_xlabel('Average Posted Wait Time (minutes)')
ax.set_title('Average Wait Time by Attraction (2015–2023)')
ax.invert_yaxis()
for bar, val in zip(bars, ride_stats['Avg Wait']):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f} min', va='center', fontsize=9)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color='#1a5276', label='Hollywood Studios'),
    plt.Rectangle((0,0),1,1, color='#7f8c8d', label='Other Parks')
], loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_avg_wait_by_ride.png', bbox_inches='tight')
plt.close()
print("  Saved: 01_avg_wait_by_ride.png")

# --- Chart 2: Day of Week ---
fig, ax = plt.subplots(figsize=(8, 4))
colors_dow = ['#e74c3c' if d == worst_day else '#27ae60' if d == best_day else '#3498db' for d in day_order]
ax.bar(day_order, day_stats.values, color=colors_dow)
ax.set_ylabel('Average Posted Wait (min)')
ax.set_title('Best & Worst Days to Visit Walt Disney World')
ax.axhline(y=day_stats.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Overall avg: {day_stats.mean():.0f} min')
ax.legend()
for i, v in enumerate(day_stats.values):
    ax.text(i, v + 0.5, f'{v:.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_day_of_week.png', bbox_inches='tight')
plt.close()
print("  Saved: 02_day_of_week.png")

# --- Chart 3: Holiday Impact ---
fig, ax = plt.subplots(figsize=(8, 4))
holiday_sorted = holiday_stats.sort_values(ascending=True)
colors_h = ['#27ae60' if p == 'Regular' else '#e74c3c' if p == 'Christmas/New Years' else '#f39c12' for p in holiday_sorted.index]
bars = ax.barh(holiday_sorted.index, holiday_sorted.values, color=colors_h)
ax.set_xlabel('Average Posted Wait Time (minutes)')
ax.set_title('Holiday Periods Drive Significantly Longer Wait Times')
ax.axvline(x=regular_avg, color='gray', linestyle='--', alpha=0.7)
for bar, val in zip(bars, holiday_sorted.values):
    pct = ((val - regular_avg) / regular_avg) * 100
    label = f'{val:.0f} min' if val == regular_avg else f'{val:.0f} min ({pct:+.0f}%)'
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_holiday_impact.png', bbox_inches='tight')
plt.close()
print("  Saved: 03_holiday_impact.png")

# --- Chart 4: Hourly Heatmap by Ride ---
fig, ax = plt.subplots(figsize=(12, 6))
hourly_pivot = df.pivot_table(values='posted_wait', index='ride_name', columns='hour', aggfunc='mean')
hourly_pivot = hourly_pivot.loc[:, 8:22]  # Park hours only
hourly_pivot = hourly_pivot.reindex(ride_stats.index)  # Sort by avg wait
sns.heatmap(hourly_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Avg Wait (min)'})
ax.set_title('Wait Times by Hour — When to Ride Each Attraction')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_hourly_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: 04_hourly_heatmap.png")

# --- Chart 5: Seasonal Box Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
season_colors = {'Winter': '#4A90D9', 'Spring': '#7BC67E', 'Summer': '#F5A623', 'Fall': '#D0763C'}
season_data = [df[df['season'] == s]['posted_wait'] for s in season_order]
bp = ax.boxplot(season_data, labels=season_order, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=2))
for patch, season in zip(bp['boxes'], season_order):
    patch.set_facecolor(season_colors[season])
ax.set_ylabel('Posted Wait Time (minutes)')
ax.set_title('Wait Time Distribution by Season')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_seasonal_boxplot.png', bbox_inches='tight')
plt.close()
print("  Saved: 05_seasonal_boxplot.png")

# --- Chart 6: Hollywood Studios Deep Dive ---
hs_df = df[df['park'] == 'Hollywood Studios']
fig, ax = plt.subplots(figsize=(10, 5))
for ride in ['Slinky Dog Dash', 'Toy Story Mania', 'Rock n Roller Coaster', 'Alien Swirling Saucers']:
    ride_hourly = hs_df[hs_df['ride_name'] == ride].groupby('hour')['posted_wait'].mean()
    ride_hourly = ride_hourly[(ride_hourly.index >= 8) & (ride_hourly.index <= 21)]
    ax.plot(ride_hourly.index, ride_hourly.values, marker='o', linewidth=2, label=ride)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Posted Wait (min)')
ax.set_title('Hollywood Studios — Hourly Wait Patterns by Ride')
ax.legend(loc='upper right')
ax.set_xticks(range(8, 22))
ax.set_xticklabels([f'{h}:00' for h in range(8, 22)], rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_hollywood_studios_hourly.png', bbox_inches='tight')
plt.close()
print("  Saved: 06_hollywood_studios_hourly.png")

# --- Chart 7: Year-over-Year Monthly Heatmap ---
fig, ax = plt.subplots(figsize=(12, 5))
monthly = df.groupby(['year', 'month'])['posted_wait'].mean().reset_index()
pivot = monthly.pivot(index='year', columns='month', values='posted_wait')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Avg Wait (min)'})
ax.set_title('Average Wait Time by Month & Year — Crowd Trends Over Time')
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_monthly_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: 07_monthly_heatmap.png")

# --- Chart 8: Ride Volatility Scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
vol = df.groupby('ride_name')['posted_wait'].agg(['mean', 'std']).reset_index()
vol['park'] = vol['ride_name'].map(ride_parks)
colors_vol = ['#1a5276' if p == 'Hollywood Studios' else '#e74c3c' for p in vol['park']]
ax.scatter(vol['mean'], vol['std'], c=colors_vol, s=120, zorder=5, edgecolors='white', linewidth=1)
for _, row in vol.iterrows():
    ax.annotate(row['ride_name'], (row['mean'] + 1, row['std'] + 0.5), fontsize=8)
ax.set_xlabel('Average Wait (min)')
ax.set_ylabel('Standard Deviation (min)')
ax.set_title('Ride Popularity vs Unpredictability')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_ride_volatility.png', bbox_inches='tight')
plt.close()
print("  Saved: 08_ride_volatility.png")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)
print(f"""
DATASET: {len(df):,} wait time records across {len(ride_files)} attractions at 4 WDW parks
DATE RANGE: {df['date'].min().date()} to {df['date'].max().date()}

1. MOST POPULAR RIDE: {ride_stats.index[0]} — {ride_stats.iloc[0]['Avg Wait']:.0f} min average wait
2. BEST DAY TO VISIT: {best_day} — {day_stats[best_day]:.0f} min average
3. WORST DAY TO VISIT: {worst_day} — {day_stats[worst_day]:.0f} min average
4. CHRISTMAS IMPACT: {((holiday_stats.get('Christmas/New Years', 0) - regular_avg) / regular_avg * 100):+.0f}% longer waits vs regular periods
5. SPRING BREAK IMPACT: {((holiday_stats.get('Spring Break', 0) - regular_avg) / regular_avg * 100):+.0f}% longer waits vs regular periods
6. PEAK HOUR: {peak_hour}:00 — {hour_stats[peak_hour]:.0f} min average
7. BEST HOUR: {quiet_hour}:00 — {hour_stats[quiet_hour]:.0f} min average
8. BUSIEST SEASON: {season_stats.idxmax()} — {season_stats.max():.0f} min average
9. QUIETEST SEASON: {season_stats.idxmin()} — {season_stats.min():.0f} min average
""")
