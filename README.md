# Walt Disney World Wait Time & Demand Analysis

An analysis of **1.75 million+ historical wait time records** across 8 major attractions at 4 Walt Disney World parks (2015–2021), identifying patterns in crowd behavior, peak demand periods, and ride popularity trends.

## Live Dashboard
[View the interactive dashboard →](https://wdw-wait-time-analysis.vercel.app)

## Key Findings

### Holiday Impact
Holiday periods significantly increase wait times across all parks. Christmas and New Year's week drives the largest spike at **+43%** above regular periods, followed by Thanksgiving (+25%), Spring Break (+22%), and Summer Peak (+12%).

![Holiday Impact](charts/03_holiday_impact.png)

### Best & Worst Days to Visit
**Wednesday** is the best day to visit with an average 56-minute wait, while **Saturday** is the worst at 62 minutes. Midweek days (Tuesday–Thursday) consistently outperform weekends.

![Day of Week](charts/02_day_of_week.png)

### Hourly Patterns — When to Ride
Peak congestion hits at **11:00 AM** across all parks. Wait times drop significantly after 7:00 PM. At Hollywood Studios, Slinky Dog Dash maintains 80+ minute waits from 9 AM to 3 PM, while Alien Swirling Saucers stays under 40 minutes even at peak.

![Hollywood Studios Hourly](charts/06_hollywood_studios_hourly.png)

### Ride Popularity Rankings
Flight of Passage leads all rides with a 115-minute average wait — nearly double the next closest ride (Seven Dwarfs Mine Train at 77 min). Hollywood Studios rides cluster between 30–73 minutes.

![Average Wait by Ride](charts/01_avg_wait_by_ride.png)

### Seasonal & Year-over-Year Trends
Winter is the busiest season (65 min avg) driven by holiday crowds, while Fall is the quietest (51 min avg). The monthly heatmap reveals year-over-year crowd growth from 2015–2019, a COVID-driven drop in 2020, and a recovery pattern in 2021.

![Monthly Heatmap](charts/07_monthly_heatmap.png)

## Dataset

- **Source:** [Touring Plans](https://touringplans.com/) via [LucyMcGowan/touringplans](https://github.com/LucyMcGowan/touringplans)
- **Records:** 1,754,414 cleaned wait time observations
- **Date Range:** January 2015 – December 2021
- **Attractions:** 8 rides across 4 parks

| Attraction | Park | Avg Wait |
|---|---|---|
| Flight of Passage | Animal Kingdom | 115 min |
| Seven Dwarfs Mine Train | Magic Kingdom | 77 min |
| Slinky Dog Dash | Hollywood Studios | 73 min |
| Rock 'n' Roller Coaster | Hollywood Studios | 59 min |
| Toy Story Mania | Hollywood Studios | 54 min |
| Soarin' | EPCOT | 46 min |
| Alien Swirling Saucers | Hollywood Studios | 30 min |
| Pirates of the Caribbean | Magic Kingdom | 29 min |

## Tech Stack

- **Data Processing:** Python, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** React, Next.js, Tailwind CSS
- **Deployment:** Vercel

## Project Structure
```
wdw-wait-time-analysis/
├── charts/              # Generated visualization PNGs
├── data/                # Raw CSV data (not in repo — see below)
├── src/
│   └── analysis.py      # Full analysis pipeline
├── dashboard/           # React dashboard (deployed to Vercel)
├── .gitignore
└── README.md
```

## Running the Analysis

### 1. Get the Data
Download CSV files from [LucyMcGowan/touringplans](https://github.com/LucyMcGowan/touringplans/tree/main/data-raw) and place them in the `data/` folder.

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn
```

### 3. Run
```bash
cd src
python analysis.py
```

## Author
**Johnny Nguyen** — Data Science & Computer Science @ UCF  
[LinkedIn](https://linkedin.com/in/johnnynguyen04)
