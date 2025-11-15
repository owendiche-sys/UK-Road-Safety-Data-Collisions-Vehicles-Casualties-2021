# UK Road Accident Analysis — 2021

## Project Overview

This project performs an exploratory data analysis (EDA) on UK road accidents in 2021. Using datasets on collisions, vehicles, and casualties, we analyze patterns, trends, and severity of accidents to gain insights into road safety.


The analysis includes:

- Temporal patterns (monthly and weekly trends)

- Accident severity distributions

- Vehicle types involved and casualty summaries

- Summary statistics and visualizations for decision-making and policy insights



---



## Datasets

The analysis uses the following CSV files from the UK Road Safety Data repository:



1\. **Collisions**: Information on each accident (date, time, location, severity, road conditions, etc.)

2\. **Vehicles**: Details about vehicles involved in each accident

3\. **Casualties**: Information on casualties resulting from accidents



All CSV files are stored in the `data/` folder.



---



## Methodology

1\. **Data Loading & Cleaning**

&nbsp;  - Read collisions, vehicles, and casualties datasets

&nbsp;  - Standardize column names and accident index identifiers

&nbsp;  - Convert date columns to datetime and remove invalid entries



2\. **Feature Engineering**

&nbsp;  - Extract Year, Month, Month Name, and Day of Week

&nbsp;  - Aggregate accidents by month and day of week

&nbsp;  - Compute additional metrics like average vehicles per accident and casualties per accident



3\. **Exploratory Data Analysis (EDA)**

&nbsp;  - Visualize monthly accident trends

&nbsp;  - Analyze accidents by day of the week

&nbsp;  - Examine severity distributions

&nbsp;  - Summarize vehicles and casualty types



4\. **Summary Statistics**

&nbsp;  - Export CSV files with overall summary, monthly summary, and day-of-week summary

&nbsp;  - Save visualizations for reporting



---



## Key Insights

- **Temporal Patterns**: Accidents occur more frequently on weekdays and certain months show higher accident counts.  

- **Severity Patterns**: Most accidents are slight; serious and fatal accidents are influenced by weather, road type, and urban/rural area.  

- **Vehicles & Casualties**: Single-vehicle accidents are most common, while multi-vehicle accidents tend to be more severe with more casualties.  

- **Practical Implications**: These insights can support traffic authorities and policymakers in targeting high-risk periods and conditions, and inform future predictive analyses for accident prevention.



---



## Visualizations

All visualizations are stored in the `images/` folder:

- `monthly\_accidents.png` — Monthly accident trends

- `accidents\_by\_dow.png` — Accidents by day of the week

- Additional charts for severity, vehicles, and casualties are included in the repository



---



## Outputs

Summary statistics CSVs are saved in the `outputs/` folder:

- `overall\_summary\_statistics.csv`

- `monthly\_accidents\_summary.csv`

\- `day\_of\_week\_summary.csv`


---



## Tools \& Libraries

- Python 3.x  

- pandas  

- numpy
    
- matplotlib  

- seaborn  



---

