# 🎬 Netflix Movies & TV Shows — Exploratory Data Analysis

A deep-dive EDA on Netflix's content catalogue (8,807 titles) using an **un-nested / exploded
dataframe approach** — cast, director, genre, and country (each a comma-separated list in the
raw data) are expanded into one long analysis-ready table so every chart can slice by any of
these dimensions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-2.x-150458)
![status](https://img.shields.io/badge/status-complete-brightgreen)

## 📌 Project Overview

Netflix lists more than 8,800 movies and TV shows, each with cast, director, country, genre,
rating, release year and duration. This project cleans the data, un-nests every multi-value
column into a single long dataframe, and analyses content mix, ratings, top talent, top
countries, and the best time to launch new titles — with full business insights and
recommendations at the end.

## 🗂️ Repository Structure

```
netflix-eda-project/
├── data/
│   └── netflix_titles.csv          # Raw dataset
├── notebooks/
│   └── full_analysis.py            # Full analysis script (generates every chart below)
├── images/
│   └── *.png                       # All 21 exported charts
├── requirements.txt
└── README.md
```

## 🛠️ Tools & Libraries

Python · pandas · NumPy · Matplotlib · Seaborn

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/netflix-eda-project.git
cd netflix-eda-project
pip install -r requirements.txt
cd notebooks
python full_analysis.py     # regenerates every chart into ../images/
```

---

## 1. Load the Data

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('netflix_titles.csv')
print(df.shape)          # (8807, 12)
df.head()
```

## 2. Initial Exploration

```python
df.describe()
df.info()
df.nunique()
(df.isnull().sum() / len(df) * 100)
```
**Result:** ~30% of `director`, ~9% of `cast`, ~9% of `country` are missing.

## 3. Un-nest Multi-Value Columns

`director`, `cast`, `listed_in` (genre), and `country` each hold comma-separated lists.
Each is split and "stacked" into its own long dataframe — one row per value.

```python
# Directors
dir_constraint = df['director'].apply(lambda x: str(x).split(', ')).tolist()
df1 = pd.DataFrame(dir_constraint, index=df['title'])
df1 = df1.stack(dropna=True).reset_index().drop(['level_1'], axis=1)
df1.rename(columns={0: 'Directors'}, inplace=True)

# Cast
cast_constraint = df['cast'].apply(lambda x: str(x).split(', ')).tolist()
df2 = pd.DataFrame(cast_constraint, index=df['title'])
df2 = df2.stack(dropna=True).reset_index().drop(['level_1'], axis=1)
df2.rename(columns={0: 'Actors'}, inplace=True)

# Genre
listed_constraint = df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df3 = pd.DataFrame(listed_constraint, index=df['title'])
df3 = df3.stack(dropna=True).reset_index().drop(['level_1'], axis=1)
df3.rename(columns={0: 'Genre'}, inplace=True)

# Country
country_constraint = df['country'].apply(lambda x: str(x).split(', ')).tolist()
df4 = pd.DataFrame(country_constraint, index=df['title'])
df4 = df4.stack(dropna=True).reset_index().drop(['level_1'], axis=1)
df4.rename(columns={0: 'Country'}, inplace=True)
```
> **Note on pandas versions:** `.stack()` must be called with `dropna=True` explicitly on
> recent pandas releases (2.1+), otherwise it keeps every padded `NaN` cell instead of
> dropping them — this silently explodes the row count by 10-20x and can crash on memory.

## 4. Merge Into One Long Dataframe

```python
df5 = df2.merge(df1, on=['title'], how='inner')
df6 = df5.merge(df3, on=['title'], how='inner')
df7 = df6.merge(df4, on=['title'], how='inner')
print(df7.shape)   # (201991, 5) — every actor × director × genre × country combination

df = df7.merge(
    df[['show_id', 'type', 'title', 'date_added', 'release_year', 'rating', 'duration']],
    on=['title'], how='left'
)
print(df.shape)    # (201991, 11)
```
**Why ~200k rows:** if a title has 5 actors, 2 genres and 1 country, it now appears
5 × 2 × 1 = 10 times — once per combination. This is expected; always use
`.agg({'title': 'nunique'})` rather than row counts when summarising.

## 5. Missing Value Treatment

```python
df['Actors'].replace(['nan'], ['Unknown Actor'], inplace=True)
df['Directors'].replace(['nan'], ['Unknown Director'], inplace=True)
df['Country'].replace(['nan'], [np.nan], inplace=True)

# Fix a real data-entry bug: a few rows have rating/duration swapped (e.g. rating = "74 min")
df.loc[df['duration'].isnull(), 'duration'] = df.loc[df['duration'].isnull(), 'rating']
df.loc[df['rating'].str.contains('min', na=False), 'rating'] = 'NR'
df['rating'].fillna('NR', inplace=True)

# Fill date_added using the most common date for that release_year
for i in df[df['date_added'].isnull()]['release_year'].unique():
    date = df[df['release_year'] == i]['date_added'].mode().values[0]
    df.loc[df['release_year'] == i, 'date_added'] = df.loc[df['release_year'] == i, 'date_added'].fillna(date)

# Fill Country using the director's usual country, then the actor's usual country
for i in df[df['Country'].isnull()]['Directors'].unique():
    if i in df[~df['Country'].isnull()]['Directors'].unique():
        country = df[df['Directors'] == i]['Country'].mode().values[0]
        df.loc[df['Directors'] == i, 'Country'] = df.loc[df['Directors'] == i, 'Country'].fillna(country)

for i in df[df['Country'].isnull()]['Actors'].unique():
    if i in df[~df['Country'].isnull()]['Actors'].unique():
        imp = df[df['Actors'] == i]['Country'].mode().values[0]
        df.loc[df['Actors'] == i, 'Country'] = df.loc[df['Actors'] == i, 'Country'].fillna(imp)

df['Country'].fillna('Unknown Country', inplace=True)
```

## 6. Date Parsing & Duration Cleanup

```python
df["date_added"] = pd.to_datetime(df['date_added'].str.strip(), format='mixed')
df['duration'] = df['duration'].str.replace(" min", "")

# Numeric copy of duration (0 for TV shows, which are measured in Seasons instead)
df['duration2'] = df.duration.copy()
df_ = df.copy()
df_.loc[df_['duration2'].str.contains('Season'), 'duration2'] = '0'
df_['duration2'] = df_.duration2.astype('int')
df_.duration2.describe()
```

### Distribution of Duration
```python
plt.style.use('dark_background')
plt.figure(figsize=(10, 4))
sns.histplot(df_['duration2'], bins=40)
plt.title('Distribution of Duration (0 = TV Shows, measured in Seasons separately)')
plt.show()
```
![Duration Histogram](images/01_duration_histogram.png)

## 7. Bin Duration Into Categories

```python
bins = [-1, 1, 50, 80, 100, 120, 150, 200, 315]
labels = ['<1', '1-50', '50-80', '80-100', '100-120', '120-150', '150-200', '200-315']
df_['duration2'] = pd.cut(df_['duration2'], bins=bins, labels=labels)

df_.loc[~df_['duration'].str.contains('Season'), 'duration'] = df_.loc[~df_['duration'].str.contains('Season'), 'duration2'].astype(str)
df_.drop(['duration2'], axis=1, inplace=True)
```

## 8. Extract Date Parts & Clean Titles

```python
df_["year_added"] = df_['date_added'].dt.year.astype("int64")
df_["month_added"] = df_['date_added'].dt.month.astype("int64")
df_['month_name'] = df_['date_added'].dt.month_name()
df_["day_added"] = df_['date_added'].dt.day.astype("int64")
df_['Weekday_added'] = df_['date_added'].dt.day_name()

df_['title'] = df_['title'].str.replace(r"\(.*\)", "", regex=True)
```

---

## 📊 Univariate Analysis

### Top 15 Genres
```python
df_genre = df_.groupby(['Genre']).agg({"title": "nunique"}).reset_index().sort_values('title', ascending=False).head(15)
plt.figure(figsize=(15, 6))
sns.barplot(x="Genre", y='title', data=df_genre)
plt.xticks(rotation=60)
plt.title('Top 15 Genres')
plt.show()
```
![Top 15 Genres](images/02_top15_genres.png)
**Result:** International Movies, Dramas and Comedies are the most popular genres overall.

### Movies vs TV Shows
```python
df_pie = df_.groupby(['type']).agg({'title': 'nunique'}).reset_index()
colors = sns.color_palette('bright')[0:5]
plt.figure(figsize=(8, 6))
plt.pie(df_pie['title'], labels=df_pie['type'], colors=colors, autopct='%.0f%%')
plt.title('Percentage of Movies and TV Shows')
plt.show()
```
![Movies vs TV Shows](images/03_movies_vs_tv_pie.png)
**Result:** ~70% Movies, ~30% TV Shows (6,115 vs 2,676 unique titles).

### Top 10 Countries
```python
df_['Country'] = df_['Country'].str.replace(',', '')
df_country = df_.groupby(['Country']).agg({'title': 'nunique'}).reset_index().sort_values('title', ascending=False).head(10)
plt.figure(figsize=(15, 6))
sns.barplot(y="Country", x='title', data=df_country)
plt.title('Top 10 Countries for Content Creation')
plt.show()
```
![Top 10 Countries](images/04_top10_countries.png)
**Result:** United States (3,690), India (1,046) and United Kingdom (806) lead content production.

### Top 10 Rating Types
```python
df_rating = df_.groupby(['rating']).agg({'title': 'nunique'}).reset_index().sort_values('title', ascending=False).head(10)
plt.figure(figsize=(15, 6))
sns.barplot(y="rating", x='title', data=df_rating)
plt.title('Top 10 Rating Types')
plt.show()
```
![Top 10 Ratings](images/05_top10_ratings.png)
**Result:** Most content is rated for mature audiences (TV-MA, TV-14).

### Top 10 Duration Categories
```python
df_duration = df_.groupby(['duration']).agg({'title': 'nunique'}).reset_index().sort_values('title', ascending=False).head(10)
plt.figure(figsize=(15, 6))
sns.barplot(y="duration", x='title', data=df_duration)
plt.title('Top 10 Duration Categories')
plt.show()
```
![Top 10 Duration Categories](images/06_top10_duration_categories.png)
**Result:** Most movies run 80-100 minutes; most TV shows have just 1 season.

### Top 15 Actors
```python
df_actors = df_.groupby(['Actors']).agg({'title': 'nunique'}).reset_index().sort_values('title', ascending=False)
df_actors = df_actors[df_actors['Actors'] != 'Unknown Actor'].head(15)
plt.figure(figsize=(15, 6))
sns.barplot(y="Actors", x='title', data=df_actors)
plt.title('Top 15 Most Popular Actors')
plt.show()
```
![Top 15 Actors](images/07_top15_actors.png)
**Result:** Anupam Kher, Shah Rukh Khan, Julie Tejwani, Naseeruddin Shah and Takahiro Sakurai top the list.

### Top 15 Directors
```python
df_directors = df_.groupby(['Directors']).agg({'title': 'nunique'}).reset_index().sort_values('title', ascending=False)
df_directors = df_directors[df_directors['Directors'] != 'Unknown Director'].head(15)
plt.figure(figsize=(15, 6))
sns.barplot(y="Directors", x='title', data=df_directors)
plt.title('Top 15 Most Popular Directors')
plt.show()
```
![Top 15 Directors](images/08_top15_directors.png)
**Result:** Rajiv Chilaka, Jan Suter and Raúl Campos are the most represented directors.

---

## 📈 Time-Based Analysis

### Titles Added Per Year
```python
df_year = df_.groupby(['year_added']).agg({'title': 'nunique'}).reset_index()
plt.figure(figsize=(15, 6))
sns.lineplot(x="year_added", y='title', data=df_year, color='red')
plt.title('Movies / TV Shows Added Across the Years')
plt.show()
```
![Titles Per Year](images/09_titles_per_year.png)
**Result:** Content additions grew steadily from 2008, peaked around 2019, then declined
slightly in 2020-21 (likely pandemic-related production slowdowns).

### Movies vs TV Shows Added, by Year
```python
plt.figure(figsize=(15, 5))
sns.countplot(data=df_, x='year_added', hue='type', palette="Reds_r")
plt.title('Movies and TV Shows Added to Netflix by Year')
plt.show()
```
![Movies vs TV by Year](images/10_movies_tv_by_year.png)
**Result:** Movies have been added in greater volume than TV Shows every year.

### Titles Added Across Months, by Type
```python
month_order = ['January','February','March','April','May','June','July',
               'August','September','October','November','December']
df_month_type = df_.groupby(['month_name', 'type']).agg({'title': 'nunique'}).reset_index()
df_month_type['month_name'] = pd.Categorical(df_month_type['month_name'], categories=month_order, ordered=True)
df_month_type = df_month_type.sort_values('month_name')

plt.figure(figsize=(15, 6))
sns.lineplot(x="month_name", y='title', data=df_month_type, hue='type')
plt.xticks(rotation=60)
plt.title('Movies/TV Shows Added Across the Months')
plt.show()
```
![Titles by Month and Type](images/11_titles_by_month_type.png)
**Result:** For both Movies and TV Shows, July is the strongest launch month, followed by December.

### Titles Added Across Months (Overall)
```python
df_month = df_.groupby(['month_name']).agg({'title': 'nunique'}).reset_index()
df_month['month_name'] = pd.Categorical(df_month['month_name'], categories=month_order, ordered=True)
df_month = df_month.sort_values('month_name')

plt.figure(figsize=(15, 6))
sns.lineplot(x="month_name", y='title', data=df_month, color='red')
plt.xticks(rotation=60)
plt.title('Movies/TV Shows Added Across the Months (Overall)')
plt.show()
```
![Titles by Month Overall](images/12_titles_by_month_overall.png)
**Result:** Most content overall gets added in July and December.

### Titles Added by Day of Month
```python
df_day = df_.groupby(['day_added']).agg({'title': 'nunique'}).reset_index()
plt.figure(figsize=(15, 6))
sns.barplot(x="day_added", y='title', data=df_day, color='red')
plt.title('Movies/TV Shows Added Across Each Day of the Month')
plt.show()
```
![Titles by Day of Month](images/13_titles_by_day_of_month.png)
**Result:** The 1st of the month sees by far the most additions (2,219 titles).

### Titles Added Across Weekdays, by Type
```python
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df_weekday_type = df_.groupby(['Weekday_added', 'type']).agg({'title': 'nunique'}).reset_index()
df_weekday_type['Weekday_added'] = pd.Categorical(df_weekday_type['Weekday_added'], categories=weekday_order, ordered=True)
df_weekday_type = df_weekday_type.sort_values('Weekday_added')

plt.figure(figsize=(15, 6))
sns.lineplot(x="Weekday_added", y='title', data=df_weekday_type, hue='type')
plt.xticks(rotation=60)
plt.title('Movies/TV Shows Added Across Weekdays')
plt.show()
```
![Titles by Weekday](images/14_titles_by_weekday_type.png)
**Result:** Friday is the best release day for both Movies and TV Shows, followed by Thursday.

---

## 📦 Bivariate Analysis

### Release Year by Content Type
```python
plt.figure(figsize=(15, 6))
sns.boxplot(x='type', y='release_year', data=df_)
sns.despine(left=True)
plt.title('Type of Show by Release Year')
plt.ylim(2000, 2021)
plt.show()
```
![Release Year Boxplot](images/15_boxplot_releaseyear_by_type.png)
**Result:** TV Shows skew to more recent release years than Movies — Netflix carries an older
film back-catalogue but mostly recent TV series.

### Heatmap — Titles Added, Month vs Year
```python
content = df_.groupby('year_added')['month_name'].value_counts().unstack().fillna(0)
plt.figure(figsize=(10, 8))
plt.title("Number of Titles Added, Month vs Year")
sns.heatmap(content, cmap='Blues')
plt.show()
```
![Heatmap Month vs Year](images/16_heatmap_month_year.png)
**Result:** The most content was added in November 2019 and July 2021; very little was added
before 2015.

### Release Year Scatter — Movies vs TV Shows
```python
plt.figure(figsize=(12, 5))
sns.scatterplot(y=df_.index, x=df_.release_year, hue=df_.type, palette='Set2')
plt.title('Release Year Scatter — Movies vs TV Shows')
plt.show()
```
![Release Year Scatter](images/17_scatter_releaseyear_type.png)

---

## 🔀 Split Analysis — Movies vs TV Shows Separately

```python
df_shows = df_[df_['type'] == 'TV Show']
df_movies = df_[df_['type'] == 'Movie']
```

### Top Genres — TV Shows
![Top Genres TV Shows](images/18_top15_genres_tvshows.png)
**Result:** International TV Shows, TV Dramas and TV Comedies are most popular for TV Shows.

### Top Genres — Movies
![Top Genres Movies](images/19_top15_genres_movies.png)
**Result:** International Movies, Dramas and Comedies lead for Movies, followed by Documentaries.

### Top Countries — TV Shows
![Top Countries TV Shows](images/20_top10_countries_tvshows.png)
**Result:** United States leads, followed by United Kingdom, Japan and South Korea.

### Top Countries — Movies
![Top Countries Movies](images/21_top10_countries_movies.png)
**Result:** United States leads by a wide margin, with India as a strong second — the number
of movies produced in India outweighs the combined Movies + TV Shows count of the next
few countries.

---

## 💡 Business Insights

- Content additions grew steadily from 2008 through 2020, then began declining — plausibly
  tied to pandemic-related production slowdowns. Movies are added in greater volume than TV
  Shows every year.
- Most content is added in **December and July**; by day of month, the **1st** sees the most
  additions, and **Friday** is the best release day (followed by Thursday).
- **Anupam Kher, Shah Rukh Khan, Julie Tejwani, Naseeruddin Shah** and **Takahiro Sakurai**
  occupy the top spots for most-watched content by cast.
- **Rajiv Chilaka, Jan Suter** and **Raúl Campos** are the most popular directors across
  Netflix — Rajiv Chilaka in particular is tied to a high volume of movies.
- Netflix's catalogue focuses more on **Movies** than TV Shows overall — a consistent
  **70:30 ratio** across the platform.
- **International Movies, Dramas** and **Comedies** are the most popular genres, both overall
  and within Movies/TV Shows individually.
- **United States** leads content creation across both Movies and TV Shows; **India's** movie
  output alone outweighs the combined Movie + TV Show count of several other top countries.

## ✅ Recommendations

1. **Double down on Drama, Comedy, and International content** — the most popular genres
   across almost every country and both content types.
2. **Launch new titles around July 1st or August 1st** — the historically strongest release
   windows.
3. **Increase movie output for the Indian audience** — Indian content has been declining
   since 2018 despite India being a top-2 market.
4. **Factor in proven actor/director combinations** for each target country when
   commissioning new content — certain director-actor pairings are reliable performers.
5. **Target 80-120 minutes for movie runtime** — the range most represented in the catalogue
   and best aligned with audience viewing habits.

## 👤 Author

**Krishna Jinde** — Business Analyst | MSc Business Analytics, Swansea University
[LinkedIn](https://linkedin.com/in/krishna-jinde-70845285)

## 📄 Dataset Source

Netflix Movies and TV Shows dataset (Shivam Bansal, Kaggle) — publicly available listing of
Netflix titles as of mid-2021.
