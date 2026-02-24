"""
Netflix Content Analysis & Recommendation System Sections:
  1. Data Generation & Preprocessing
  2. Exploratory Data Analysis (EDA)
  3. Visualizations
  4. Recommendation System (Item-Based Collaborative Filtering)
  5. Predictive Modelling (Linear Regression + Gradient Boosting)
  6. TF-IDF Text Analysis
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

# Aesthetics 
NETFLIX_RED  = "#E50914"
NETFLIX_DARK = "#141414"
PALETTE      = ["#E50914", "#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c",
                "#d62728", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

plt.rcParams.update({
    "figure.facecolor": NETFLIX_DARK,
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
})

OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1.  DEMO DATA GENERATION

np.random.seed(42)

COUNTRIES   = ["United States", "India", "United Kingdom", "Canada",
                "France", "Japan", "Spain", "South Korea", "Mexico", "Australia"]
RATINGS     = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "PG", "TV-G", "G", "NR"]
GENRES      = ["Drama", "Comedy", "Action", "Documentary", "Thriller",
                "Romance", "Horror", "Sci-Fi", "Animation", "Crime"]
TYPES       = ["Movie", "TV Show"]
DIRECTORS   = [f"Director_{i}" for i in range(1, 50)]
ACTORS      = [f"Actor_{i}"    for i in range(1, 80)]

N = 6000   # total titles

def _sample(seq, size, replace=True, p=None):
    return np.random.choice(seq, size=size, replace=replace, p=p)

#  Netflix catalogue 
dates = pd.date_range("2010-01-01", "2023-12-31", periods=N)
netflix_df = pd.DataFrame({
    "show_id":      [f"s{i}" for i in range(1, N + 1)],
    "type":         _sample(TYPES,     N, p=[0.70, 0.30]),
    "title":        [f"Title_{i}"      for i in range(1, N + 1)],
    "director":     _sample(DIRECTORS, N),
    "cast":         [", ".join(_sample(ACTORS, np.random.randint(1, 6)))
                     for _ in range(N)],
    "country":      _sample(COUNTRIES, N),
    "date_added":   dates,
    "release_year": np.random.randint(2000, 2024, N),
    "rating":       _sample(RATINGS,   N),
    "duration":     [f"{np.random.randint(60, 180)} min"
                     if t == "Movie" else f"{np.random.randint(1, 8)} Seasons"
                     for t in _sample(TYPES, N, p=[0.70, 0.30])],
    "listed_in":    [", ".join(np.random.choice(GENRES, np.random.randint(1, 4), replace=False))
                     for _ in range(N)],
    "description":  [f"A {g} story about drama and suspense."
                     for g in _sample(GENRES, N)],
})

# ── IMDB supplement 
imdb_df = netflix_df[["title"]].copy()
imdb_df["weighted_average_vote"] = np.clip(
    np.random.normal(6.5, 1.2, N), 1, 10).round(1)
imdb_df["genre"]  = _sample(GENRES, N)
imdb_df["reviews_from_users"] = np.random.randint(50, 5000, N)
imdb_df["year"]   = np.random.randint(2000, 2024, N)

joint_df = netflix_df.merge(imdb_df, on="title", how="inner")

print(f"✅ Dataset created — Netflix: {len(netflix_df)} titles | Joint: {len(joint_df)} titles")
print(f"   Columns: {list(netflix_df.columns)}\n")



# 2.  PREPROCESSING


def fill_mode(series: pd.Series) -> pd.Series:
    """Fill NaN values with the mode."""
    mode_val = series.mode()[0]
    return series.fillna(mode_val)

netflix_df["rating"] = fill_mode(netflix_df["rating"])
netflix_df.drop_duplicates(subset=["title", "country", "type", "release_year"],
                            inplace=True)

print(f"After dedup: {len(netflix_df)} rows")
print(f"Missing values per column:\n{netflix_df.isnull().sum()}\n")


# 3.  VISUALISATIONS

def save_fig(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=NETFLIX_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  💾 Saved → {path}")


#  3.1  Content type distribution (Pie) 
type_counts = netflix_df["type"].value_counts()

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    type_counts,
    labels=type_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=[NETFLIX_RED, "#1f77b4"],
    wedgeprops=dict(width=0.6, edgecolor=NETFLIX_DARK, linewidth=3),
    textprops=dict(color="white", fontsize=13),
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight("bold")
ax.set_title("Netflix Content by Type", fontsize=16, fontweight="bold",
             color="white", pad=20)
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "01_content_type_pie.png")


#  3.2  Top 10 countries (Stacked bar) 
country_type = (netflix_df.groupby(["country", "type"])
                           .size()
                           .reset_index(name="count"))
pivot = country_type.pivot(index="country", columns="type", values="count").fillna(0)
pivot["total"] = pivot.sum(axis=1)
top10 = pivot.nlargest(10, "total").drop(columns="total")

fig, ax = plt.subplots(figsize=(12, 6))
top10.plot(kind="bar", ax=ax, color=[NETFLIX_RED, "#1f77b4"],
           edgecolor="none", width=0.7)
ax.set_title("Top 10 Countries — Netflix Content Distribution",
             fontsize=15, fontweight="bold", color="white")
ax.set_xlabel("Country", fontsize=12)
ax.set_ylabel("Number of Titles", fontsize=12)
ax.legend(title="Type", facecolor="#222", labelcolor="white",
          title_fontsize=11)
ax.set_xticklabels(top10.index, rotation=35, ha="right", fontsize=10)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "02_top_countries_stacked.png")


#  3.3  Content added over time (Cumulative line) 
time_df = (netflix_df.groupby(["date_added", "type"])
                      .size()
                      .reset_index(name="added_today"))
time_df = time_df.sort_values("date_added")
time_df["cumulative"] = time_df.groupby("type")["added_today"].cumsum()

total_time = (netflix_df.groupby("date_added")
                         .size()
                         .reset_index(name="added_today")
                         .assign(type="Total"))
total_time = total_time.sort_values("date_added")
total_time["cumulative"] = total_time["added_today"].cumsum()

fig, ax = plt.subplots(figsize=(13, 6))
palette = {"Movie": NETFLIX_RED, "TV Show": "#1f77b4", "Total": "#ff7f0e"}
for label, grp in pd.concat([time_df, total_time]).groupby("type"):
    ax.plot(grp["date_added"], grp["cumulative"],
            label=label, color=palette[label], linewidth=2)
ax.set_title("Cumulative Netflix Content Over Time",
             fontsize=15, fontweight="bold", color="white")
ax.set_xlabel("Date Added", fontsize=12)
ax.set_ylabel("Total Titles", fontsize=12)
ax.legend(facecolor="#222", labelcolor="white")
ax.grid(True)
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "03_content_over_time.png")


#  3.4  Content rating distribution 
rating_counts = netflix_df["rating"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(rating_counts.index, rating_counts.values,
              color=PALETTE[:len(rating_counts)], edgecolor="none", width=0.6)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=9, color="white")
ax.set_title("Netflix Content by Rating", fontsize=15, fontweight="bold", color="white")
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Count",  fontsize=12)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "04_rating_distribution.png")


#  3.5  Top 20 genres 
all_genres = []
for g in netflix_df["listed_in"]:
    all_genres.extend([x.strip() for x in g.split(",")])
genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(genre_counts.index, genre_counts.values,
              color="#2ca02c", edgecolor="none", width=0.7)
ax.set_title("Top 20 Genres on Netflix", fontsize=15, fontweight="bold", color="white")
ax.set_xlabel("Genre", fontsize=12)
ax.set_ylabel("Count",  fontsize=12)
ax.set_xticklabels(genre_counts.index, rotation=40, ha="right", fontsize=9)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "05_top_genres.png")


#  3.6  Movie duration box-plots by country 
movies = netflix_df[netflix_df["type"] == "Movie"].copy()
movies["duration_min"] = movies["duration"].str.extract(r"(\d+)").astype(float)

top_countries = (movies["country"].value_counts().head(10).index.tolist())
box_df = movies[movies["country"].isin(top_countries)]

fig, ax = plt.subplots(figsize=(14, 6))
box_data = [box_df[box_df["country"] == c]["duration_min"].dropna().values
            for c in top_countries]
bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(top_countries, rotation=30, ha="right", fontsize=10)
ax.set_title("Movie Duration Distribution by Country (min)",
             fontsize=15, fontweight="bold", color="white")
ax.set_xlabel("Country", fontsize=12)
ax.set_ylabel("Duration (minutes)", fontsize=12)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "06_duration_boxplots.png")


#  3.7  Genre correlation heatmap 
from itertools import combinations

def genre_cooccurrence(df, content_type):
    subset = df[df["type"] == content_type]
    genre_lists = [set(x.split(", ")) for x in subset["listed_in"]]
    all_g = sorted({g for s in genre_lists for g in s})
    matrix = pd.DataFrame(0, index=all_g, columns=all_g)
    for gl in genre_lists:
        for g1, g2 in combinations(sorted(gl), 2):
            matrix.loc[g1, g2] += 1
            matrix.loc[g2, g1] += 1
    return matrix

movie_corr = genre_cooccurrence(netflix_df, "Movie")

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(movie_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(movie_corr, ax=ax, mask=mask, cmap="RdYlGn",
            linewidths=0.3, linecolor="#111",
            cbar_kws={"shrink": 0.7, "label": "Co-occurrence Count"})
ax.set_title("Genre Co-occurrence — Movies", fontsize=15, fontweight="bold", color="white")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "07_genre_correlation_heatmap.png")


#  3.8  Top directors & actors 
dir_counts = netflix_df["director"].value_counts().head(10)
act_all = []
for row in netflix_df["cast"]:
    act_all.extend([x.strip() for x in row.split(",")])
act_counts = pd.Series(Counter(act_all)).sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, counts, label, color in zip(
        axes, [dir_counts, act_counts],
        ["Top 10 Directors", "Top 10 Actors (US)"],
        ["#b5421a", "#1f77b4"]):
    ax.barh(counts.index[::-1], counts.values[::-1],
            color=color, edgecolor="none")
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9, color="white")
    ax.set_title(label, fontsize=13, fontweight="bold", color="white")
    ax.set_xlabel("Number of Titles", fontsize=11)
    ax.grid(axis="x")

fig.suptitle("People Behind Netflix Content",
             fontsize=16, fontweight="bold", color=NETFLIX_RED, y=1.01)
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "08_top_directors_actors.png")


#  3.9  Simulated word-cloud (bar chart fallback) 
all_words = []
for desc in joint_df["description"]:
    all_words.extend(re.findall(r"\b[a-z]{4,}\b", desc.lower()))
stop = {"this", "that", "with", "from", "about", "story", "have", "been",
        "their", "they", "will", "when", "what", "into", "then", "more"}
word_freq = {w: c for w, c in Counter(all_words).most_common(30) if w not in stop}
wf = pd.Series(word_freq).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 5))
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(wf)))
bars = ax.bar(wf.index, wf.values, color=colors, edgecolor="none", width=0.7)
ax.set_title("Top 20 Words in Netflix Descriptions (TF-IDF Proxy)",
             fontsize=14, fontweight="bold", color="white")
ax.set_xlabel("Word", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_xticklabels(wf.index, rotation=40, ha="right", fontsize=9)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "09_word_frequency.png")


# 4.  RECOMMENDATION SYSTEM  (Item-Based CF via Cosine Similarity)

print("\n" + "="*60)
print("4. RECOMMENDATION SYSTEM (IBCF)")
print("="*60)

# Build a user × item rating matrix (sparse demo)
N_USERS = 200
N_ITEMS = 300
DENSITY = 0.08

user_ids   = [f"user_{i}"   for i in range(N_USERS)]
item_ids   = [f"s{i}"       for i in range(1, N_ITEMS + 1)]

rating_matrix = np.full((N_USERS, N_ITEMS), np.nan)
mask_indices  = np.random.rand(N_USERS, N_ITEMS) < DENSITY
rating_matrix[mask_indices] = np.random.uniform(1, 10,
                                                  mask_indices.sum()).round(1)

def ibcf_recommend(rating_mat, user_idx: int, top_n: int = 10):
    """Item-Based CF using cosine similarity on item vectors."""
    item_mat = np.nan_to_num(rating_mat.T)          # items × users
    sim      = cosine_similarity(item_mat)           # item × item
    user_row = rating_mat[user_idx]
    rated    = np.where(~np.isnan(user_row))[0]
    if len(rated) == 0:
        return []
    scores = sim[:, rated].mean(axis=1)
    scores[rated] = -np.inf                          # exclude already-rated
    top_items = np.argsort(scores)[::-1][:top_n]
    return top_items.tolist()

recs_user0 = ibcf_recommend(rating_matrix, user_idx=0, top_n=10)
print(f"\n🎬 Top-10 recommendations for User_0:")
for rank, item_idx in enumerate(recs_user0, 1):
    sid   = item_ids[item_idx]
    title = netflix_df[netflix_df["show_id"] == sid]["title"].values
    label = title[0] if len(title) else sid
    print(f"  {rank:>2}. {label}")

#  Similarity heatmap (first 20 items) 
item_mat_20 = np.nan_to_num(rating_matrix[:, :20].T)
sim_20 = cosine_similarity(item_mat_20)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_20, ax=ax, cmap="YlOrRd",
            xticklabels=[f"i{i}" for i in range(20)],
            yticklabels=[f"i{i}" for i in range(20)],
            linewidths=0.3, linecolor="#222",
            cbar_kws={"shrink": 0.75, "label": "Cosine Similarity"})
ax.set_title("Item-Item Cosine Similarity (First 20 Items)",
             fontsize=14, fontweight="bold", color="white")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "10_item_similarity_heatmap.png")


# 5.  PREDICTIVE MODELLING

print("\n" + "="*60)
print("5. PREDICTIVE MODELLING")
print("="*60)

model_df = joint_df[["weighted_average_vote", "rating", "listed_in",
                       "duration", "country", "reviews_from_users",
                       "type"]].copy().dropna()

# Encode categoricals
for col in ["rating", "listed_in", "country", "type"]:
    model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))

# Parse duration to numeric (movies)
model_df["duration_num"] = (model_df["duration"]
                             .str.extract(r"(\d+)").astype(float).squeeze())
model_df.dropna(subset=["duration_num"], inplace=True)

FEATURES = ["rating", "listed_in", "country", "duration_num",
            "reviews_from_users", "type"]
X = model_df[FEATURES]
y = model_df["weighted_average_vote"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#  5a. Linear Regression 
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"\n📈 Linear Regression  — RMSE: {rmse_lr:.4f}")

#  5b. Gradient Boosting 
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, random_state=42
).fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print(f"🚀 Gradient Boosting — RMSE: {rmse_gb:.4f}")

#  Feature importance plot 
importance = pd.Series(gb.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors = [NETFLIX_RED if i == importance.idxmax() else "#4a90d9"
          for i in importance.index]
ax.barh(importance.index, importance.values, color=colors, edgecolor="none")
ax.set_title("Gradient Boosting — Feature Importance",
             fontsize=14, fontweight="bold", color="white")
ax.set_xlabel("Importance Score", fontsize=11)
ax.grid(axis="x")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "11_feature_importance.png")

#  Actual vs Predicted scatter 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title, color in zip(
        axes,
        [y_pred_lr, y_pred_gb],
        [f"Linear Regression (RMSE={rmse_lr:.3f})",
         f"Gradient Boosting  (RMSE={rmse_gb:.3f})"],
        [NETFLIX_RED, "#1f77b4"]):
    ax.scatter(y_test, y_pred, alpha=0.4, color=color, s=15, edgecolors="none")
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], "w--", lw=1.5)
    ax.set_xlabel("Actual Rating",    fontsize=11)
    ax.set_ylabel("Predicted Rating", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.grid(True)

fig.suptitle("Actual vs Predicted IMDb Ratings",
             fontsize=15, fontweight="bold", color=NETFLIX_RED)
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "12_actual_vs_predicted.png")


# 6.  TF-IDF TEXT ANALYSIS

print("\n" + "="*60)
print("6. TF-IDF TEXT ANALYSIS")
print("="*60)

tfidf = TfidfVectorizer(stop_words="english", max_features=500, min_df=3)
tfidf_matrix = tfidf.fit_transform(joint_df["description"].fillna(""))
vocab = tfidf.get_feature_names_out()

mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
top_idx = np.argsort(mean_tfidf)[::-1][:20]
top_words = pd.Series(mean_tfidf[top_idx], index=vocab[top_idx])

print(f"\n📝 Top 10 TF-IDF terms across descriptions:")
print(top_words.head(10).to_string())

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(top_words.index, top_words.values,
       color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_words))),
       edgecolor="none", width=0.7)
ax.set_title("Top 20 TF-IDF Terms in Netflix Descriptions",
             fontsize=14, fontweight="bold", color="white")
ax.set_xlabel("Term", fontsize=11)
ax.set_ylabel("Mean TF-IDF Score", fontsize=11)
ax.set_xticklabels(top_words.index, rotation=40, ha="right", fontsize=9)
ax.grid(axis="y")
fig.patch.set_facecolor(NETFLIX_DARK)
save_fig(fig, "13_tfidf_terms.png")


#  Summary dashboard 
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(NETFLIX_DARK)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A — type pie
ax_pie = fig.add_subplot(gs[0, 0])
ax_pie.pie(type_counts, labels=type_counts.index,
           autopct="%1.0f%%", colors=[NETFLIX_RED, "#1f77b4"],
           wedgeprops=dict(width=0.55, edgecolor=NETFLIX_DARK),
           textprops=dict(color="white", fontsize=10))
ax_pie.set_title("Content Type", color="white", fontsize=11, fontweight="bold")

# Panel B — rating bar
ax_rat = fig.add_subplot(gs[0, 1])
ax_rat.bar(rating_counts.index, rating_counts.values,
           color=PALETTE[:len(rating_counts)], edgecolor="none")
ax_rat.set_title("Content Ratings", color="white", fontsize=11, fontweight="bold")
ax_rat.set_xticklabels(rating_counts.index, rotation=45, ha="right", fontsize=8)
ax_rat.grid(axis="y")

# Panel C — feature importance
ax_fi = fig.add_subplot(gs[0, 2])
ax_fi.barh(importance.index, importance.values, color="#4a90d9", edgecolor="none")
ax_fi.set_title("Feature Importance (GB)", color="white", fontsize=11, fontweight="bold")
ax_fi.grid(axis="x")

# Panel D — time series
ax_ts = fig.add_subplot(gs[1, :2])
for label, grp in time_df.groupby("type"):
    ax_ts.plot(grp["date_added"], grp["cumulative"],
               label=label, color=palette[label], linewidth=1.8)
ax_ts.set_title("Cumulative Titles Over Time", color="white", fontsize=11, fontweight="bold")
ax_ts.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax_ts.grid(True)

# Panel E — top genres
ax_gen = fig.add_subplot(gs[1, 2])
ax_gen.barh(genre_counts.index[:10][::-1], genre_counts.values[:10][::-1],
            color="#2ca02c", edgecolor="none")
ax_gen.set_title("Top 10 Genres", color="white", fontsize=11, fontweight="bold")
ax_gen.grid(axis="x")

fig.suptitle("Netflix Content Analysis — Dashboard",
             fontsize=18, fontweight="bold", color=NETFLIX_RED, y=1.01)
save_fig(fig, "00_dashboard.png")


# ── Final summary ───
print("\n" + "="*60)
print("✅  ANALYSIS COMPLETE")
print("="*60)
print(f"  Total titles analysed : {len(netflix_df):,}")
print(f"  Movies                : {(netflix_df['type']=='Movie').sum():,}")
print(f"  TV Shows              : {(netflix_df['type']=='TV Show').sum():,}")
print(f"  Countries represented : {netflix_df['country'].nunique()}")
print(f"  Unique genres         : {len(genre_counts)}")
print(f"  Linear Reg RMSE       : {rmse_lr:.4f}")
print(f"  Gradient Boost RMSE   : {rmse_gb:.4f}")
print(f"\n  Output images saved to: {OUTPUT_DIR}/")
