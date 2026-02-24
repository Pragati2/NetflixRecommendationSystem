# =============================================================================
# Netflix Content Analysis & Recommendation System
# =============================================================================
# Author  : Pragati Khekale
# Purpose : EDA, visualisation, IBCF recommendation, and rating prediction
#           for the Netflix catalogue enriched with IMDb metadata.
#
# Required datasets (place in working directory):
#   - netflix_titles.csv
#   - IMDb ratings.csv
#   - IMDb movies.csv
# =============================================================================


# ── 0. Dependencies ──────────────────────────────────────────────────────────
required_packages <- c(
  "tidyverse", "reshape2", "ggrepel", "plotly", "lubridate",
  "data.table", "recommenderlab", "tm", "SnowballC", "wordcloud",
  "RColorBrewer", "tidytext", "caret", "gbm"
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing)) install.packages(missing, dependencies = TRUE)
}
install_if_missing(required_packages)

suppressPackageStartupMessages({
  library(tidyverse)
  library(reshape2)
  library(ggrepel)
  library(plotly)
  library(lubridate)
  library(data.table)
  library(recommenderlab)
  library(tm)
  library(SnowballC)
  library(wordcloud)
  library(RColorBrewer)
  library(tidytext)
  library(caret)
  library(gbm)
})

options(warn = -1)


# ── 1. Load Data ─────────────────────────────────────────────────────────────
netflix_df <- read.csv(
  "netflix_titles.csv",
  na.strings     = c("NA", ""),
  stringsAsFactors = FALSE
)

ratings_imdb <- fread("IMDb ratings.csv", select = "weighted_average_vote")
titles_imdb  <- fread("IMDb movies.csv",  select = c("title", "year", "genre",
                                                       "reviews_from_users"))

imdb_df <- data.frame(ratings_imdb, titles_imdb) |>
  distinct(title, year, weighted_average_vote, genre, .keep_all = TRUE) |>
  drop_na(weighted_average_vote)

joint_df <- inner_join(netflix_df, imdb_df, by = "title")

message(sprintf("✅ Loaded %d Netflix titles | Joint df: %d rows",
                nrow(netflix_df), nrow(joint_df)))


# ── 2. Preprocessing ─────────────────────────────────────────────────────────
# Parse date
netflix_df$date_added <- as.Date(netflix_df$date_added, format = "%B %d, %Y")

# Fill missing rating with mode
get_mode <- function(v) {
  u <- unique(na.omit(v))
  u[which.max(tabulate(match(v, u)))]
}
netflix_df$rating[is.na(netflix_df$rating)] <- get_mode(netflix_df$rating)

# De-duplicate
netflix_df <- distinct(netflix_df, title, country, type, release_year,
                        .keep_all = TRUE)

# Missing value report
missing_report <- data.frame(
  variable       = colnames(netflix_df),
  missing_values = sapply(netflix_df, function(x) sum(is.na(x))),
  row.names      = NULL
)
print(missing_report)


# ── 3. Exploratory Visualisations ───────────────────────────────────────────

# 3.1 Content type distribution
type_count <- netflix_df |> count(type)

plot_ly(type_count, labels = ~type, values = ~n, type = "pie",
        marker = list(colors = c("#E50914", "#1f77b4"))) |>
  layout(title = "Netflix Content by Type")

# 3.2 Top countries (stacked bar)
country_type <- netflix_df |>
  mutate(country = strsplit(country, ", ")) |>
  unnest(country) |>
  mutate(country = trimws(gsub(",", "", country))) |>
  drop_na(country) |>
  count(country, type)

top10_countries <- country_type |>
  group_by(country) |>
  summarise(total = sum(n), .groups = "drop") |>
  slice_max(total, n = 10) |>
  pull(country)

country_wide <- country_type |>
  filter(country %in% top10_countries) |>
  pivot_wider(names_from = type, values_from = n, values_fill = 0) |>
  arrange(desc(Movie + `TV Show`))

plot_ly(country_wide, x = ~country, y = ~Movie,      type = "bar",
        name = "Movie",   marker = list(color = "#E50914")) |>
  add_trace(               y = ~`TV Show`, name = "TV Show",
            marker = list(color = "#1f77b4")) |>
  layout(
    barmode = "stack",
    title   = "Top 10 Countries — Netflix Content",
    xaxis   = list(title = "Country"),
    yaxis   = list(title = "Number of Titles")
  )

# 3.3 Content growth over time
by_date_type <- netflix_df |>
  count(date_added, type) |>
  drop_na(date_added) |>
  group_by(type) |>
  arrange(date_added) |>
  mutate(cumulative = cumsum(n))

by_date_total <- netflix_df |>
  count(date_added) |>
  drop_na(date_added) |>
  arrange(date_added) |>
  mutate(cumulative = cumsum(n), type = "Total")

time_df <- bind_rows(by_date_type, by_date_total)

plot_ly(time_df, x = ~date_added, y = ~cumulative, color = ~type,
        type = "scatter", mode = "lines",
        colors = c("#E50914", "#ff7f0e", "#1f77b4")) |>
  layout(
    title = "Cumulative Netflix Content Over Time",
    xaxis = list(title = "Date Added"),
    yaxis = list(title = "Total Titles")
  )

# 3.4 Rating distribution
plot_ly(count(netflix_df, rating), x = ~rating, y = ~n, type = "bar",
        marker = list(color = "#E50914")) |>
  layout(title = "Content Rating Distribution",
         xaxis = list(title = "Rating"),
         yaxis = list(title = "Count"))

# 3.5 Top 20 genres
top_genres <- netflix_df |>
  mutate(genre = strsplit(listed_in, ",")) |>
  unnest(genre) |>
  mutate(genre = trimws(genre)) |>
  count(genre, sort = TRUE) |>
  slice_max(n, n = 20)

plot_ly(top_genres, x = ~genre, y = ~n, type = "bar",
        marker = list(color = "palegreen")) |>
  layout(title   = "Top 20 Genres on Netflix",
         xaxis   = list(categoryorder = "array", categoryarray = top_genres$genre,
                        title = "Genre"),
         yaxis   = list(title = "Count"))

# 3.6 Movie duration by country (box plots)
movie_duration <- netflix_df |>
  filter(type == "Movie") |>
  select(country, duration) |>
  drop_na() |>
  mutate(
    country      = strsplit(country, ", "),
    duration_min = as.numeric(gsub(" min", "", duration))
  ) |>
  unnest(country) |>
  mutate(country = trimws(country)) |>
  filter(country %in% c("United States", "India", "United Kingdom", "Canada",
                         "France", "Japan", "Spain", "South Korea",
                         "Mexico", "Australia", "Taiwan"))

plot_ly(movie_duration, y = ~duration_min, color = ~country, type = "box") |>
  layout(title  = "Movie Duration by Country",
         xaxis  = list(title = "Country"),
         yaxis  = list(title = "Duration (min)"))

# 3.7 Genre co-occurrence heatmap
show_cats <- netflix_df |>
  select(show_id, type, listed_in) |>
  separate_rows(listed_in, sep = ",") |>
  mutate(listed_in = trimws(listed_in)) |>
  rename(category = listed_in)

genre_corr <- show_cats |>
  filter(type == "Movie") |>
  inner_join(show_cats |> filter(type == "Movie"),
             by = c("show_id", "type"),
             suffix = c("_1", "_2")) |>
  filter(as.character(category_1) < as.character(category_2)) |>
  count(category_1, category_2, name = "matched_count") |>
  filter(matched_count > 0)

ggplot(genre_corr, aes(category_1, category_2, fill = matched_count)) +
  geom_tile() +
  scale_fill_distiller(palette = "Spectral") +
  theme_dark() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Genre Co-occurrence Heatmap — Movies",
       x = NULL, y = NULL, fill = "Count")

# 3.8 Top directors & actors (US)
top_directors <- netflix_df |>
  mutate(director = strsplit(director, ", ")) |>
  unnest(director) |>
  mutate(director = trimws(gsub(",", "", director))) |>
  drop_na(director) |>
  count(director, sort = TRUE) |>
  slice_max(n, n = 10)

ggplot(top_directors, aes(x = fct_reorder(director, n), y = n)) +
  geom_col(fill = "#b5421a") +
  geom_text(aes(label = n), hjust = -0.2, color = "white") +
  coord_flip() +
  theme_dark() +
  labs(title = "Top 10 Directors on Netflix", x = NULL, y = "Titles")


# ── 4. Recommendation System (IBCF) ─────────────────────────────────────────
rating_data  <- subset(joint_df, select = c("reviews_from_users", "show_id",
                                             "weighted_average_vote"))
movie_data   <- subset(joint_df, select = c("title", "description", "show_id",
                                             "genre"))

rating_matrix <- dcast(rating_data, reviews_from_users ~ show_id,
                        value.var = "weighted_average_vote",
                        na.rm = FALSE, fun = mean)
rating_matrix <- as.matrix(rating_matrix[, -1])
rating_matrix <- as(rating_matrix, "realRatingMatrix")

# Filter to active users/items
active_ratings <- rating_matrix[rowCounts(rating_matrix) > 1,
                                 colCounts(rating_matrix) > 1]

# Train/test split (80/20)
set.seed(42)
train_idx    <- sample(c(TRUE, FALSE), nrow(active_ratings),
                        replace = TRUE, prob = c(0.8, 0.2))
train_data   <- active_ratings[train_idx, ]
test_data    <- active_ratings[!train_idx, ]

# Build IBCF model
ibcf_model <- Recommender(data   = train_data,
                           method = "IBCF",
                           parameter = list(k = 30))

# Generate top-10 recommendations for test users
n_recommend  <- 10
predictions  <- predict(object  = ibcf_model,
                         newdata = test_data,
                         n       = n_recommend)

# Extract titles for the first test user
resolve_titles <- function(item_ids, movie_df) {
  vapply(item_ids, function(id) {
    found <- movie_df$title[movie_df$show_id == id]
    if (length(found)) found[1] else id
  }, character(1))
}

user1_items  <- predictions@items[[1]]
user1_labels <- predictions@itemLabels[user1_items]
user1_titles <- resolve_titles(user1_labels, movie_data)
message("Top-10 Recommendations (User 1):")
print(user1_titles)

# Item similarity heatmap (first 20 items)
sim_matrix <- similarity(active_ratings[, 1:20],
                          method = "cosine", which = "items")
image(as.matrix(sim_matrix), main = "Item Cosine Similarity (first 20)")


# ── 5. Predictive Modelling ──────────────────────────────────────────────────
model_df <- joint_df |>
  select(weighted_average_vote, rating, listed_in, duration,
         country, reviews_from_users) |>
  drop_na() |>
  mutate(
    rating    = as.numeric(factor(rating)),
    listed_in = as.numeric(factor(listed_in)),
    country   = as.numeric(factor(country))
  )

# 80/20 partition
set.seed(42)
train_idx2 <- createDataPartition(model_df$weighted_average_vote,
                                   p = 0.8, list = FALSE)
training   <- model_df[train_idx2, ]
validation <- model_df[-train_idx2, ]

formula_rating <- weighted_average_vote ~ rating + listed_in +
                  reviews_from_users + country + duration

# 5a. Linear Regression
lm_model <- lm(formula_rating, data = training, drop.unused.levels = TRUE)
cat("\n── Linear Regression Summary ──\n")
print(summary(lm_model))

# 5b. Gradient Boosting
gb_model <- gbm(
  formula_rating,
  data             = training,
  distribution     = "gaussian",
  n.trees          = 5000,
  interaction.depth = 4
)
gb_preds <- predict(gb_model, newdata = validation, n.trees = 5000)
gb_sse   <- sum((validation$weighted_average_vote - gb_preds)^2, na.rm = TRUE)
message(sprintf("Gradient Boosting SSE: %.4f", gb_sse))
summary(gb_model)


# ── 6. TF-IDF Text Analysis ──────────────────────────────────────────────────
# Word corpus from IMDb genre descriptions
corpus <- Corpus(VectorSource(joint_df$genre)) |>
  tm_map(content_transformer(function(x, p) gsub(p, " ", x), p = "[/|@\\\\|]")) |>
  tm_map(content_transformer(tolower)) |>
  tm_map(removeNumbers) |>
  tm_map(removeWords, stopwords("english")) |>
  tm_map(removePunctuation) |>
  tm_map(stripWhitespace)

dtm   <- TermDocumentMatrix(corpus)
m_dtm <- as.matrix(dtm)
freq  <- sort(rowSums(m_dtm), decreasing = TRUE)
freq_df <- data.frame(word = names(freq), freq = freq)

set.seed(42)
wordcloud(
  words        = freq_df$word,
  freq         = freq_df$freq,
  min.freq     = 1,
  max.words    = 100,
  random.order = FALSE,
  rot.per      = 0.35,
  colors       = brewer.pal(8, "Dark2")
)

# TF-IDF per title
netflix_words <- joint_df |>
  unnest_tokens(word, description) |>
  count(title, word, sort = TRUE) |>
  left_join(
    joint_df |>
      unnest_tokens(word, description) |>
      count(title, word, sort = TRUE) |>
      group_by(title) |>
      summarise(total = sum(n), .groups = "drop"),
    by = "title"
  ) |>
  bind_tf_idf(word, title, n)

cat("\nTop TF-IDF terms overall:\n")
print(netflix_words |>
        select(-total) |>
        arrange(desc(tf_idf)) |>
        head(10))
