# The Evolution of Music: 1921–2020

How has music changed over a century, and can a song's sound reveal when it was made?

I analyzed about 170,000 Spotify tracks to measure how audio features changed from the 1920s to the 2010s, built machine learning models that predict a song's decade from its sound alone, and clustered songs into six style groups to see how the mix of styles changed over time.

## Key findings

- **Music got far less acoustic and far more energetic, mostly between the 1950s and 1970s.** Average acousticness fell from 0.84 (1950s) to 0.40 (1970s) and 0.26 (2010s), while energy rose from 0.29 to 0.53 and then 0.63.
- **Songs got louder.** Average loudness rose from −16.7 dB (1920s) to −7.4 dB (2010s), a gain of about 9 dB. The biggest decade-to-decade jump came between the 1990s and 2000s.
- **Songs got longer, then shorter.** Average length grew from about 3.0 minutes (1920s) to about 4.2 minutes (1980s) and has fallen to 3.8 minutes in the 2010s (and about 3.3 minutes by 2020).
- **Mood (valence) has slipped recently.** It peaked in the 1970s (0.59) and was lowest in the 2010s (0.46).
- **A Random Forest predicts a song's decade with 45–46% exact accuracy** (about 10% by chance, 12% by always guessing the most common decade), and **about 77% of predictions are within one decade** of the true one.
- **Acousticness, loudness, song length and energy are the strongest signals of era.**
- **Six sound styles** (found with K-Means) show the shift clearly: a "loud, fast, high-energy" style grows from about 1% of songs before 1960 to roughly 40% by the 2000s and 2010s, while an "instrumental and soft" style shrinks from about 36% (1920s) to about 5% (2000s).

## Data

**Spotify Dataset 1921–2020, 160k+ Tracks** (Kaggle, uploaded by Yamac Eren Ay): 170,653 tracks with audio features (danceability, energy, valence, acousticness, instrumentalness, liveness, speechiness, loudness, tempo, key, mode, duration) and release year.

**Cleaning**
- No missing values and no duplicate rows were found.
- Removed 968 tracks (under 1%): tempo of 0, shorter than 30 seconds, or longer than 15 minutes. These are broken recordings, silence, or very long non-song tracks.
- Added a `decade` column and converted duration to minutes.
- **2020 is left out of decade-level analysis**, because it is a single year with only about 2,000 tracks.
- Final data: 169,685 tracks, with about 19,500–20,000 per decade from the 1950s to the 2010s and fewer in the 1920s–1940s (5,106 / 9,432 / 15,127).

## Method and results

### 1. Trends over time
Yearly and decade averages of each audio feature (`outputs/trends_by_year.png`, `outputs/decade_averages.csv`).

Energy and loudness are strongly correlated (0.78), and energy and acousticness are strongly negatively correlated (−0.75), which explains why the "acoustic to electric" shift shows up in several features at once (`outputs/feature_correlation.png`).

### 2. Predicting a song's decade
Inputs: 13 audio features. `year`, `release_date` and `popularity` were deliberately left out, because they would give the answer away. 80/20 stratified train-test split.

| Model | Exact accuracy | Within one decade |
|---|---|---|
| Logistic Regression (baseline) | 33.9% | – |
| Random Forest | 46.0% | 77.0% |
| Random Forest, spoken-word tracks removed | 44.8% | 76.5% |

- The Random Forest beats the linear baseline by about 12 points, which suggests the link between sound and era is non-linear.
- Most mistakes are between **neighbouring decades** (`outputs/confusion_matrix.png`), which fits the fact that music changes gradually and not in sharp breaks. The 1990s are the hardest decade to identify (recall 0.29), because they sit between the 1980s and 2000s. The 2010s are the easiest (F1 0.59).
- Feature importance: acousticness, loudness, duration and energy rank highest (`outputs/feature_importance.png`). Mode and explicit rank lowest, but Random Forest importance tends to favour continuous features, so this should not be read as "they don't matter".

### 3. Robustness check: audiobooks in the data
A cluster of tracks with very high speechiness turned out to be mostly **audiobook chapters** (for example, novels read aloud in Russian, German, Spanish and French), dated by the book's original publication year and not the recording date. About 4,900 such tracks (speechiness above 0.66) were removed and the model rerun. The headline result held (45% exact, 77% within one decade), but the prediction scores for the 1920s and 1930s dropped noticeably (1920s F1 from 0.64 to 0.51, 1930s from 0.48 to 0.37), which means the early decades looked easier to predict than real music is.

### 4. Clustering songs by sound (K-Means)
Ten scaled audio features, with no decade information. K was chosen using the elbow and silhouette plots (`outputs/choose_k.png`). K = 3 had the best silhouette score (0.24) and mostly separates old from new music and spoken word. K = 6 (silhouette about 0.17) gives a more detailed picture, so it is the main result. The low scores mean styles overlap and the clusters should be read as tendencies, not strict categories.

| Cluster | Sound profile | Suggested name | Songs |
|---|---|---|---|
| 0 | High danceability (0.69), happiest (valence 0.77) | Upbeat and danceable | 49,526 |
| 1 | Acousticness 0.81, energy 0.26, quiet | Quiet acoustic | 42,823 |
| 2 | Speechiness 0.88, about 2.7 minutes | Spoken word (mostly audiobooks) | 5,433 |
| 3 | Liveness 0.71, energy 0.61 | Live recordings | 10,742 |
| 4 | Instrumentalness 0.81, acousticness 0.90 | Instrumental and soft | 22,933 |
| 5 | Energy 0.74, loudest (−7.1 dB), fastest (135 BPM) | Loud, fast, high energy | 36,199 |

Cluster names are my interpretation of the audio profiles, not labels from the data. See `outputs/cluster_share_by_decade_k6.png` and `outputs/cluster_profiles_k6.csv` (K = 3 versions are saved alongside).

The share of each cluster by decade tells the same story as the averages: the quiet acoustic and instrumental styles dominate until the 1950s, the crossover happens in the 1960s, and the loud, high-energy style takes over from the 1970s onward. A hypothesis, not a tested conclusion: the lower average valence in the 2010s may partly reflect this shift in the mix of styles, since the loud high-energy cluster is less happy (valence 0.46) than the upbeat danceable one (0.77).

## Limitations

- **Spotify's catalogue is not a random sample of all music.** Older music is under-represented and biased towards recordings that survived or were re-released.
- **Audio features are algorithmic estimates**, not ground truth. Valence in particular is a model's guess at "positivity".
- **The `year` field is not always the recording year.** Audiobooks, re-releases and remasters can carry other dates, so early-decade results should be read with caution.
- **Recording technology and musical style are mixed together.** Older recordings sound different partly because of the equipment and not only the music.
- **The 1920s–1940s have fewer songs**, so their numbers are less stable. 2020 is excluded from decade comparisons.
- **The same artist can appear in both the training and test sets**, which may make the models look slightly better than they would on completely new artists. A split by artist (for example `GroupKFold`) would be a stricter test.
- **Exact decade boundaries are arbitrary.** A song from 1989 and one from 1990 sound alike, which is why the "within one decade" score is the fairer measure.
- **Correlation is not causation.** The analysis describes how music changed and does not test why.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
python music_project_clean.py
```

The script downloads the dataset from Kaggle (you may need a free Kaggle account), runs every step, and saves all charts and tables to the `outputs/` folder. A fixed random seed (42) makes the results repeatable. Running it from top to bottom takes a few minutes.

## Repository contents

- `music_project_clean.py`: the full analysis
- `outputs/`: charts and tables referred to above
