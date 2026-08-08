# 🐉 DragonFlix — Movie Recommendation System

<div align="center">
  <img src="dragon_banner.jpg" alt="DragonFlix Banner" width="800" style="border-radius:16px"/>
  <h3>🔥 Legendary Movie Recommendations Powered by Dragon Intelligence 🔥</h3>
</div>

---

## ✨ Features

- 🐉 **Dragon-themed cinematic UI** with animated fire effects & glassmorphism
- 🎬 **TMDB API Integration** — live posters, ratings, cast, genres, runtime
- 🤖 **Content-Based Filtering** — TF-IDF + Cosine Similarity recommendation engine
- 📈 **Trending & Popular** movies updated weekly
- 🎭 **Genre Browser** — explore by Action, Horror, Sci-Fi, Romance, and more
- 🔍 **Instant Search** with AI recommendations from top results
- 📋 **Personal Watchlist** — save movies to your sidebar
- 🕓 **Search History** — quick-access recent searches

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Web framework |
| `TMDB API` | Movie data (posters, ratings, cast) |
| `scikit-learn` | TF-IDF vectorization + cosine similarity |
| `pandas / numpy` | Data manipulation |
| `Pillow` | Image processing |

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mohit4215/movie-recommendation.git
cd movie-recommendation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 📁 Project Structure

```
movie-recommendation/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── dragon_banner.jpg      # Hero banner image
├── .streamlit/
│   └── config.toml       # Dark cinematic theme
└── README.md
```

## 🎯 How It Works

### Content-Based Filtering
1. Fetches movie data from TMDB API (overview, genres, cast, keywords)
2. Builds a **TF-IDF matrix** from combined text features
3. Computes **cosine similarity** between the selected movie and candidates
4. Returns top-N most similar movies, ranked by similarity score

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select `mohit4215/movie-recommendation` → `app.py`
5. Click **Deploy** 🚀

---

<div align="center">Made with 🐉 & ❤️ using Python + Streamlit</div>
