import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os
from PIL import Image
import io

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🐉 DragonFlix — Movie Recommender",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS — DARK CINEMATIC DRAGON THEME
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;900&family=Rajdhani:wght@300;400;600;700&family=Exo+2:wght@300;400;700&display=swap');

/* ── Root Variables ── */
:root {
    --fire-red:    #ff2b2b;
    --fire-orange: #ff6a00;
    --fire-gold:   #ffd700;
    --dark-bg:     #07070f;
    --card-bg:     #10101e;
    --card-border: #2a1f3d;
    --glow-red:    rgba(255,43,43,0.35);
    --glow-gold:   rgba(255,215,0,0.25);
    --text-primary:#f0e6ff;
    --text-muted:  #8a7fa0;
}

/* ── Base App ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background: var(--dark-bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: var(--fire-red); border-radius: 3px; }

/* ── Hide Streamlit watermark ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1e 0%, #0a0a15 100%) !important;
    border-right: 1px solid #2a1f3d !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Dragon Hero Banner ── */
.dragon-hero {
    width: 100%;
    height: 360px;
    border-radius: 20px;
    overflow: hidden;
    position: relative;
    margin-bottom: 2rem;
    box-shadow: 0 0 60px var(--glow-red), 0 0 120px rgba(255,43,43,0.15);
    border: 1px solid rgba(255,43,43,0.3);
}
.dragon-hero img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.dragon-hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to bottom,
        rgba(7,7,15,0.1) 0%,
        rgba(7,7,15,0.3) 40%,
        rgba(7,7,15,0.85) 100%
    );
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    padding: 2.5rem;
}
.dragon-hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff6a00, #ffd700, #ff2b2b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    line-height: 1.1;
    text-align: center;
    letter-spacing: 2px;
    animation: flamePulse 3s ease-in-out infinite;
}
@keyframes flamePulse {
    0%, 100% { filter: drop-shadow(0 0 12px #ff6a00) drop-shadow(0 0 30px #ff2b2b); }
    50%       { filter: drop-shadow(0 0 25px #ffd700) drop-shadow(0 0 50px #ff6a00); }
}
.dragon-hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: rgba(255,220,180,0.85);
    margin-top: 0.5rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Cinzel', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--fire-gold);
    border-left: 4px solid var(--fire-red);
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
    text-shadow: 0 0 15px var(--glow-gold);
    letter-spacing: 1px;
}

/* ── Movie Cards ── */
.movie-card {
    background: linear-gradient(145deg, #13132a, #0e0e20);
    border: 1px solid #2a1f4d;
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(.175,.885,.32,1.275);
    cursor: pointer;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.movie-card:hover {
    transform: translateY(-8px) scale(1.03);
    border-color: var(--fire-red);
    box-shadow: 0 12px 40px var(--glow-red), 0 0 0 1px rgba(255,43,43,0.2);
}
.movie-card img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.movie-card-body {
    padding: 0.9rem;
}
.movie-card-title {
    font-family: 'Cinzel', serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #f0e6ff;
    margin: 0 0 0.4rem 0;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-card-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rating-badge {
    background: linear-gradient(135deg, #ff6a00, #ffd700);
    color: #000;
    font-weight: 700;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-family: 'Exo 2', sans-serif;
}
.genre-tag {
    background: rgba(255,43,43,0.15);
    border: 1px solid rgba(255,43,43,0.3);
    color: #ff8080;
    font-size: 0.68rem;
    padding: 2px 7px;
    border-radius: 12px;
    margin: 0.1rem;
    display: inline-block;
}

/* ── Search Bar ── */
[data-testid="stTextInput"] input {
    background: #13132a !important;
    border: 1.5px solid #2a1f4d !important;
    border-radius: 12px !important;
    color: #f0e6ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1.2rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--fire-red) !important;
    box-shadow: 0 0 20px var(--glow-red) !important;
    outline: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #13132a !important;
    border: 1.5px solid #2a1f4d !important;
    border-radius: 12px !important;
    color: #f0e6ff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff2b2b, #ff6a00) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px var(--glow-red) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px var(--glow-red), 0 0 40px rgba(255,106,0,0.3) !important;
    background: linear-gradient(135deg, #ff0000, #ffd700) !important;
}

/* ── Info/Detail Panel ── */
.detail-panel {
    background: linear-gradient(145deg, #13132a, #0e0e1e);
    border: 1px solid #2a1f4d;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
}
.detail-title {
    font-family: 'Cinzel', serif;
    font-size: 1.9rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffd700, #ff6a00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.detail-overview {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.95rem;
    color: #b0a8c8;
    line-height: 1.7;
    margin-top: 0.8rem;
}

/* ── Stats Row ── */
.stat-box {
    text-align: center;
    padding: 0.8rem;
    background: rgba(255,43,43,0.07);
    border: 1px solid rgba(255,43,43,0.2);
    border-radius: 12px;
}
.stat-value {
    font-family: 'Cinzel', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--fire-gold);
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Fire divider ── */
.fire-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff2b2b, #ffd700, #ff6a00, transparent);
    border: none;
    margin: 1.5rem 0;
    animation: fireLine 2s ease-in-out infinite;
}
@keyframes fireLine {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}

/* ── Watchlist badge ── */
.watchlist-item {
    background: rgba(255,43,43,0.08);
    border: 1px solid rgba(255,43,43,0.2);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #d0c8e8;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #ff6a00 !important; }

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #10101e !important;
    border-bottom: 2px solid #2a1f4d !important;
    gap: 0.5rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.5rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(255,43,43,0.15) !important;
    color: var(--fire-gold) !important;
    border-bottom: 2px solid var(--fire-red) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: var(--fire-gold) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* ── No poster placeholder ── */
.no-poster {
    background: linear-gradient(135deg, #1a1a35, #0f0f22);
    border: 1px dashed #2a1f4d;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 2/3;
    color: #3a3060;
    font-size: 3rem;
}

/* ── Flame particle animation ── */
@keyframes floatUp {
    0%   { transform: translateY(0) scale(1);   opacity: 0.8; }
    100% { transform: translateY(-80px) scale(0.3); opacity: 0; }
}
.flame-emoji {
    display: inline-block;
    animation: floatUp 2s ease-out infinite;
    animation-delay: var(--delay, 0s);
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background: rgba(255,43,43,0.08) !important;
    border: 1px solid rgba(255,43,43,0.3) !important;
    border-radius: 12px !important;
    color: #ff9090 !important;
}

/* ── Main content padding ── */
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TMDB API CONFIGURATION
# ─────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"   # free public demo key
TMDB_BASE    = "https://api.themoviedb.org/3"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"
TMDB_IMG_LG  = "https://image.tmdb.org/t/p/original"
NO_POSTER    = "https://via.placeholder.com/300x450/10101e/3a3060?text=🐉+No+Poster"

GENRE_MAP = {
    28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime",
    99:"Documentary", 18:"Drama", 10751:"Family", 14:"Fantasy", 36:"History",
    27:"Horror", 10402:"Music", 9648:"Mystery", 10749:"Romance", 878:"Sci-Fi",
    10770:"TV Movie", 53:"Thriller", 10752:"War", 37:"Western"
}


# ─────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────
def init_state():
    defaults = {
        "watchlist":        [],
        "search_history":   [],
        "selected_movie":   None,
        "recommendations":  [],
        "popular_movies":   [],
        "trending_movies":  [],
        "genre_filter":     "All",
        "current_tab":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────
#  TMDB API HELPERS
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_get(endpoint, **params):
    params["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_popular_movies(page=1):
    data = tmdb_get("/movie/popular", page=page, language="en-US")
    return data.get("results", [])

@st.cache_data(ttl=3600, show_spinner=False)
def get_trending_movies():
    data = tmdb_get("/trending/movie/week")
    return data.get("results", [])

@st.cache_data(ttl=1800, show_spinner=False)
def search_movies(query, page=1):
    if not query.strip():
        return []
    data = tmdb_get("/search/movie", query=query, page=page, language="en-US")
    return data.get("results", [])

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_details(movie_id):
    return tmdb_get(f"/movie/{movie_id}", language="en-US", append_to_response="credits,similar,keywords,videos")

@st.cache_data(ttl=3600, show_spinner=False)
def get_movies_by_genre(genre_id, page=1):
    data = tmdb_get("/discover/movie", with_genres=genre_id, sort_by="popularity.desc",
                    page=page, language="en-US")
    return data.get("results", [])

def poster_url(path, large=False):
    if not path:
        return NO_POSTER
    base = TMDB_IMG_LG if large else TMDB_IMG
    return f"{base}{path}"

def genre_names(genre_ids):
    return [GENRE_MAP.get(gid, "") for gid in (genre_ids or []) if gid in GENRE_MAP]


# ─────────────────────────────────────────
#  CONTENT-BASED FILTERING ENGINE
# ─────────────────────────────────────────
class RecommendationEngine:
    def __init__(self):
        self.tfidf   = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix  = None
        self.movies  = []

    def build_corpus(self, movies):
        """Build TF-IDF corpus from a list of TMDB movie dicts."""
        self.movies = movies
        docs = []
        for m in movies:
            genres  = " ".join(genre_names(m.get("genre_ids", [])))
            overview = m.get("overview", "")
            title    = m.get("title", "")
            docs.append(f"{title} {title} {genres} {genres} {overview}")
        if docs:
            self.matrix = self.tfidf.fit_transform(docs)
        return self

    def build_from_detail(self, detail):
        """Build a rich corpus using movie detail + credits/keywords."""
        cast_names = " ".join(
            c["name"] for c in detail.get("credits", {}).get("cast", [])[:8]
        )
        crew_names = " ".join(
            c["name"] for c in detail.get("credits", {}).get("crew", [])
            if c.get("job") in ("Director", "Producer")
        )
        keywords = " ".join(
            k["name"] for k in detail.get("keywords", {}).get("keywords", [])[:15]
        )
        genres    = " ".join(g["name"] for g in detail.get("genres", []))
        overview  = detail.get("overview", "")
        title     = detail.get("title", "")
        return f"{title} {title} {genres} {genres} {cast_names} {crew_names} {keywords} {overview}"

    def recommend(self, source_movie, candidate_movies, top_n=10):
        """Return top_n recommended movies from candidates similar to source_movie."""
        if not candidate_movies:
            return []
        all_movies = [source_movie] + candidate_movies
        docs = []
        for m in all_movies:
            genres   = " ".join(genre_names(m.get("genre_ids", [])))
            overview = m.get("overview", "")
            title    = m.get("title", "")
            docs.append(f"{title} {title} {genres} {genres} {overview}")
        try:
            matrix  = self.tfidf.fit_transform(docs)
            scores  = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            indices = scores.argsort()[::-1][:top_n]
            return [candidate_movies[i] for i in indices if scores[i] > 0]
        except Exception:
            return candidate_movies[:top_n]

    def recommend_from_detail(self, detail, top_n=10):
        """Use TMDB similar movies (already content-filtered server-side) + re-rank."""
        similar = detail.get("similar", {}).get("results", [])
        if not similar:
            return []
        source_text = self.build_from_detail(detail)
        docs = [source_text]
        for m in similar:
            genres   = " ".join(genre_names(m.get("genre_ids", [])))
            overview = m.get("overview", "")
            title    = m.get("title", "")
            docs.append(f"{title} {title} {genres} {overview}")
        try:
            matrix  = self.tfidf.fit_transform(docs)
            scores  = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            indices = scores.argsort()[::-1][:top_n]
            return [similar[i] for i in indices]
        except Exception:
            return similar[:top_n]


engine = RecommendationEngine()


# ─────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────
def render_hero():
    """Dragon hero banner with animated title."""
    img_path = os.path.join(os.path.dirname(__file__), "dragon_banner.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="Dragon Banner">'
    else:
        img_tag = '<div style="width:100%;height:100%;background:linear-gradient(135deg,#1a0000,#3d0000,#1a0000);display:flex;align-items:center;justify-content:center;font-size:8rem;">🐉</div>'

    st.markdown(f"""
    <div class="dragon-hero">
        {img_tag}
        <div class="dragon-hero-overlay">
            <div class="dragon-hero-title">🐉 DRAGONFLIX</div>
            <div class="dragon-hero-subtitle">
                <span class="flame-emoji" style="--delay:0s">🔥</span>
                &nbsp; Your Legendary Movie Recommender &nbsp;
                <span class="flame-emoji" style="--delay:0.8s">🔥</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_movie_card(movie, cols_per_row=5, show_btn=True):
    """Render a single movie card with poster, title, rating."""
    title    = movie.get("title", "Unknown")
    rating   = movie.get("vote_average", 0)
    year     = (movie.get("release_date") or "")[:4]
    genre_ids = movie.get("genre_ids", [])
    genres   = genre_names(genre_ids)[:2]
    poster   = poster_url(movie.get("poster_path"))
    movie_id = movie.get("id", 0)

    genre_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres)
    rating_color = "#ff4b4b" if rating < 6 else "#ffd700" if rating < 7.5 else "#4bff91"

    st.markdown(f"""
    <div class="movie-card">
        <img src="{poster}" alt="{title}" onerror="this.src='{NO_POSTER}'">
        <div class="movie-card-body">
            <div class="movie-card-title" title="{title}">{title}</div>
            <div class="movie-card-meta">
                <span style="color:{rating_color};font-weight:700;font-family:'Cinzel',serif;">
                    ⭐ {rating:.1f}
                </span>
                <span style="color:var(--text-muted)">{year}</span>
            </div>
            <div style="margin-top:0.4rem">{genre_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_btn:
        if st.button("🎬 Details & Recs", key=f"detail_{movie_id}_{title[:8]}"):
            with st.spinner("🐉 Fetching dragon intelligence..."):
                detail = get_movie_details(movie_id)
            st.session_state["selected_movie"] = detail
            recs = engine.recommend_from_detail(detail)
            st.session_state["recommendations"] = recs
            st.rerun()


def render_movie_grid(movies, cols=5, show_btn=True):
    """Render a responsive grid of movie cards."""
    if not movies:
        st.info("🐉 No movies found. Try a different search!")
        return
    rows = [movies[i:i+cols] for i in range(0, len(movies), cols)]
    for row in rows:
        grid_cols = st.columns(cols)
        for col, movie in zip(grid_cols, row):
            with col:
                render_movie_card(movie, cols, show_btn)


def render_detail_panel(detail):
    """Render a detailed movie info panel."""
    if not detail:
        return
    title     = detail.get("title", "Unknown")
    overview  = detail.get("overview", "No description available.")
    rating    = detail.get("vote_average", 0)
    year      = (detail.get("release_date") or "")[:4]
    runtime   = detail.get("runtime", 0)
    genres    = [g["name"] for g in detail.get("genres", [])]
    tagline   = detail.get("tagline", "")
    poster    = poster_url(detail.get("poster_path"), large=True)
    movie_id  = detail.get("id", 0)
    budget    = detail.get("budget", 0)
    revenue   = detail.get("revenue", 0)

    cast_list = detail.get("credits", {}).get("cast", [])[:8]
    directors = [c["name"] for c in detail.get("credits", {}).get("crew", [])
                 if c.get("job") == "Director"]

    st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎬 Movie Details</div>', unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 2.5], gap="large")

    with col_img:
        st.markdown(f"""
        <img src="{poster}" alt="{title}" style="width:100%;border-radius:16px;
             box-shadow:0 0 40px rgba(255,43,43,0.4);border:2px solid rgba(255,43,43,0.3);"
             onerror="this.src='{NO_POSTER}'">
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Watchlist button
        in_wl = any(w.get("id") == movie_id for w in st.session_state["watchlist"])
        if in_wl:
            if st.button("✅ In Watchlist", key=f"wl_remove_{movie_id}"):
                st.session_state["watchlist"] = [
                    w for w in st.session_state["watchlist"] if w.get("id") != movie_id
                ]
                st.rerun()
        else:
            if st.button("➕ Add to Watchlist", key=f"wl_add_{movie_id}"):
                st.session_state["watchlist"].append({
                    "id": movie_id, "title": title, "rating": rating, "year": year
                })
                st.rerun()

    with col_info:
        genre_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres)
        st.markdown(f"""
        <div class="detail-panel">
            <div class="detail-title">{title}</div>
            {'<div style="color:#ff8040;font-style:italic;margin-bottom:0.5rem;font-size:0.9rem;">'+tagline+'</div>' if tagline else ''}
            <div style="margin:0.5rem 0">{genre_html}</div>
            <div class="detail-overview">{overview}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in zip(
            [c1, c2, c3, c4],
            ["⭐ Rating", "📅 Year", "⏱ Runtime", "🎬 Votes"],
            [
                f"{rating:.1f}/10",
                year,
                f"{runtime} min" if runtime else "N/A",
                f"{detail.get('vote_count', 0):,}"
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if directors:
            st.markdown(f"""
            <div style="margin-top:1rem;color:var(--text-muted);font-size:0.85rem;">
                🎥 <strong style="color:#ffd700">Director:</strong>
                {', '.join(directors)}
            </div>
            """, unsafe_allow_html=True)

        if cast_list:
            cast_names = ", ".join(c["name"] for c in cast_list)
            st.markdown(f"""
            <div style="margin-top:0.4rem;color:var(--text-muted);font-size:0.85rem;">
                🌟 <strong style="color:#ffd700">Cast:</strong> {cast_names}
            </div>
            """, unsafe_allow_html=True)

        if budget:
            st.markdown(f"""
            <div style="margin-top:0.4rem;color:var(--text-muted);font-size:0.85rem;">
                💰 <strong style="color:#ffd700">Budget:</strong> ${budget:,.0f}
                &nbsp;|&nbsp;
                🏆 <strong style="color:#ffd700">Revenue:</strong> ${revenue:,.0f}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <span style="font-size:3.5rem;animation:flamePulse 2s infinite">🐉</span>
            <div style="font-family:'Cinzel',serif;font-size:1.2rem;color:#ffd700;
                        font-weight:700;letter-spacing:2px;margin-top:0.3rem;">
                DRAGONFLIX
            </div>
            <div style="font-size:0.7rem;color:#5a4f70;letter-spacing:3px;text-transform:uppercase;">
                Movie Recommender
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)

        # Genre filter
        st.markdown("**🎭 Filter by Genre**")
        genre_options = ["All"] + sorted(GENRE_MAP.values())
        selected_genre = st.selectbox("Genre", genre_options, label_visibility="collapsed")
        st.session_state["genre_filter"] = selected_genre

        st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)

        # Stats
        st.markdown("**📊 Your Stats**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Watchlist", len(st.session_state["watchlist"]))
        with c2:
            st.metric("Searches", len(st.session_state["search_history"]))

        st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)

        # Watchlist
        st.markdown("**📋 My Watchlist**")
        if not st.session_state["watchlist"]:
            st.caption("No movies added yet. Click ➕ on any movie!")
        else:
            for i, item in enumerate(st.session_state["watchlist"]):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"""
                    <div style="font-size:0.82rem;color:#d0c8e8;">
                        🎬 {item['title']}<br>
                        <span style="color:#5a4f70;font-size:0.72rem;">
                            ⭐ {item.get('rating', '?'):.1f} · {item.get('year','?')}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    if st.button("✕", key=f"rm_wl_{i}"):
                        st.session_state["watchlist"].pop(i)
                        st.rerun()

        st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)

        # About
        st.markdown("""
        <div style="font-size:0.75rem;color:#3a3050;text-align:center;line-height:1.6;">
            🐉 Powered by TMDB API<br>
            🤖 Content-Based ML Filtering<br>
            ⚡ Built with Streamlit
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────
def main():
    render_sidebar()
    render_hero()

    # ── Tabs ──
    tab_home, tab_search, tab_trending, tab_genres = st.tabs([
        "🏠  Home", "🔍  Search", "📈  Trending", "🎭  Genres"
    ])

    # ════════════════════════════════════
    #  TAB 1 — HOME
    # ════════════════════════════════════
    with tab_home:
        st.markdown('<div class="section-header">🔥 Popular Right Now</div>', unsafe_allow_html=True)

        with st.spinner("🐉 Dragon fetching movies..."):
            popular = get_popular_movies()

        # Apply genre filter
        genre_filter = st.session_state["genre_filter"]
        if genre_filter != "All":
            gid = {v: k for k, v in GENRE_MAP.items()}.get(genre_filter)
            if gid:
                popular = [m for m in popular if gid in m.get("genre_ids", [])]

        if popular:
            render_movie_grid(popular[:10], cols=5)
        else:
            st.warning("No movies found with this filter.")

        # ── Selected movie detail + recommendations ──
        if st.session_state["selected_movie"]:
            detail = st.session_state["selected_movie"]
            render_detail_panel(detail)

            recs = st.session_state.get("recommendations", [])
            if recs:
                st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="section-header">🐉 Because you liked '
                    f'<em>{detail.get("title","this")}</em>...</div>',
                    unsafe_allow_html=True
                )
                render_movie_grid(recs[:10], cols=5)

            if st.button("🔙 Back to Home", key="back_home"):
                st.session_state["selected_movie"] = None
                st.session_state["recommendations"] = []
                st.rerun()

    # ════════════════════════════════════
    #  TAB 2 — SEARCH
    # ════════════════════════════════════
    with tab_search:
        st.markdown('<div class="section-header">🔍 Search the Dragon\'s Vault</div>', unsafe_allow_html=True)

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query = st.text_input(
                "Search movies",
                placeholder="🎬 Type a movie title, genre, or actor...",
                label_visibility="collapsed",
                key="search_query"
            )
        with col_btn:
            search_clicked = st.button("🐉 Search", key="do_search")

        if query or search_clicked:
            if query and query not in st.session_state["search_history"]:
                st.session_state["search_history"].insert(0, query)
                st.session_state["search_history"] = st.session_state["search_history"][:10]

            with st.spinner(f"🐉 Searching for *{query}*..."):
                results = search_movies(query)

            if results:
                st.markdown(f"""
                <div style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1rem;">
                    Found <strong style="color:#ffd700">{len(results)}</strong> results for
                    <em style="color:#ff6a00">"{query}"</em>
                </div>
                """, unsafe_allow_html=True)
                render_movie_grid(results[:10], cols=5)

                # ── Recommendations based on top result ──
                if st.button("🤖 Get AI Recommendations from Top Result", key="rec_from_search"):
                    with st.spinner("🐉 Computing dragon intelligence..."):
                        top_detail = get_movie_details(results[0]["id"])
                        recs = engine.recommend_from_detail(top_detail)
                        st.session_state["selected_movie"] = top_detail
                        st.session_state["recommendations"] = recs
                    st.rerun()
            else:
                st.warning(f"🐉 No results found for '{query}'. Try another title!")

        # Recent Searches
        if st.session_state["search_history"]:
            st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)
            st.markdown("**🕓 Recent Searches**")
            cols = st.columns(5)
            for i, h in enumerate(st.session_state["search_history"][:5]):
                with cols[i % 5]:
                    if st.button(f"🔍 {h}", key=f"hist_{i}"):
                        st.session_state["search_query"] = h
                        st.rerun()

    # ════════════════════════════════════
    #  TAB 3 — TRENDING
    # ════════════════════════════════════
    with tab_trending:
        st.markdown('<div class="section-header">📈 Trending This Week</div>', unsafe_allow_html=True)

        with st.spinner("🐉 Fetching this week's fire..."):
            trending = get_trending_movies()

        if trending:
            # Top 3 podium
            st.markdown("**🏆 Top 3 of the Week**")
            top3 = st.columns(3, gap="large")
            medals = ["🥇", "🥈", "🥉"]
            for i, (col, movie) in enumerate(zip(top3, trending[:3])):
                with col:
                    title  = movie.get("title", "Unknown")
                    poster = poster_url(movie.get("poster_path"))
                    rating = movie.get("vote_average", 0)
                    year   = (movie.get("release_date") or "")[:4]
                    st.markdown(f"""
                    <div class="movie-card" style="border-color:{'#ffd700' if i==0 else '#c0c0c0' if i==1 else '#cd7f32'}">
                        <div style="text-align:center;padding:0.5rem;font-size:2rem">{medals[i]}</div>
                        <img src="{poster}" alt="{title}" onerror="this.src='{NO_POSTER}'">
                        <div class="movie-card-body">
                            <div class="movie-card-title">{title}</div>
                            <div class="movie-card-meta">
                                <span style="color:#ffd700;font-weight:700">⭐ {rating:.1f}</span>
                                <span>{year}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🎬 Details", key=f"trend_top_{movie.get('id')}"):
                        with st.spinner("Fetching..."):
                            detail = get_movie_details(movie.get("id"))
                        recs = engine.recommend_from_detail(detail)
                        st.session_state["selected_movie"] = detail
                        st.session_state["recommendations"] = recs
                        st.rerun()

            st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)
            st.markdown("**🔥 More Trending**")
            render_movie_grid(trending[3:13], cols=5)
        else:
            st.warning("Could not fetch trending movies.")

        if st.session_state["selected_movie"]:
            detail = st.session_state["selected_movie"]
            render_detail_panel(detail)
            recs = st.session_state.get("recommendations", [])
            if recs:
                st.markdown('<div class="section-header">🐉 Similar Movies</div>', unsafe_allow_html=True)
                render_movie_grid(recs[:10], cols=5)
            if st.button("🔙 Back", key="back_trending"):
                st.session_state["selected_movie"] = None
                st.rerun()

    # ════════════════════════════════════
    #  TAB 4 — GENRES
    # ════════════════════════════════════
    with tab_genres:
        st.markdown('<div class="section-header">🎭 Browse by Genre</div>', unsafe_allow_html=True)

        # Genre selector as pill buttons
        all_genres = sorted(GENRE_MAP.items(), key=lambda x: x[1])
        genre_cols = st.columns(6)
        selected_genre_id = None

        if "selected_genre_id" not in st.session_state:
            st.session_state["selected_genre_id"] = 28  # Action by default

        for i, (gid, gname) in enumerate(all_genres):
            col = genre_cols[i % 6]
            with col:
                is_selected = st.session_state["selected_genre_id"] == gid
                btn_style = "primary" if is_selected else "secondary"
                if st.button(
                    f"{'🔥 ' if is_selected else ''}{gname}",
                    key=f"genre_btn_{gid}",
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_genre_id"] = gid
                    st.rerun()

        st.markdown('<hr class="fire-divider">', unsafe_allow_html=True)

        active_gid  = st.session_state["selected_genre_id"]
        active_name = GENRE_MAP.get(active_gid, "Movies")
        st.markdown(
            f'<div class="section-header">🎬 Top {active_name} Movies</div>',
            unsafe_allow_html=True
        )

        with st.spinner(f"🐉 Loading {active_name} movies..."):
            genre_movies = get_movies_by_genre(active_gid)

        if genre_movies:
            render_movie_grid(genre_movies[:10], cols=5)
        else:
            st.warning("No movies found for this genre.")

        if st.session_state["selected_movie"]:
            detail = st.session_state["selected_movie"]
            render_detail_panel(detail)
            recs = st.session_state.get("recommendations", [])
            if recs:
                st.markdown('<div class="section-header">🐉 Similar Movies</div>', unsafe_allow_html=True)
                render_movie_grid(recs[:10], cols=5)
            if st.button("🔙 Back", key="back_genres"):
                st.session_state["selected_movie"] = None
                st.rerun()


if __name__ == "__main__":
    main()
