# ============================================================================
# STREAMLIT WEB APP FOR MOVIE RECOMMENDATION SYSTEM
# ============================================================================
# Save this file as 'app.py' and run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []
if 'selected_movie' not in st.session_state:
    st.session_state['selected_movie'] = None

# Custom CSS - Professional dark/glassmorphism theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Remove default padding */
    .main > div {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Settings row styling */
    .settings-row {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Movie info card - Dark theme */
    .movie-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(102,126,234,0.3);
    }
    .movie-card h3 {
        margin: 0 0 0.3rem 0;
        color: #e2e8f0;
        font-size: 1.1rem;
    }
    .movie-card p {
        margin: 0;
        color: #a0aec0;
        font-size: 0.85rem;
    }
    
    /* Recommendation cards - Dark theme with gradient */
    .rec-card {
        background: linear-gradient(135deg, rgba(45,55,72,0.9) 0%, rgba(26,32,44,0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        transition: all 0.2s;
    }
    .rec-card:hover {
        transform: translateX(3px);
        background: linear-gradient(135deg, rgba(55,65,81,0.95) 0%, rgba(36,42,54,0.95) 100%);
        border-left: 3px solid #764ba2;
    }
    .rec-card h4 {
        margin: 0 0 0.3rem 0;
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
    }
    .rec-card p {
        margin: 0;
        color: #a0aec0;
        font-size: 0.8rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 15px;
        font-size: 0.7rem;
        display: inline-block;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.4);
    }
    
    /* Divider */
    hr {
        margin: 0.8rem 0;
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #718096;
        font-size: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 1.5rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Select box styling */
    .stSelectbox label, .stSlider label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #e2e8f0;
    }
    
    /* Select box background */
    .stSelectbox > div > div {
        background-color: rgba(45,55,72,0.8);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(45,55,72,0.8);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Info/warning boxes */
    .stAlert {
        background-color: rgba(45,55,72,0.8);
        color: #e2e8f0;
    }
    
    /* Spinner text */
    .stSpinner > div > div {
        color: #e2e8f0;
    }
    
    /* Caption styling */
    .stCaption {
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Content-Based + Collaborative Filtering | Powered by MovieLens</p>
</div>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('models/movie_to_idx.pkl', 'rb') as f:
            movie_to_idx = pickle.load(f)
        with open('models/knn.pkl', 'rb') as f:
            knn = pickle.load(f)
        with open('models/movies.pkl', 'rb') as f:
            movies_df = pickle.load(f)
        
        tfidf_matrix = load_npz('models/tfidf_matrix.npz')
        user_item_matrix = load_npz('models/user_item_matrix.npz')
        
        return tfidf, tfidf_matrix, movie_to_idx, knn, user_item_matrix, movies_df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Recommendation functions
def normalize_scores(scores):
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    min_val, max_val = values.min(), values.max()
    if max_val - min_val < 1e-8:
        normalized = np.ones_like(values) * 0.5
    else:
        normalized = (values - min_val) / (max_val - min_val)
    return {item: float(score) for item, score in zip(scores.keys(), normalized)}

def get_user_confidence(user_id, ratings_df, min_ratings=5):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    n_ratings = len(user_ratings)
    if n_ratings == 0:
        return 0.0
    return min(1.0, n_ratings / min_ratings)

def adaptive_weights(user_confidence):
    if user_confidence < 0.3:
        return 0.8, 0.2
    elif user_confidence < 0.7:
        return 0.5, 0.5
    else:
        return 0.3, 0.7

def calculate_popularity_scores(ratings_df):
    popularity = ratings_df.groupby('movieId').size().to_dict()
    if not popularity:
        return {}
    scaler = MinMaxScaler()
    movie_ids = list(popularity.keys())
    counts = np.array(list(popularity.values())).reshape(-1, 1)
    normalized_counts = scaler.fit_transform(counts).flatten()
    return {movie: float(score) for movie, score in zip(movie_ids, normalized_counts)}

def merge_recommendations(cb_scores, cf_scores, user_id, ratings_df, content_weight=0.6):
    norm_cb = normalize_scores(cb_scores)
    norm_cf = normalize_scores(cf_scores)
    user_conf = get_user_confidence(user_id, ratings_df)
    cb_w, cf_w = adaptive_weights(user_conf)
    
    hybrid_scores = {}
    all_items = set(norm_cb.keys()) | set(norm_cf.keys())
    for item in all_items:
        cb_score = norm_cb.get(item, 0.0)
        cf_score = norm_cf.get(item, 0.0)
        hybrid_scores[item] = (cb_w * cb_score) + (cf_w * cf_score)
    
    popularity_scores = calculate_popularity_scores(ratings_df)
    for item in hybrid_scores:
        hybrid_scores[item] += 0.10 * popularity_scores.get(item, 0)
        if item in cb_scores and item in cf_scores:
            hybrid_scores[item] += 0.10
    
    return dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True))

def get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, top_n=10, threshold=0.15):
    if movie_id not in movie_to_idx:
        return []
    idx = movie_to_idx[movie_id]
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][1:top_n+1]
    recommendations = []
    for i in top_indices:
        if similarities[i] > threshold:
            recommendations.append({
                'title': movies_df.iloc[i]['title'],
                'similarity': similarities[i],
                'genres': movies_df.iloc[i]['genres']
            })
    return recommendations

def get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, top_n=10, threshold=0.15):
    if movie_id not in movie_to_idx:
        return []
    idx = movie_to_idx[movie_id]
    distances, indices = knn.kneighbors(user_item_matrix.T[idx], n_neighbors=top_n+1)
    recommendations = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        similarity = 1 - dist
        if similarity > threshold:
            recommendations.append({
                'title': movies_df.iloc[i]['title'],
                'similarity': similarity,
                'genres': movies_df.iloc[i]['genres']
            })
    return recommendations

def get_hybrid_recs(movie_id, user_id, movies_df, tfidf_matrix, movie_to_idx, 
                    user_item_matrix, knn, ratings_df, top_n=5, content_weight=0.6, threshold=0.15):
    
    content_recs = get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, top_n*2, threshold)
    collab_recs = get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, top_n*2, threshold)
    
    content_scores = {}
    collab_scores = {}
    
    for rec in content_recs:
        movie_match = movies_df[movies_df['title'] == rec['title']]
        if not movie_match.empty:
            content_scores[movie_match.iloc[0]['movieId']] = rec['similarity']
    
    for rec in collab_recs:
        movie_match = movies_df[movies_df['title'] == rec['title']]
        if not movie_match.empty:
            collab_scores[movie_match.iloc[0]['movieId']] = rec['similarity']
    
    hybrid_scores = merge_recommendations(content_scores, collab_scores, user_id, ratings_df, content_weight)
    
    recommendations = []
    for movie_id, score in list(hybrid_scores.items())[:top_n]:
        movie_data = movies_df[movies_df['movieId'] == movie_id]
        if not movie_data.empty:
            recommendations.append({
                'title': movie_data.iloc[0]['title'],
                'genres': movie_data.iloc[0]['genres'],
                'score': round(score, 3)
            })
    return recommendations

def search_movies(query, movies_df, limit=10):
    query = query.lower().strip()
    if len(query) == 0:
        return []
    results = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
    return results['title'].head(limit).tolist()

# Load models
with st.spinner("Loading models..."):
    tfidf, tfidf_matrix, movie_to_idx, knn, user_item_matrix, movies_df = load_models()

if movies_df is not None:
    ratings_df = pd.read_csv("Dataset/rating.csv")
    
    # Settings Row - Always visible
    with st.container():
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_a:
            rec_type = st.selectbox(
                "🎯 Recommendation Type",
                ["Hybrid", "Content-Based", "Collaborative"],
                help="Choose how recommendations are generated"
            )
        
        with col_b:
            num_recs = st.slider("📊 Number", 3, 10, 5, help="How many movies to recommend")
        
        with col_c:
            if rec_type == "Hybrid":
                content_weight = st.slider("🎭 Content Weight", 0.0, 1.0, 0.6, 
                                          help="Higher = more genre similarity")
    
    # Main content area
    col_search, col_recs = st.columns([1, 1.2], gap="medium")
    
    with col_search:
        st.markdown("### 🔍 Search Movie")
        search_input = st.text_input("", placeholder="Type movie name...", label_visibility="collapsed")
        
        if search_input:
            suggestions = search_movies(search_input, movies_df, 10)
            if suggestions:
                selected = st.selectbox("Select:", suggestions, label_visibility="collapsed")
                
                if selected:
                    movie_data = movies_df[movies_df['title'] == selected]
                    if not movie_data.empty:
                        movie_id = movie_data.iloc[0]['movieId']
                        genres = movie_data.iloc[0]['genres']
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>🎬 {selected}</h3>
                            <p>{genres.replace(' ', ' • ').title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("✨ Get Recommendations", use_container_width=True):
                            with st.spinner("Processing..."):
                                if rec_type == "Hybrid":
                                    recs = get_hybrid_recs(movie_id, 1, movies_df, tfidf_matrix, movie_to_idx,
                                                          user_item_matrix, knn, ratings_df, num_recs, content_weight)
                                elif rec_type == "Content-Based":
                                    recs = get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, num_recs)
                                    recs = [{'title': r['title'], 'genres': r['genres'], 'score': r['similarity']} for r in recs]
                                else:
                                    recs = get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, num_recs)
                                    recs = [{'title': r['title'], 'genres': r['genres'], 'score': r['similarity']} for r in recs]
                                
                                st.session_state['recommendations'] = recs
                                st.session_state['selected_movie'] = selected
            else:
                st.info("No movies found")
    
    with col_recs:
        st.markdown("### 🎯 Recommendations")
        
        if st.session_state['recommendations']:
            st.caption(f"Based on: **{st.session_state['selected_movie']}** | Using: **{rec_type}**")
            st.markdown("---")
            
            for i, rec in enumerate(st.session_state['recommendations'], 1):
                st.markdown(f"""
                <div class="rec-card">
                    <h4>{i}. {rec['title']}</h4>
                    <p>{rec['genres'].replace(' ', ' • ').title()}</p>
                    <span class="score-badge">⭐ Score: {rec['score']:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <p style="color: #a0aec0;">✨ Select a movie and click</p>
                <p style="color: #a0aec0;">"Get Recommendations"</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("Models not found. Please run the training notebook first.")

# Footer
st.markdown("""
<div class="footer">
    MovieLens Dataset | Hybrid Recommendation System | Content-Based + Collaborative Filtering
</div>
""", unsafe_allow_html=True)