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

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        color: white;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #e0e0e0;
    }
    .rec-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .rec-card h3 {
        color: white;
        margin: 0 0 0.5rem 0;
    }
    .rec-card p {
        color: #cbd5e0;
        margin: 0.3rem 0;
    }
    .similarity-badge {
        background-color: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #5a67d8;
    }
    /* Updated movie info box with dark background */
    .movie-info {
        background-color: #1a202c;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4a5568;
    }
    .movie-info p {
        color: #e2e8f0;
        margin: 0.3rem 0;
    }
    .movie-info strong {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Movie Recommendation System</h1>
    <p>Content-Based + Collaborative Filtering Hybrid</p>
</div>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
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
def get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, top_n=10):
    """Content-based recommendations"""
    if movie_id not in movie_to_idx:
        return []
    
    idx = movie_to_idx[movie_id]
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][1:top_n+1]
    
    recommendations = []
    for i in top_indices:
        if similarities[i] > 0.15:
            recommendations.append({
                'title': movies_df.iloc[i]['title'],
                'similarity': similarities[i],
                'genres': movies_df.iloc[i]['genres']
            })
    return recommendations

def get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, top_n=10):
    """Collaborative recommendations"""
    if movie_id not in movie_to_idx:
        return []
    
    idx = movie_to_idx[movie_id]
    distances, indices = knn.kneighbors(user_item_matrix.T[idx], n_neighbors=top_n+1)
    
    recommendations = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        similarity = 1 - dist
        if similarity > 0.15:
            recommendations.append({
                'title': movies_df.iloc[i]['title'],
                'similarity': similarity,
                'genres': movies_df.iloc[i]['genres']
            })
    return recommendations

def get_hybrid_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx,
                    user_item_matrix, knn, top_n=10, content_weight=0.6):
    """Hybrid recommendations combining both methods"""
    
    content_recs = get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, top_n*2)
    collab_recs = get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, top_n*2)
    
    scores = {}
    for rec in content_recs:
        scores[rec['title']] = {
            'score': rec['similarity'] * content_weight,
            'genres': rec['genres']
        }
    
    for rec in collab_recs:
        if rec['title'] in scores:
            scores[rec['title']]['score'] += rec['similarity'] * (1 - content_weight)
        else:
            scores[rec['title']] = {
                'score': rec['similarity'] * (1 - content_weight),
                'genres': rec['genres']
            }
    
    sorted_recs = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    results = []
    for title, data in sorted_recs[:top_n]:
        results.append({
            'title': title,
            'similarity': data['score'],
            'genres': data['genres']
        })
    return results

def search_movies(query, movies_df, limit=10):
    """Search movies by title"""
    query = query.lower().strip()
    if len(query) == 0:
        return []
    results = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
    return results['title'].head(limit).tolist()

# Load models
with st.spinner("Loading recommendation models..."):
    tfidf, tfidf_matrix, movie_to_idx, knn, user_item_matrix, movies_df = load_models()

if movies_df is not None:
    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        
        rec_type = st.selectbox(
            "Recommendation Algorithm",
            ["Hybrid", "Content-Based", "Collaborative"]
        )
        
        if rec_type == "Hybrid":
            content_weight = st.slider(
                "Content Weight", 0.0, 1.0, 0.6,
                help="Higher = more genre similarity, Lower = more user patterns"
            )
        else:
            content_weight = 0.6
        
        num_recs = st.slider("Number of Recommendations", 3, 10, 5)
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        st.metric("Total Movies", len(movies_df))
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## Search Movie")
        search_term = st.text_input("Enter movie name:", placeholder="e.g., Inception, Top Gun, Matrix")
        
        if search_term:
            suggestions = search_movies(search_term, movies_df, 10)
            if suggestions:
                selected_movie = st.selectbox("Select movie:", suggestions)
                
                if selected_movie:
                    movie_data = movies_df[movies_df['title'] == selected_movie]
                    if len(movie_data) > 0:
                        movie_id = movie_data.iloc[0]['movieId']
                        movie_genres = movie_data.iloc[0]['genres']
                        
                        st.markdown(f"""
                        <div class="movie-info">
                            <p><strong>Selected Movie:</strong> {selected_movie}</p>
                            <p><strong>Genres:</strong> {movie_genres}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Get Recommendations", use_container_width=True):
                            if rec_type == "Content-Based":
                                recs = get_content_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx, num_recs)
                            elif rec_type == "Collaborative":
                                recs = get_collab_recs(movie_id, movies_df, user_item_matrix, knn, movie_to_idx, num_recs)
                            else:
                                recs = get_hybrid_recs(movie_id, movies_df, tfidf_matrix, movie_to_idx,
                                                      user_item_matrix, knn, num_recs, content_weight)
                            
                            st.session_state['recommendations'] = recs
                            st.session_state['selected_movie'] = selected_movie
            else:
                st.warning("No movies found")
    
    with col2:
        if 'recommendations' in st.session_state and st.session_state['recommendations']:
            st.markdown(f"## Recommendations for: {st.session_state['selected_movie']}")
            st.markdown("---")
            
            for i, rec in enumerate(st.session_state['recommendations'], 1):
                st.markdown(f"""
                <div class="rec-card">
                    <h3>{i}. {rec['title']}</h3>
                    <p><strong>Genres:</strong> {rec['genres']}</p>
                    <p><span class="similarity-badge">Similarity: {rec['similarity']:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
        elif 'recommendations' in st.session_state:
            st.info("No recommendations found. Try a different movie.")

else:
    st.error("Models not found. Please run the training notebook first to save models.")
    st.info("""
    **How to fix:**
    1. Run all cells in your Jupyter notebook
    2. Make sure models are saved to 'models/' folder
    3. Restart this app with: streamlit run app.py
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Powered by MovieLens Dataset | Content-Based + Collaborative Filtering</p>", unsafe_allow_html=True)