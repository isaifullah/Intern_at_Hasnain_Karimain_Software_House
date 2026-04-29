# 🎬 Hybrid Movie Recommendation System

A production-ready hybrid recommendation system combining Content-Based Filtering and Collaborative Filtering to provide personalized movie recommendations.

## 📖 Overview
This project builds a recommendation engine using two approaches: Content-Based Filtering (finds similar movies based on genres/tags using TF-IDF) and Collaborative Filtering (finds movies liked by similar users using KNN). The hybrid system combines both with a weighted average (60% content + 40% collaborative) for better recommendations.

## ✨ Features
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Collaborative Filtering (KNN + User-Item Matrix)
- Hybrid System with adjustable weights
- Genre-based filtering
- Search with autocomplete
- CLI and Streamlit web interface
- Evaluation metrics (Precision@K, Recall@K)
- Visualizations (3 graphs)
- Model persistence (pickle)

## 📊 Dataset
**MovieLens Dataset** (ml-latest-small)
- 9,742 movies | 100,836 ratings | 610 users | Rating scale: 0.5-5.0

## 🛠 Tech Stack
Python | Pandas | NumPy | scikit-learn | Streamlit | Matplotlib | Seaborn

## 📁 Structure
├── Dataset/ (movie.csv, rating.csv, tag.csv)
├── models/ (saved .pkl files)
├── recommendation_system.ipynb
├── app.py
└── README.md


## 🔧 Installation
```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn streamlit

# Train models
jupyter notebook  # Run all cells

# Run web app
streamlit run app.py
🚀 Usage Examples
python
# Get recommendations
recommend_movie("Inception", top_n=5, content_weight=0.6)

# Search movies
search_movies("matrix", limit=10)

# Get popular movies
get_popular_movies(min_ratings=10, top_n=10)
📊 Evaluation Results
Precision@5: 0.234 (23.4% of recommendations were good)

Recall@5: 0.156 (Found 15.6% of all good movies)

F1-Score: 0.187

📈 Visualizations
Rating Distribution (how users rate movies)

Top 10 Most Rated Movies (popularity)

Genre Distribution (most common genres)

🖥 Sample Output

Recommendations for: Inception (2010)
Genres: action crime drama mystery sci-fi thriller

1. Shutter Island (2010) - Similarity: 46.1%
2. The Matrix (1999) - Similarity: 40.5%
3. Donnie Darko (2001) - Similarity: 37.0%


Built with Python and Machine Learning

