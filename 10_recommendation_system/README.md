# 🎬 Hybrid Movie Recommendation System

A production-ready **Hybrid Movie Recommendation System** that combines **Content-Based Filtering** and **Collaborative Filtering** to deliver personalized and accurate movie recommendations.

---

# 📖 Project Overview

This system implements a **hybrid recommendation engine** using two complementary approaches:

### 🎯 Content-Based Filtering
- Uses TF-IDF Vectorization
- Computes similarity based on movie genres, tags, and metadata
- Finds movies similar to a given movie

### 👥 Collaborative Filtering
- Uses User-Item interaction matrix
- Applies K-Nearest Neighbors (KNN)
- Finds movies liked by similar users

### ⚡ Hybrid Approach
Final recommendations are generated using a weighted combination:
- 60% Content-Based Filtering
- 40% Collaborative Filtering

This improves accuracy and reduces cold-start problems.

---

# ✨ Key Features

- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Collaborative Filtering (KNN-based User Similarity)
- Hybrid Recommendation Engine (Weighted System)
- Movie Search with Autocomplete Support
- Genre-based Filtering
- Streamlit Interactive Web App
- CLI-based Recommendation Interface
- Model Evaluation Metrics (Precision, Recall, F1-score)
- Data Visualizations (Rating & Genre Analysis)
- Model Persistence using Pickle

---

# 📊 Dataset Information

Dataset Used: MovieLens Latest Small Dataset

Movies: 9,742  
Ratings: 100,836  
Users: 610  
Rating Scale: 0.5 – 5.0  

---

# 🧠 System Architecture

1. Data Preprocessing  
   - Clean movie metadata  
   - Merge ratings, tags, and movie details  

2. Feature Engineering  
   - TF-IDF vectorization of movie metadata  
   - User-item interaction matrix  

3. Model Building  
   - Cosine Similarity for content-based filtering  
   - KNN for collaborative filtering  

4. Hybrid Recommendation  
   - Weighted combination of both models  

---

# 🛠 Tech Stack

Python  
Pandas  
NumPy  
Scikit-learn  
SciPy  
Matplotlib  
Seaborn  
Streamlit  

---

# 📁 Project Structure

├── Dataset/  
│   ├── movies.csv  
│   ├── ratings.csv  
│   └── tags.csv  
│  
├── models/  
│   ├── tfidf_model.pkl  
│   ├── similarity_matrix.pkl  
│   └── user_item_matrix.pkl  
│  
├── recommendation_system.ipynb  
├── app.py  
└── README.md  

---

# 🔧 Installation & Setup

## 1. Install Dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn streamlit

---

## 2. Train Models
jupyter notebook

Run all cells to generate trained models.

---

## 3. Run Streamlit App
streamlit run app.py

---

# 🚀 Usage Examples

## Get Movie Recommendations
recommend_movie("Inception", top_n=5, content_weight=0.6)

## Search Movies
search_movies("matrix", limit=10)

## Get Popular Movies
get_popular_movies(min_ratings=10, top_n=10)

---

# 📊 Evaluation Metrics

Precision@5: 0.234  
Recall@5: 0.156  
F1-Score: 0.187  

These metrics show balanced recommendation performance.

---

# 📈 Visualizations

- Rating Distribution Analysis  
- Top 10 Most Rated Movies  
- Genre Distribution Insights  

---

# 🖥 Sample Output

Recommendations for: Inception (2010)  
Genres: Action | Crime | Drama | Mystery | Sci-Fi | Thriller  

1. Shutter Island (2010) - Similarity: 46.1%  
2. The Matrix (1999) - Similarity: 40.5%  
3. Donnie Darko (2001) - Similarity: 37.0%  

---

# 🔮 Future Improvements

- Deep Learning embeddings (BERT / Transformers)  
- Real-time recommendation API (FastAPI)  
- User authentication system  
- Database integration (PostgreSQL / MongoDB)  
- Cloud deployment (AWS / GCP / Azure)  

---

# 👨‍💻 Author

Saif Ullah  
AI/ML Engineer | Machine Learning Enthusiast  

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub. It helps a lot.