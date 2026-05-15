# ============================================================
# PRODUCT RECOMMENDATION LOGIC
# ============================================================

# Import Libraries
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD SAVED FILES
# ============================================================

# Load TF-IDF Vectorizer
with open("app/model/tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

# Load Similarity Matrix
with open("app/model/similarity_matrix.pkl", "rb") as file:
    similarity_matrix = pickle.load(file)

# Load Dataframe
with open("app/model/products_dataframe.pkl", "rb") as file:
    df = pickle.load(file)


# ============================================================
# GET ALL PRODUCTS
# ============================================================

def get_all_products():
    
    return df.to_dict(orient="records")


# ============================================================
# SEARCH PRODUCTS
# ============================================================

def search_products(query):
    
    # Search Matching Products
    results = df[
        df['name'].str.lower().str.contains(query.lower())
    ]
    
    # Return Top 10 Matches
    return results.head(10).to_dict(orient="records")


# ============================================================
# RECOMMEND PRODUCTS
# ============================================================

def recommend_products(product_name, top_n=5):
    
    # Find Matching Products
    matches = df[
        df['name'].str.lower().str.contains(product_name.lower())
    ]
    
    # Handle Invalid Product
    if matches.empty:
        return {"error": "Product not found"}
    
    # Get Product Index
    product_index = matches.index[0]
    
    # Similarity Scores
    scores = list(enumerate(similarity_matrix[product_index]))
    
    # Sort Scores
    sorted_scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )
    
    # Remove Same Product
    sorted_scores = sorted_scores[1:top_n+1]
    
    # Store Recommendations
    recommendations = []
    
    for index, score in sorted_scores:
        
        recommendations.append({
            "name": df.iloc[index]['name'],
            "category": df.iloc[index]['main_category'],
            "sub_category": df.iloc[index]['sub_category'],
            "similarity_score": round(float(score), 2),
            "image": df.iloc[index]['image'],
            "link": df.iloc[index]['link']
        })
    
    return recommendations


# ============================================================
# USER PREFERENCE RECOMMENDATION
# ============================================================

def recommend_by_preferences(preferences, top_n=5):
    
    # Convert Preferences into Vector
    user_vector = tfidf.transform([preferences])
    
    # Compute Similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf.transform(df['combined_features'])
    )
    
    # Get Top Products
    top_indices = similarity_scores[0].argsort()[::-1][:top_n]
    
    # Store Results
    results = []
    
    for index in top_indices:
        
        results.append({
            "name": df.iloc[index]['name'],
            "category": df.iloc[index]['main_category'],
            "score": round(float(similarity_scores[0][index]), 2),
            "image": df.iloc[index]['image'],
            "link": df.iloc[index]['link']
        })
    
    return results