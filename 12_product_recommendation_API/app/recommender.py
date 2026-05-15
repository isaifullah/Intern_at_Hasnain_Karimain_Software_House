# ============================================================
# PRODUCT RECOMMENDATION LOGIC
# ============================================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("app/data/products.csv")


# ============================================================
# PREPROCESSING
# ============================================================

df = df.fillna("")

df["tags"] = (
    df["name"] + " " +
    df["main_category"] + " " +
    df["sub_category"]
)


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["tags"])


# ============================================================
# SIMILARITY MATRIX
# ============================================================

similarity_matrix = cosine_similarity(tfidf_matrix)


# ============================================================
# GET ALL PRODUCTS
# ============================================================

def get_all_products():

    return df[
        ["name", "main_category"]
    ].head(50).to_dict(orient="records")


# ============================================================
# SEARCH PRODUCTS
# ============================================================

def search_products(query):

    results = df[
        df["name"].str.contains(query, case=False)
    ]

    return results[
        ["name", "main_category"]
    ].head(10).to_dict(orient="records")


# ============================================================
# RECOMMEND PRODUCTS
# ============================================================

def recommend_products(product_name):

    matches = df[
        df["name"].str.contains(product_name, case=False)
    ]

    if matches.empty:
        return {"error": "Product not found"}

    index = matches.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[index])
    )

    sorted_products = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for item in sorted_products:

        product_index = item[0]

        recommendations.append({
            "name": df.iloc[product_index]["name"],
            "category": df.iloc[product_index]["main_category"],
            "score": round(float(item[1]), 2)
        })

    return recommendations


# ============================================================
# USER PREFERENCE RECOMMENDATION
# ============================================================

def recommend_by_preferences(preferences):

    results = df[
        df["tags"].str.contains(preferences, case=False)
    ]

    return results[
        ["name", "main_category"]
    ].head(10).to_dict(orient="records")